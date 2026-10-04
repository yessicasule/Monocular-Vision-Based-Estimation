"""
Simultaneous capture session: records raw video + logs joint angles frame-by-frame
while optionally streaming to Unity over UDP.

Output layout (under data/sessions/<session_name>/):
  raw.mp4              — raw camera recording
  angles.csv           — per-frame: raw + filtered angles, inference time, pose detected flag
  landmarks.csv        — optional (--save_landmarks): all 33 MediaPipe keypoints per frame
  <session>_angles.png — joint angle plot saved automatically at session end

After a session, run the existing benchmark pipeline on the saved video:
  python scripts/run_benchmark_all.py --video data/sessions/<name>/raw.mp4

Keyboard controls during capture:
  q / ESC  — stop session and save
  s        — toggle recording on/off
  p        — toggle live angle plot window
  c        — enter calibration mode (cycles arm_down → arm_forward → elbow_flexed)
  SPACE    — capture current calibration pose
  d        — show/hide debug overlay
"""

import argparse
import collections
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from config.config import Config
from src.pose.mediapipe_runner import MediaPipeRunner
from src.processing.angle_filter import AngleFilterBank
from src.pose import angle_landmarks
from src.processing.angle_solver import compute_arm_angles, ArmAngles
from src.processing.calibration import CalibrationManager
from src.streaming.udp_streamer import UdpAngleSender

CALIB_SEQUENCE = ["arm_down", "arm_forward", "arm_side", "elbow_bent"]

_JOINT_KEYS = ["shoulder_flexion", "shoulder_abduction", "shoulder_rotation", "elbow_flexion"]

_PLOT_COLOURS = {
    "elbow_flexion":       (92,  92,  224),   # BGR
    "shoulder_flexion":    (224, 158,  92),
    "shoulder_rotation":   (92,  224, 122),
    "shoulder_abduction":  (92,  192, 224),
}
_PLOT_LABELS = {
    "elbow_flexion":      "Elbow Flex",
    "shoulder_flexion":   "Shldr Flex",
    "shoulder_rotation":  "Shldr Rot",
    "shoulder_abduction": "Shldr Abd",
}


class _RollingPlot:
    """Lightweight OpenCV-based rolling angle plot — no matplotlib, no threads."""

    WINDOW = "Joint Angles (live)"
    _W, _H  = 560, 300          # plot canvas size
    _MARGIN = (36, 12, 28, 10)  # top, right, bottom, left
    _HISTORY = 200              # samples shown

    def __init__(self):
        self._buf = {k: collections.deque(maxlen=self._HISTORY) for k in _JOINT_KEYS}
        self._visible = False
        self._y_range = (-10.0, 190.0)

    def toggle(self):
        self._visible = not self._visible
        if not self._visible:
            cv2.destroyWindow(self.WINDOW)

    @property
    def visible(self):
        return self._visible

    def push(self, angles: dict):
        for k in _JOINT_KEYS:
            self._buf[k].append(float(angles.get(k, 0.0)))

    def _to_px(self, val, y_min, y_max, plot_h, plot_top):
        norm = (val - y_min) / max(y_max - y_min, 1e-6)
        return int(plot_top + plot_h * (1.0 - norm))

    def render(self):
        if not self._visible:
            return
        mt, mr, mb, ml = self._MARGIN
        plot_w = self._W - ml - mr
        plot_h = self._H - mt - mb

        canvas = np.full((self._H, self._W, 3), 25, dtype=np.uint8)

        # Grid lines at 0, 45, 90, 135, 180 degrees
        y_min, y_max = self._y_range
        for deg in (0, 45, 90, 135, 180):
            if y_min <= deg <= y_max:
                py = self._to_px(deg, y_min, y_max, plot_h, mt)
                cv2.line(canvas, (ml, py), (self._W - mr, py), (50, 50, 50), 1)
                cv2.putText(canvas, f"{deg}", (2, py + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, (120, 120, 120), 1)

        for k in _JOINT_KEYS:
            vals = list(self._buf[k])
            if len(vals) < 2:
                continue
            colour = _PLOT_COLOURS[k]
            n = len(vals)
            pts = []
            for i, v in enumerate(vals):
                x = ml + int(i / max(n - 1, 1) * plot_w)
                y = self._to_px(v, y_min, y_max, plot_h, mt)
                pts.append((x, y))
            pts_arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas, [pts_arr], False, colour, 1, cv2.LINE_AA)

            # Legend label at right edge
            last_y = pts[-1][1]
            cv2.putText(canvas, _PLOT_LABELS[k],
                        (self._W - mr - 2, max(mt + 6, min(last_y, self._H - mb - 2))),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, colour, 1, cv2.LINE_AA)

        # Axis border
        cv2.rectangle(canvas, (ml, mt), (self._W - mr, self._H - mb), (80, 80, 80), 1)
        cv2.putText(canvas, "deg", (2, mt - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, (120, 120, 120), 1)

        cv2.imshow(self.WINDOW, canvas)


def _open_camera(camera_id: int):
    backends = [
        (camera_id, cv2.CAP_MSMF),
        (camera_id, 0),
        (1, cv2.CAP_MSMF),
    ]
    for idx, backend in backends:
        cap = cv2.VideoCapture(idx, backend) if backend else cv2.VideoCapture(idx)
        if cap.isOpened():
            print(f"Camera opened: index={idx} backend={'MSMF' if backend == cv2.CAP_MSMF else 'auto'}")
            return cap
        cap.release()
    return None


def _make_session_dir(base: Path, session_name: str) -> Path:
    d = base / session_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _draw_overlay(frame, angles_raw, angles_filt, fps_actual, recording, calib_state, debug):
    h, w = frame.shape[:2]
    if recording:
        cv2.circle(frame, (w - 20, 20), 8, (0, 0, 220), -1)

    if not debug:
        return

    lines = [
        f"FPS: {fps_actual:.1f}  {'REC' if recording else 'PAUSED'}",
        f"Flex:  raw {angles_raw.shoulder_flexion:6.1f}  filt {angles_filt.shoulder_flexion:6.1f}",
        f"Abd:   raw {angles_raw.shoulder_abduction:6.1f}  filt {angles_filt.shoulder_abduction:6.1f}",
        f"Rot:   raw {angles_raw.shoulder_rotation:6.1f}  filt {angles_filt.shoulder_rotation:6.1f}",
        f"Elbow: raw {angles_raw.elbow_flexion:6.1f}  filt {angles_filt.elbow_flexion:6.1f}",
    ]
    if calib_state:
        lines.append(f"CALIB: Hold '{calib_state}' then press SPACE")

    for i, line in enumerate(lines):
        cv2.putText(frame, line, (8, 20 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 100), 1, cv2.LINE_AA)


def _open_angle_csv(session_dir: Path) -> tuple:
    path = session_dir / "angles.csv"
    f = open(path, "w", newline="", encoding="utf-8")
    writer = csv.writer(f)
    writer.writerow([
        "frame_index", "timestamp_unix", "elapsed_s", "inference_ms", "pose_detected",
        "shoulder_flexion_raw", "shoulder_abduction_raw", "shoulder_rotation_raw", "elbow_flexion_raw",
        "shoulder_flexion_filt", "shoulder_abduction_filt", "shoulder_rotation_filt", "elbow_flexion_filt",
    ])
    return f, writer, path


def _open_landmark_csv(session_dir: Path) -> tuple:
    path = session_dir / "landmarks.csv"
    f = open(path, "w", newline="", encoding="utf-8")
    header = ["frame_index", "timestamp_unix"]
    for i in range(33):
        header += [f"lm{i}_x", f"lm{i}_y", f"lm{i}_z", f"lm{i}_vis"]
    writer = csv.writer(f)
    writer.writerow(header)
    return f, writer, path


def main():
    ap = argparse.ArgumentParser(description="Simultaneous video capture + joint angle logging")
    ap.add_argument("--session", default=None, help="Session name (default: session_<unix>)")
    ap.add_argument("--camera", type=int, default=Config.CAMERA_ID)
    ap.add_argument("--host", default=Config.UDP_IP)
    ap.add_argument("--port", type=int, default=Config.UDP_PORT)
    ap.add_argument("--filter", default="kalman", choices=["kalman", "ema", "ma", "sg"],
                    help="Filter type: kalman (default), ema, ma (moving average), sg (Savitzky-Golay)")
    ap.add_argument("--no_stream", action="store_true", help="Disable UDP streaming to Unity")
    ap.add_argument("--save_landmarks", action="store_true", help="Save all 33 MediaPipe keypoints per frame")
    ap.add_argument("--calib", type=str, default=None, help="Load calibration JSON before starting")
    ap.add_argument("--width", type=int, default=Config.FRAME_WIDTH)
    ap.add_argument("--height", type=int, default=Config.FRAME_HEIGHT)
    args = ap.parse_args()

    session_name = args.session or f"session_{int(time.time())}"
    sessions_base = Path(__file__).resolve().parent.parent / "data" / "sessions"
    session_dir = _make_session_dir(sessions_base, session_name)
    print(f"Session directory: {session_dir}")

    cap = _open_camera(args.camera)
    if cap is None:
        print("ERROR: Could not open camera.")
        print("  Check: Settings > Privacy > Camera  |  No other app holding camera")
        print("  Try: --camera 1 or --camera 2")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    print("Warming up camera...")
    for _ in range(10):
        cap.read()
    time.sleep(0.5)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Resolution: {actual_w}x{actual_h}")

    # Video writer
    video_path = session_dir / "raw.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, Config.OUTPUT_VIDEO_FPS, (actual_w, actual_h))
    print(f"Recording to: {video_path}")

    # Pose + filter + calibration
    pose_runner = MediaPipeRunner(
        min_detection_confidence=Config.MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=Config.MIN_TRACKING_CONFIDENCE,
    )
    filter_sys = AngleFilterBank(filter_type=args.filter, stream_hz=Config.STREAM_HZ)
    calib_mgr = CalibrationManager()
    if args.calib:
        calib_path = Path(args.calib)
        if calib_path.exists():
            calib_mgr.load(calib_path)
            print(f"Calibration loaded from {calib_path}")

    # UDP stream (optional)
    streamer = None
    if not args.no_stream:
        streamer = UdpAngleSender(host=args.host, port=args.port, hz=Config.STREAM_HZ)
        streamer.start()
        print(f"Streaming to Unity at {args.host}:{args.port}")

    # CSV loggers
    angle_f, angle_writer, angle_path = _open_angle_csv(session_dir)
    lm_f = lm_writer = None
    if args.save_landmarks:
        lm_f, lm_writer, lm_path = _open_landmark_csv(session_dir)
        print(f"Saving landmarks to: {lm_path}")

    recording = True
    debug_overlay = True
    calib_idx = -1
    calib_state = None

    frame_index = 0
    session_t0 = time.perf_counter()
    fps_window = []
    consecutive_failures = 0

    rolling_plot = _RollingPlot()

    print("\nControls: [q/ESC] stop  [s] toggle recording  [p] angle plot  "
          "[c] calibrate  [SPACE] capture pose  [d] debug overlay")
    print("Session running...\n")

    ZERO_ANGLES = {k: 0.0 for k in _JOINT_KEYS}

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                consecutive_failures += 1
                if consecutive_failures >= 30:
                    print("Camera read failed repeatedly — stopping.")
                    break
                time.sleep(0.033)
                continue
            consecutive_failures = 0

            t_unix = time.time()
            elapsed = time.perf_counter() - session_t0

            # --- Pose estimation ---
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            t_infer = time.perf_counter()
            landmarks = pose_runner.process(image_rgb)
            inference_ms = (time.perf_counter() - t_infer) * 1000.0

            pose_detected = landmarks is not None
            raw_arm: ArmAngles | None = None
            filt_arm: ArmAngles | None = None

            if pose_detected:
                raw_arm = compute_arm_angles(angle_landmarks(
                    pose_runner, landmarks, image_rgb.shape[1], image_rgb.shape[0]))
                if raw_arm is not None:
                    cal_arm = calib_mgr.apply(raw_arm)
                    filt_arm = filter_sys.update(cal_arm)

            _zero = ArmAngles(0.0, 0.0, 0.0, 0.0)
            raw_angles = raw_arm or _zero
            filtered_angles = filt_arm or _zero

            # --- Record frame (when recording) ---
            if recording:
                writer.write(frame)

            # --- Log angles ---
            angle_writer.writerow([
                frame_index,
                f"{t_unix:.4f}",
                f"{elapsed:.4f}",
                f"{inference_ms:.2f}",
                int(pose_detected and raw_arm is not None),
                f"{raw_angles.shoulder_flexion:.3f}",
                f"{raw_angles.shoulder_abduction:.3f}",
                f"{raw_angles.shoulder_rotation:.3f}",
                f"{raw_angles.elbow_flexion:.3f}",
                f"{filtered_angles.shoulder_flexion:.3f}",
                f"{filtered_angles.shoulder_abduction:.3f}",
                f"{filtered_angles.shoulder_rotation:.3f}",
                f"{filtered_angles.elbow_flexion:.3f}",
            ])

            # --- Log landmarks (optional) ---
            if lm_writer and pose_detected:
                row = [frame_index, f"{t_unix:.4f}"]
                for lm in landmarks:
                    row += [f"{lm.x:.5f}", f"{lm.y:.5f}", f"{lm.z:.5f}", f"{lm.visibility:.3f}"]
                lm_writer.writerow(row)

            # --- Stream to Unity ---
            if streamer and filt_arm is not None:
                streamer.update(filt_arm)

            # --- FPS calculation ---
            fps_window.append(time.perf_counter())
            fps_window = [t for t in fps_window if fps_window[-1] - t < 1.0]
            fps_actual = len(fps_window)

            # --- Rolling plot ---
            rolling_plot.push(filtered_angles)
            rolling_plot.render()

            # --- Draw overlay ---
            display_frame = frame.copy()
            _draw_overlay(display_frame, raw_angles, filtered_angles, fps_actual, recording, calib_state, debug_overlay)
            cv2.imshow("Capture Session", display_frame)

            frame_index += 1

            # --- Key handling ---
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("s"):
                recording = not recording
                print(f"Recording {'ON' if recording else 'OFF'}")
            elif key == ord("p"):
                rolling_plot.toggle()
            elif key == ord("d"):
                debug_overlay = not debug_overlay
            elif key == ord("c"):
                calib_idx = (calib_idx + 1) % len(CALIB_SEQUENCE)
                calib_state = CALIB_SEQUENCE[calib_idx]
                calib_mgr.start_calibration_pose(calib_state)
            elif key == ord(" ") and calib_state and pose_detected:
                if calib_mgr.capture_pose(raw_angles):
                    print(f"  Captured: {calib_state} = {raw_angles}")
                    if calib_idx == len(CALIB_SEQUENCE) - 1:
                        calib_mgr.finalize_calibration()
                        calib_path = session_dir / "calibration.json"
                        calib_mgr.save(calib_path)
                        print(f"Calibration complete. Saved to {calib_path}")
                        calib_state = None
                    else:
                        calib_idx += 1
                        calib_state = CALIB_SEQUENCE[calib_idx]
                        calib_mgr.start_calibration_pose(calib_state)

    except KeyboardInterrupt:
        pass
    finally:
        print(f"\nShutting down — {frame_index} frames processed.")
        cap.release()
        writer.release()
        cv2.destroyAllWindows()
        pose_runner.close()
        angle_f.close()
        if lm_f:
            lm_f.close()
        if streamer:
            streamer.stop()

        print(f"\nSession saved to: {session_dir}")
        print(f"  Video:  {video_path.name}")
        print(f"  Angles: {angle_path.name}")
        if args.save_landmarks:
            print(f"  Landmarks: landmarks.csv")

        # Auto-generate angle plot from the CSV we just wrote
        try:
            import importlib.util, subprocess
            plot_script = Path(__file__).resolve().parent / "plot_angles.py"
            if plot_script.exists():
                print("\nGenerating angle plot...")
                subprocess.run(
                    [sys.executable, str(plot_script),
                     "--csv", str(angle_path),
                     "--comparison"],
                    check=False, timeout=30
                )
        except Exception as e:
            print(f"  (plot_angles.py skipped: {e})")

        frames_dir = session_dir / "frames"
        print(f"\nTo benchmark all frameworks on this session:")
        print(f"  1. Extract frames:")
        print(f"     python benchmarks/extract_frames.py --video {video_path} --out_dir {frames_dir}")
        print(f"  2. Run all framework benchmarks:")
        print(f"     python benchmarks/run_all_benchmarks.py --session_name {session_name} --frames_dir {frames_dir}")


if __name__ == "__main__":
    main()
