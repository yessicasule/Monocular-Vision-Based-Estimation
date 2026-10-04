"""
run_demo.py — MonoArm Live Pipeline Entry Point
================================================

Runs the complete real-time arm angle estimation pipeline:
    Camera → Pose Estimation → Angle Computation → Filtering
    → Calibration → UDP Send → CSV Logging → Live Display

Controls
--------
    Q  : Quit
    F  : Cycle filter type (kalman → ma → sg → kalman)
    C  : Enter calibration mode (static reference poses)
    L  : Toggle CSV logging
    R  : Reset filters

Usage
-----
    python scripts/run_demo.py
    python scripts/run_demo.py --camera 0 --port 9000 --filter kalman
    python scripts/run_demo.py --no-udp   (display only, no Unity)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import cv2
import numpy as np

from src.pose import load_estimator, PoseEstimator
from src.processing.coordinate_frame import build_torso_frame
from src.pose import angle_landmarks
from src.processing.angle_solver import compute_arm_angles
from src.processing.angle_filter import AngleFilterBank
from src.processing.calibration import CalibrationManager, REQUIRED_POSES
from src.processing.angle_logger import CsvAngleLogger, RollingAnglePlot
from src.streaming.udp_streamer import UdpAngleSender


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_GREEN  = (100, 220, 100)
_ORANGE = (80, 180, 255)
_RED    = (80,  80, 240)
_WHITE  = (230, 230, 230)
_GREY   = (140, 140, 140)
_BLACK  = (0, 0, 0)


def _draw_overlay(
    frame: np.ndarray,
    landmarks,
    angles,
    fps: float,
    filter_type: str,
    logging: bool,
    calibrated: bool,
    rot_reliable: bool,
) -> None:
    """Draw keypoint skeleton, angle HUD, and status bar onto the frame."""
    h, w = frame.shape[:2]

    # Draw right-arm skeleton
    if landmarks is not None:
        CONNECTIONS = [(12, 14), (14, 16)]   # right shoulder-elbow-wrist
        KEY_INDICES = [11, 12, 13, 14, 15, 16, 23, 24]
        for idx in KEY_INDICES:
            lm = landmarks[idx]
            cx, cy = int(lm.x * w), int(lm.y * h)
            color = _GREEN if idx in (12, 14, 16) else _GREY
            cv2.circle(frame, (cx, cy), 5, color, -1, cv2.LINE_AA)
        for a, b in CONNECTIONS:
            lmA, lmB = landmarks[a], landmarks[b]
            pt1 = (int(lmA.x * w), int(lmA.y * h))
            pt2 = (int(lmB.x * w), int(lmB.y * h))
            cv2.line(frame, pt1, pt2, _GREEN, 2, cv2.LINE_AA)

    # Angle HUD (top-left panel)
    panel_h = 130
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (270, panel_h), _BLACK, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    if angles is not None:
        lines = [
            (f"Flex  {angles.shoulder_flexion:+7.1f} deg", _GREEN),
            (f"Abd   {angles.shoulder_abduction:+7.1f} deg", _ORANGE),
            (f"Rot   {angles.shoulder_rotation:+7.1f} deg{'  *' if not rot_reliable else ''}", _WHITE),
            (f"Elbow {angles.elbow_flexion:+7.1f} deg", (220, 80, 220)),
        ]
        for i, (text, color) in enumerate(lines):
            cv2.putText(frame, text, (8, 20 + i * 27), _FONT, 0.55, color, 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, "No pose detected", (8, 40), _FONT, 0.55, _RED, 1, cv2.LINE_AA)

    # Status bar (bottom)
    bar_y = h - 28
    status_parts = [
        f"FPS:{fps:4.1f}",
        f"Filter:{filter_type}",
        f"{'CAL' if calibrated else 'UNCAL'}",
        f"{'REC' if logging else '---'}",
        f"{'ROT-UNRELIABLE' if not rot_reliable else ''}",
    ]
    status = "  ".join(p for p in status_parts if p)
    cv2.rectangle(frame, (0, bar_y - 5), (w, h), _BLACK, -1)
    cv2.putText(frame, status, (6, h - 8), _FONT, 0.40, _GREY, 1, cv2.LINE_AA)
    cv2.putText(frame, "Q=Quit  F=Filter  C=Calib  L=Log  R=Reset",
                (6, bar_y - 8), _FONT, 0.38, _GREY, 1, cv2.LINE_AA)


def _run_calibration_session(
    cap: cv2.VideoCapture,
    runner: PoseEstimator,
    mgr: CalibrationManager,
    save_path: Path,
) -> bool:
    """
    Interactive calibration session.
    Returns True if calibration was completed successfully.
    """
    pose_instructions = {
        "arm_down":    "Let your right arm hang naturally at your side.",
        "arm_forward": "Raise your right arm straight forward (shoulder height).",
        "arm_side":    "Raise your right arm straight out to the right side.",
        "elbow_bent":  "Bend your right elbow 90 degrees, upper arm at side.",
    }

    mgr.begin_static()
    sample_counts: dict[str, list] = {p: [] for p in REQUIRED_POSES}

    pose_idx = 0
    current_pose = REQUIRED_POSES[pose_idx]
    collecting = False
    collect_start = 0.0
    COLLECT_S = 2.0  # collect for 2 seconds per pose
    COLLECT_HZ = 30.0

    print(f"\n[calibration] Starting. Hold each pose for {COLLECT_S:.0f}s when prompted.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        lms   = runner.process(rgb)

        instr = pose_instructions.get(current_pose, "")
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 80), _BLACK, -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.putText(frame, f"Calibration [{pose_idx+1}/{len(REQUIRED_POSES)}]: {current_pose}",
                    (10, 25), _FONT, 0.6, _GREEN, 1, cv2.LINE_AA)
        cv2.putText(frame, instr, (10, 50), _FONT, 0.5, _WHITE, 1, cv2.LINE_AA)

        if collecting:
            elapsed = time.perf_counter() - collect_start
            progress = min(elapsed / COLLECT_S, 1.0)
            bar_w = int(frame.shape[1] * progress)
            cv2.rectangle(frame, (0, frame.shape[0] - 8), (bar_w, frame.shape[0]), _GREEN, -1)

            if lms is not None:
                angles = compute_arm_angles(
                    angle_landmarks(runner, lms, frame.shape[1], frame.shape[0]))
                if angles is not None:
                    sample_counts[current_pose].append(angles)

            if elapsed >= COLLECT_S:
                # Average samples
                samples = sample_counts[current_pose]
                if samples:
                    from src.processing.angle_solver import ArmAngles
                    avg = ArmAngles(
                        shoulder_flexion   = float(np.mean([s.shoulder_flexion   for s in samples])),
                        shoulder_abduction = float(np.mean([s.shoulder_abduction for s in samples])),
                        shoulder_rotation  = float(np.mean([s.shoulder_rotation  for s in samples])),
                        elbow_flexion      = float(np.mean([s.elbow_flexion      for s in samples])),
                        rotation_reliable  = samples[-1].rotation_reliable,
                    )
                    mgr._current_pose = current_pose
                    mgr.capture_pose(avg)
                    print(f"  [✓] {current_pose}: flex={avg.shoulder_flexion:.1f}  "
                          f"abd={avg.shoulder_abduction:.1f}  "
                          f"rot={avg.shoulder_rotation:.1f}  "
                          f"elb={avg.elbow_flexion:.1f}")

                pose_idx += 1
                collecting = False
                if pose_idx >= len(REQUIRED_POSES):
                    break
                current_pose = REQUIRED_POSES[pose_idx]
        else:
            cv2.putText(frame, "Press SPACE to capture this pose.",
                        (10, 72), _FONT, 0.45, _ORANGE, 1, cv2.LINE_AA)

        cv2.imshow("MonoArm — Calibration", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" ") and not collecting:
            collecting     = True
            collect_start  = time.perf_counter()
        elif key == ord("q"):
            cv2.destroyWindow("MonoArm — Calibration")
            return False

    cv2.destroyWindow("MonoArm — Calibration")
    mgr.finalise()
    mgr.save(save_path)
    print(f"[calibration] Complete. Saved to {save_path}")
    return True


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="MonoArm live arm angle estimation.")
    parser.add_argument("--camera",   type=int, default=0)
    parser.add_argument("--port",     type=int, default=9000)
    parser.add_argument("--host",     default="127.0.0.1")
    parser.add_argument("--hz",       type=float, default=30.0)
    parser.add_argument("--filter",   choices=["kalman", "ma", "sg"], default="kalman")
    parser.add_argument("--calib",    default="calibration.json",
                        help="Path to calibration file (auto-loaded if exists)")
    parser.add_argument("--no-udp",   action="store_true", help="Disable UDP transmission")
    parser.add_argument("--framework", choices=["mediapipe", "movenet_lightning", "movenet_thunder", "posenet"], default="mediapipe")
    parser.add_argument("--log-dir",  default="outputs/logs")
    args = parser.parse_args()

    from config.config import Config
    Config.ensure_directories()

    # --- Setup ---
    runner   = load_estimator(args.framework)
    filt     = AngleFilterBank(filter_type=args.filter, stream_hz=args.hz)
    calib_mgr = CalibrationManager()
    logger   = CsvAngleLogger(output_dir=args.log_dir, filter_type=args.filter)
    plot     = RollingAnglePlot(width=640, height=180)

    calib_path = Path(args.calib)
    if calib_path.exists():
        calib_mgr.load(calib_path)
        print(f"[run_demo] Loaded calibration from {calib_path}")

    sender: UdpAngleSender | None = None
    if not args.no_udp:
        sender = UdpAngleSender(host=args.host, port=args.port, hz=args.hz)
        sender.start()
        print(f"[run_demo] UDP streaming to {args.host}:{args.port}")

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("[run_demo] Error: Cannot open camera.")
        sys.exit(1)

    logging_active = False
    filter_cycle   = ["kalman", "ma", "sg"]
    filter_idx     = filter_cycle.index(args.filter)
    fps_buf        = []
    t_prev         = time.perf_counter()

    print("[run_demo] Running. Q=Quit  F=Filter  C=Calib  L=Log  R=Reset")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # FPS estimate
            now   = time.perf_counter()
            fps_buf.append(1.0 / max(now - t_prev, 1e-6))
            t_prev = now
            if len(fps_buf) > 30:
                fps_buf.pop(0)
            fps = float(np.mean(fps_buf))

            # Pose + angles
            lms    = runner.process(rgb)
            angles = (compute_arm_angles(angle_landmarks(runner, lms, rgb.shape[1], rgb.shape[0]))
                      if lms is not None else None)

            raw_angles = angles
            if angles is not None:
                angles = filt.update(angles)
                angles = calib_mgr.apply(angles)
                if sender is not None:
                    sender.update(angles)
                if logging_active:
                    logger.log(angles)
                plot.update(angles)

            # Draw overlay
            _draw_overlay(
                frame,
                lms,
                angles,
                fps,
                filter_cycle[filter_idx],
                logging_active,
                calib_mgr.data.calibrated,
                raw_angles.rotation_reliable if raw_angles else False,
            )

            # Compose plot below frame
            plot_img = plot.render()
            frame_h, frame_w = frame.shape[:2]
            plot_resized = cv2.resize(plot_img, (frame_w, 180))
            combined = np.vstack([frame, plot_resized])

            cv2.imshow("MonoArm — Live", combined)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("f"):
                filter_idx = (filter_idx + 1) % len(filter_cycle)
                new_type   = filter_cycle[filter_idx]
                filt       = AngleFilterBank(filter_type=new_type, stream_hz=args.hz)
                logger._filter = new_type
                print(f"[run_demo] Filter → {new_type}")
            elif key == ord("c"):
                if logging_active:
                    logger.stop()
                    logging_active = False
                _run_calibration_session(cap, runner, calib_mgr, calib_path)
            elif key == ord("l"):
                if logging_active:
                    logger.stop()
                    logging_active = False
                    print("[run_demo] Logging stopped.")
                else:
                    log_path = logger.start(calibrated=calib_mgr.data.calibrated)
                    logging_active = True
                    print(f"[run_demo] Logging → {log_path}")
            elif key == ord("r"):
                filt.reset()
                print("[run_demo] Filters reset.")

    except KeyboardInterrupt:
        pass
    finally:
        if logging_active:
            logger.stop()
        if sender is not None:
            sender.stop()
        cap.release()
        runner.close()
        cv2.destroyAllWindows()
        print("[run_demo] Exited cleanly.")


if __name__ == "__main__":
    main()
