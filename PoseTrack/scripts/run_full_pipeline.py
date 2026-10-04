"""
run_full_pipeline.py — MonoArm Full End-to-End Pipeline
========================================================

Unified script that:
  1. Captures from a real-time webcam OR processes a pre-recorded video
  2. Runs MediaPipe as the primary framework (streamed to Unity via UDP)
  3. Simultaneously runs MoveNet and PoseNet for side-by-side comparison
  4. Displays a 3-panel skeleton overlay + rolling angle plots
  5. Saves the comparison session as a video, CSV logs, and summary JSON

Usage
-----
    # Live webcam (default) with Unity streaming
    python scripts/run_full_pipeline.py

    # Pre-recorded video
    python scripts/run_full_pipeline.py --video path/to/clip.mp4

    # Save the side-by-side comparison as a video file
    python scripts/run_full_pipeline.py --video clip.mp4 --save_video

    # Disable UDP (display-only, no Unity)
    python scripts/run_full_pipeline.py --no_udp

Controls
--------
    Q  : Quit
    F  : Cycle filter type (kalman -> ma -> sg)
    C  : Enter calibration mode (MediaPipe only)
    L  : Toggle CSV logging
    R  : Reset all filters
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from src.pose import load_estimator, PoseEstimator
from src.pose import angle_landmarks
from src.processing.angle_solver import compute_arm_angles, compute_bilateral_angles, ArmAngles, BilateralArmAngles
from src.processing.angle_filter import BilateralFilterBank
from src.processing.calibration import CalibrationManager, REQUIRED_POSES
from src.processing.angle_logger import CsvAngleLogger, RollingAnglePlot
from src.streaming.udp_streamer import UdpAngleSender

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ALL_FRAMEWORKS = ["mediapipe", "movenet_lightning", "posenet"]
PRIMARY_FRAMEWORK = "mediapipe"

_FONT   = cv2.FONT_HERSHEY_SIMPLEX
_WHITE  = (230, 230, 230)
_GREY   = (140, 140, 140)
_BLACK  = (0, 0, 0)
_RED    = (80, 80, 240)
_ORANGE = (80, 180, 255)

FRAMEWORK_COLORS = {
    "mediapipe":         (100, 220, 100),   # green
    "movenet_lightning": (255, 160, 100),   # blue-orange
    "movenet_thunder":   (255, 160, 100),
    "posenet":           (220, 80, 220),    # purple
}
FRAMEWORK_DISPLAY_NAMES = {
    "mediapipe":         "MediaPipe",
    "movenet_lightning": "MoveNet-Lightning",
    "movenet_thunder":   "MoveNet-Thunder",
    "posenet":           "PoseNet",
}

# MediaPipe bilateral arm connections for skeleton overlay
_RIGHT_ARM_CONNECTIONS = [(12, 14), (14, 16)]
_LEFT_ARM_CONNECTIONS  = [(11, 13), (13, 15)]
_KEY_INDICES     = [11, 12, 13, 14, 15, 16, 23, 24]
_LEFT_COLOR = (255, 220, 120)   # accent color so the left arm reads distinctly from the framework color


# ---------------------------------------------------------------------------
# Framework loader with graceful fallback
# ---------------------------------------------------------------------------

def load_frameworks(names: list[str]) -> tuple[dict[str, PoseEstimator], dict[str, str]]:
    """
    Load requested frameworks, skipping any that fail to import.

    Returns (runners, failures) where failures maps framework name to the
    error message so the UI can show WHY a panel is unavailable.
    """
    import traceback
    runners: dict[str, PoseEstimator] = {}
    failures: dict[str, str] = {}
    for name in names:
        try:
            runner = load_estimator(name)
            runners[name] = runner
            print(f"  [OK] {runner.name} loaded")
        except Exception as e:
            failures[name] = str(e)
            print(f"  [SKIP] {name}: {e}")
            traceback.print_exc()
    return runners, failures


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_skeleton(
    img: np.ndarray,
    landmarks: list | None,
    color: tuple,
    label: str = "",
) -> None:
    """Draw bilateral (both arms) skeleton overlay on a frame panel."""
    h, w = img.shape[:2]
    if landmarks is None:
        cv2.putText(img, "NO POSE", (w // 2 - 40, h // 2),
                    _FONT, 0.6, _RED, 2, cv2.LINE_AA)
        return

    for idx in _KEY_INDICES:
        try:
            lm = landmarks[idx]
            cx, cy = int(lm.x * w), int(lm.y * h)
            is_right_arm = idx in (12, 14, 16)
            is_left_arm  = idx in (11, 13, 15)
            c = color if is_right_arm else (_LEFT_COLOR if is_left_arm else _GREY)
            radius = 6 if (is_right_arm or is_left_arm) else 4
            cv2.circle(img, (cx, cy), radius, c, -1, cv2.LINE_AA)
        except (IndexError, AttributeError):
            pass

    for a, b in _RIGHT_ARM_CONNECTIONS:
        try:
            lmA, lmB = landmarks[a], landmarks[b]
            pt1 = (int(lmA.x * w), int(lmA.y * h))
            pt2 = (int(lmB.x * w), int(lmB.y * h))
            cv2.line(img, pt1, pt2, color, 3, cv2.LINE_AA)
        except (IndexError, AttributeError):
            pass

    for a, b in _LEFT_ARM_CONNECTIONS:
        try:
            lmA, lmB = landmarks[a], landmarks[b]
            pt1 = (int(lmA.x * w), int(lmA.y * h))
            pt2 = (int(lmB.x * w), int(lmB.y * h))
            cv2.line(img, pt1, pt2, _LEFT_COLOR, 3, cv2.LINE_AA)
        except (IndexError, AttributeError):
            pass


def draw_angle_hud(
    img: np.ndarray,
    angles: BilateralArmAngles | None,
    framework_name: str,
    color: tuple,
    fps: float | None = None,
    is_primary: bool = False,
) -> None:
    """Draw the bilateral (right + left) angle values HUD onto a panel."""
    h, w = img.shape[:2]

    # Framework name header
    tag = "[PRIMARY + UDP]" if is_primary else "[COMPARE]"
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 30), _BLACK, -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    cv2.putText(img, f"{framework_name}  {tag}",
                (6, 20), _FONT, 0.5, color, 1, cv2.LINE_AA)

    if fps is not None:
        cv2.putText(img, f"FPS:{fps:.1f}",
                    (w - 80, 20), _FONT, 0.4, _GREY, 1, cv2.LINE_AA)

    right = angles.right if angles else None
    left  = angles.left  if angles else None

    # Angle values panel
    if right is not None or left is not None:
        panel_y = h - 110
        overlay2 = img.copy()
        cv2.rectangle(overlay2, (0, panel_y), (w, h), _BLACK, -1)
        cv2.addWeighted(overlay2, 0.5, img, 0.5, 0, img)

        def _side_lines(a: ArmAngles | None, prefix: str) -> list[str]:
            if a is None:
                return [f"{prefix} --"]
            return [
                f"{prefix} Flex {a.shoulder_flexion:+6.1f}  Abd {a.shoulder_abduction:+6.1f}",
                f"{prefix} Rot  {a.shoulder_rotation:+6.1f}  Elb {a.elbow_flexion:+6.1f}",
            ]

        lines = _side_lines(right, "R") + _side_lines(left, "L")
        colors = [(100, 220, 100), (100, 160, 255), _LEFT_COLOR, _LEFT_COLOR]
        for i, text in enumerate(lines):
            cv2.putText(img, text, (6, panel_y + 20 + i * 22),
                        _FONT, 0.42, colors[i % len(colors)], 1, cv2.LINE_AA)
    else:
        cv2.putText(img, "No pose detected", (6, h - 20),
                    _FONT, 0.5, _RED, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Calibration session (reused from run_demo.py logic)
# ---------------------------------------------------------------------------

def run_calibration_session(
    cap: cv2.VideoCapture,
    runner: PoseEstimator,
    mgr: CalibrationManager,
    save_path: Path,
) -> bool:
    """Interactive calibration (same as run_demo.py). Returns True on success."""
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
    COLLECT_S = 2.0

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
                    (10, 25), _FONT, 0.6, (100, 220, 100), 1, cv2.LINE_AA)
        cv2.putText(frame, instr, (10, 50), _FONT, 0.5, _WHITE, 1, cv2.LINE_AA)

        if collecting:
            elapsed = time.perf_counter() - collect_start
            progress = min(elapsed / COLLECT_S, 1.0)
            bar_w = int(frame.shape[1] * progress)
            cv2.rectangle(frame, (0, frame.shape[0] - 8), (bar_w, frame.shape[0]),
                          (100, 220, 100), -1)

            if lms is not None:
                angles = compute_arm_angles(
                    angle_landmarks(runner, lms, frame.shape[1], frame.shape[0]))
                if angles is not None:
                    sample_counts[current_pose].append(angles)

            if elapsed >= COLLECT_S:
                samples = sample_counts[current_pose]
                if samples:
                    avg = ArmAngles(
                        shoulder_flexion   = float(np.mean([s.shoulder_flexion   for s in samples])),
                        shoulder_abduction = float(np.mean([s.shoulder_abduction for s in samples])),
                        shoulder_rotation  = float(np.mean([s.shoulder_rotation  for s in samples])),
                        elbow_flexion      = float(np.mean([s.elbow_flexion      for s in samples])),
                        rotation_reliable  = samples[-1].rotation_reliable,
                    )
                    mgr._current_pose = current_pose
                    mgr.capture_pose(avg)
                    print(f"  [OK] {current_pose}: flex={avg.shoulder_flexion:.1f}  "
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

        cv2.imshow("MonoArm -- Calibration", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" ") and not collecting:
            collecting    = True
            collect_start = time.perf_counter()
        elif key == ord("q"):
            cv2.destroyWindow("MonoArm -- Calibration")
            return False

    cv2.destroyWindow("MonoArm -- Calibration")
    mgr.finalise()
    mgr.save(save_path)
    print(f"[calibration] Complete. Saved to {save_path}")
    return True


# ---------------------------------------------------------------------------
# Per-framework state container
# ---------------------------------------------------------------------------

class FrameworkState:
    """Holds per-framework processing state."""
    def __init__(self, name: str, runner: PoseEstimator, filter_type: str, hz: float):
        self.name = name
        self.display_name = FRAMEWORK_DISPLAY_NAMES.get(name, name)
        self.color = FRAMEWORK_COLORS.get(name, _WHITE)
        self.runner = runner
        self.filt = BilateralFilterBank(filter_type=filter_type, stream_hz=hz)
        self.landmarks = None
        self.raw_angles: BilateralArmAngles | None = None
        self.filt_angles: BilateralArmAngles | None = None
        self.inference_ms: float = 0.0
        self.detection_count = 0
        self.frame_count = 0
        self.total_inference_ms = 0.0

    def process_frame(self, rgb: np.ndarray) -> None:
        """Run inference + bilateral angle solver + filter for one frame."""
        self.frame_count += 1
        t0 = time.perf_counter()
        self.landmarks = self.runner.process(rgb)
        self.inference_ms = (time.perf_counter() - t0) * 1000.0
        self.total_inference_ms += self.inference_ms

        if self.landmarks is not None:
            self.raw_angles = compute_bilateral_angles(
                angle_landmarks(self.runner, self.landmarks, rgb.shape[1], rgb.shape[0]))
            if self.raw_angles is not None and (self.raw_angles.right or self.raw_angles.left):
                self.detection_count += 1
                self.filt_angles = self.filt.update(self.raw_angles)
            else:
                self.filt_angles = None
        else:
            self.raw_angles = None
            self.filt_angles = None

    def reset_filter(self, filter_type: str, hz: float) -> None:
        self.filt = BilateralFilterBank(filter_type=filter_type, stream_hz=hz)

    @property
    def avg_fps(self) -> float:
        if self.frame_count == 0:
            return 0.0
        avg_ms = self.total_inference_ms / self.frame_count
        return 1000.0 / avg_ms if avg_ms > 0 else 0.0

    @property
    def detection_rate(self) -> float:
        return self.detection_count / self.frame_count if self.frame_count > 0 else 0.0

    def summary_dict(self) -> dict:
        return {
            "framework": self.name,
            "display_name": self.display_name,
            "total_frames": self.frame_count,
            "detections": self.detection_count,
            "detection_rate": round(self.detection_rate, 4),
            "avg_fps": round(self.avg_fps, 1),
            "avg_inference_ms": round(self.total_inference_ms / max(self.frame_count, 1), 2),
        }


# ---------------------------------------------------------------------------
# CSV logger for multi-framework comparison
# ---------------------------------------------------------------------------

class MultiFrameworkLogger:
    """Logs per-frame angles from all frameworks into separate CSV files."""

    def __init__(self, output_dir: Path, framework_names: list[str]):
        self._dir = output_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._writers: dict[str, tuple] = {}  # name -> (file, csv.writer)
        self._frame = 0
        self._start_time = time.perf_counter()

        for name in framework_names:
            path = self._dir / f"angles_{name}.csv"
            f = open(path, "w", newline="", encoding="utf-8")
            w = csv.writer(f)
            w.writerow([
                "frame", "elapsed_s",
                "right_shoulder_flexion", "right_shoulder_abduction",
                "right_shoulder_rotation", "right_elbow_flexion",
                "left_shoulder_flexion", "left_shoulder_abduction",
                "left_shoulder_rotation", "left_elbow_flexion",
                "right_detected", "left_detected",
            ])
            self._writers[name] = (f, w)

    def log(self, states: dict[str, FrameworkState]) -> None:
        elapsed = time.perf_counter() - self._start_time

        def _fields(a: ArmAngles | None) -> list:
            if a is None:
                return [0, 0, 0, 0]
            return [f"{a.shoulder_flexion:.2f}", f"{a.shoulder_abduction:.2f}",
                    f"{a.shoulder_rotation:.2f}", f"{a.elbow_flexion:.2f}"]

        for name, (f, w) in self._writers.items():
            st = states.get(name)
            bilateral = st.filt_angles if st else None
            right = bilateral.right if bilateral else None
            left  = bilateral.left  if bilateral else None
            w.writerow([
                self._frame, f"{elapsed:.3f}",
                *_fields(right), *_fields(left),
                int(right is not None), int(left is not None),
            ])
        self._frame += 1

    def close(self) -> None:
        for f, w in self._writers.values():
            f.flush()
            f.close()
        self._writers.clear()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="MonoArm Full Pipeline: Multi-framework comparison + Unity streaming."
    )
    src_group = parser.add_mutually_exclusive_group()
    src_group.add_argument("--camera", type=int, default=0,
                           help="Camera device index (default: 0)")
    src_group.add_argument("--video", type=str, default=None,
                           help="Path to a pre-recorded video file")

    parser.add_argument("--port",     type=int,   default=9000)
    parser.add_argument("--host",     type=str,   default="127.0.0.1")
    parser.add_argument("--hz",       type=float, default=30.0)
    parser.add_argument("--filter",   choices=["kalman", "ma", "sg"], default="kalman")
    parser.add_argument("--calib",    default="calibration.json",
                        help="Path to calibration file (auto-loaded if exists)")
    parser.add_argument("--no_udp",   action="store_true",
                        help="Disable UDP transmission to Unity")
    parser.add_argument("--save_video", action="store_true",
                        help="Save the comparison display as an MP4 video")
    parser.add_argument("--frameworks", nargs="+", default=ALL_FRAMEWORKS,
                        choices=ALL_FRAMEWORKS + ["movenet_thunder"],
                        help="Frameworks to run (default: all three)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (auto-generated if not set)")
    args = parser.parse_args()

    from config.config import Config
    Config.ensure_directories()

    # ── Output directory ──────────────────────────────────────────────────
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_dir = Path("outputs") / f"pipeline_session_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load frameworks ───────────────────────────────────────────────────
    print("\n==========================================================")
    print("  MonoArm Full Pipeline — Multi-Framework + Unity Streaming")
    print("==========================================================")
    print("\n[->] Loading frameworks...")

    # Ensure mediapipe is always first (primary)
    fw_names = list(dict.fromkeys([PRIMARY_FRAMEWORK] + args.frameworks))
    runners, fw_failures = load_frameworks(fw_names)

    if PRIMARY_FRAMEWORK not in runners:
        print(f"[FAIL] Primary framework '{PRIMARY_FRAMEWORK}' failed to load. Cannot continue.")
        sys.exit(1)

    # Build per-framework state
    states: dict[str, FrameworkState] = {}
    for name, runner in runners.items():
        states[name] = FrameworkState(name, runner, args.filter, args.hz)

    active_fw_names = list(states.keys())
    n_panels = len(active_fw_names)
    print(f"\n[OK] {n_panels} framework(s) active: {', '.join(active_fw_names)}")

    # ── Calibration ───────────────────────────────────────────────────────
    calib_mgr = CalibrationManager()
    calib_path = Path(args.calib)
    if calib_path.exists():
        calib_mgr.load(calib_path)
        print(f"[OK] Loaded calibration from {calib_path}")

    # ── UDP sender ────────────────────────────────────────────────────────
    sender: UdpAngleSender | None = None
    if not args.no_udp:
        # verbose=False: TX status is shown in this script's own console
        # progress line instead of the sender's periodic log.
        sender = UdpAngleSender(host=args.host, port=args.port, hz=args.hz,
                                bilateral=True, verbose=False)
        sender.start()
        print(f"[OK] UDP streaming to {args.host}:{args.port} at {args.hz} Hz "
              f"(MediaPipe bilateral angles -> Unity avatar)")

    # ── Video capture ─────────────────────────────────────────────────────
    if args.video:
        vpath = Path(args.video)
        if not vpath.exists():
            print(f"[FAIL] Video not found: {vpath}")
            sys.exit(1)
        cap = cv2.VideoCapture(str(vpath))
        source_label = f"Video: {vpath.name}"
        print(f"[OK] Input: {vpath}")
    else:
        cap = cv2.VideoCapture(args.camera)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        source_label = f"Camera: {args.camera}"
        print(f"[OK] Input: camera {args.camera}")

    if not cap.isOpened():
        print("[FAIL] Cannot open video source.")
        sys.exit(1)

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # ── Rolling angle plot (primary framework) ────────────────────────────
    plot = RollingAnglePlot(width=640, height=180)

    # ── Video writer (optional) ───────────────────────────────────────────
    writer: cv2.VideoWriter | None = None
    comparison_video_path = out_dir / "comparison_video.mp4"

    # ── Multi-framework CSV logger ────────────────────────────────────────
    mf_logger = MultiFrameworkLogger(out_dir, active_fw_names)
    logging_active = True

    # ── Filter cycling state ──────────────────────────────────────────────
    filter_cycle = ["kalman", "ma", "sg"]
    filter_idx   = filter_cycle.index(args.filter)

    # ── FPS tracking ──────────────────────────────────────────────────────
    fps_buf = deque(maxlen=30)
    t_prev  = time.perf_counter()

    # ── Panel dimensions ──────────────────────────────────────────────────
    PANEL_W = 320
    PANEL_H = 240
    PLOT_H  = 180

    print(f"\n>>> PIPELINE RUNNING. Press Q to quit, F=Filter, C=Calib, L=Log, R=Reset")
    print("-" * 70)

    total_frames = 0

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                if args.video:
                    print("\n[OK] End of video reached.")
                break

            # Flip for webcam (mirror), not for video files
            if not args.video:
                frame_bgr = cv2.flip(frame_bgr, 1)

            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            total_frames += 1

            # FPS estimate
            now = time.perf_counter()
            fps_buf.append(1.0 / max(now - t_prev, 1e-6))
            t_prev = now
            overall_fps = float(np.mean(fps_buf))

            # ── Process each framework ────────────────────────────────────
            for name, st in states.items():
                st.process_frame(rgb)

            # ── Apply calibration to primary framework ────────────────────
            # Only MediaPipe (calibrated + filtered — the most accurate
            # output) is streamed to Unity; the other frameworks are for
            # on-screen comparison and CSV logging only. Calibration is
            # right-arm-only (CalibrationManager predates bilateral support);
            # the left arm is streamed filtered but uncalibrated.
            primary = states[PRIMARY_FRAMEWORK]
            if primary.filt_angles is not None and primary.filt_angles.right is not None:
                calibrated_right = calib_mgr.apply(primary.filt_angles.right)
                if sender is not None:
                    sender.update_bilateral(BilateralArmAngles(
                        right=calibrated_right, left=primary.filt_angles.left,
                    ))
                # Update rolling plot (right arm only)
                plot.update(calibrated_right)

            # ── Build comparison display ──────────────────────────────────
            panels = []
            for name in active_fw_names:
                st = states[name]
                # Resize frame to panel size
                panel = cv2.resize(frame_bgr, (PANEL_W, PANEL_H))
                # Draw skeleton
                draw_skeleton(panel, st.landmarks, st.color, st.display_name)
                # Draw angle HUD
                is_primary = (name == PRIMARY_FRAMEWORK)
                fw_fps = 1000.0 / st.inference_ms if st.inference_ms > 0 else 0.0
                draw_angle_hud(
                    panel, st.filt_angles,
                    st.display_name, st.color,
                    fps=fw_fps, is_primary=is_primary,
                )
                panels.append(panel)

            # Pad panels to always have at least 3, naming the framework
            # that failed to load and why (see console for full traceback).
            failed_iter = iter(fw_failures.items())
            while len(panels) < 3:
                blank = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
                name, reason = next(failed_iter, ("", ""))
                title = (FRAMEWORK_DISPLAY_NAMES.get(name, name) + ": load failed"
                         if name else "Not Available")
                cv2.putText(blank, title, (10, PANEL_H // 2 - 12),
                            _FONT, 0.5, _GREY, 1, cv2.LINE_AA)
                if reason:
                    cv2.putText(blank, reason[:44], (10, PANEL_H // 2 + 12),
                                _FONT, 0.35, _GREY, 1, cv2.LINE_AA)
                    cv2.putText(blank, "(see console traceback)", (10, PANEL_H // 2 + 32),
                                _FONT, 0.35, _GREY, 1, cv2.LINE_AA)
                panels.append(blank)

            # Horizontal stack of framework panels
            top_row = np.hstack(panels[:3])
            total_w = top_row.shape[1]

            # Rolling angle plot (resized to match width)
            plot_img = plot.render()
            plot_resized = cv2.resize(plot_img, (total_w, PLOT_H))

            # Status bar
            status_h = 28
            status_bar = np.zeros((status_h, total_w, 3), dtype=np.uint8)
            filter_name = filter_cycle[filter_idx]
            cal_status = "CAL" if calib_mgr.data.calibrated else "UNCAL"
            log_status = "REC" if logging_active else "---"
            udp_status = f"UDP:{args.host}:{args.port}" if sender else "NO-UDP"
            status_text = (
                f"  {source_label}  |  Overall FPS: {overall_fps:.1f}  |  "
                f"Filter: {filter_name}  |  {cal_status}  |  {log_status}  |  {udp_status}"
            )
            cv2.putText(status_bar, status_text, (6, 18),
                        _FONT, 0.4, _GREY, 1, cv2.LINE_AA)

            # Controls hint
            controls_bar = np.zeros((20, total_w, 3), dtype=np.uint8)
            cv2.putText(controls_bar, "Q=Quit  F=Filter  C=Calib  L=Log  R=Reset",
                        (6, 14), _FONT, 0.35, (100, 100, 100), 1, cv2.LINE_AA)

            # Compose final display
            combined = np.vstack([top_row, plot_resized, status_bar, controls_bar])

            # ── Video writer init (deferred to know dimensions) ───────────
            if args.save_video and writer is None:
                ch, cw = combined.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(
                    str(comparison_video_path), fourcc,
                    min(source_fps, 30.0), (cw, ch),
                )

            if writer is not None:
                writer.write(combined)

            # ── CSV logging ───────────────────────────────────────────────
            if logging_active:
                mf_logger.log(states)

            # ── Display ───────────────────────────────────────────────────
            cv2.imshow("MonoArm -- Full Pipeline", combined)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("f"):
                filter_idx = (filter_idx + 1) % len(filter_cycle)
                new_type = filter_cycle[filter_idx]
                for st in states.values():
                    st.reset_filter(new_type, args.hz)
                print(f"[->] Filter -> {new_type}")
            elif key == ord("c"):
                # Calibration uses the primary framework's runner
                run_calibration_session(
                    cap, states[PRIMARY_FRAMEWORK].runner,
                    calib_mgr, calib_path,
                )
            elif key == ord("l"):
                logging_active = not logging_active
                print(f"[->] Logging {'ON' if logging_active else 'OFF'}")
            elif key == ord("r"):
                for st in states.values():
                    st.reset_filter(filter_cycle[filter_idx], args.hz)
                print("[->] All filters reset.")

            # Console progress (every ~1 second worth of frames)
            if total_frames % int(max(source_fps, 10)) == 0:
                det_strs = [f"{st.display_name}:{st.detection_rate*100:.0f}%"
                            for st in states.values()]
                if sender is not None:
                    udp_str = (f"UDP TX OK -> {args.host}:{args.port} "
                               f"({sender.packets_sent} pkts)")
                    if sender.send_errors:
                        udp_str += f" [{sender.send_errors} errors]"
                else:
                    udp_str = "UDP off"
                sys.stdout.write(
                    f"\r[OK] Frame {total_frames}  |  "
                    f"FPS: {overall_fps:.1f}  |  "
                    f"Det: {' / '.join(det_strs)}  |  "
                    f"{udp_str}   "
                )
                sys.stdout.flush()

    except KeyboardInterrupt:
        print("\n\n[->] Interrupted by user.")
    finally:
        # ── Cleanup ───────────────────────────────────────────────────────
        if writer is not None:
            writer.release()
            size_mb = comparison_video_path.stat().st_size / 1e6
            print(f"\n[OK] Comparison video saved: {comparison_video_path} ({size_mb:.1f} MB)")

        mf_logger.close()

        # Save session summary JSON
        summary = {
            "source": str(args.video) if args.video else f"camera:{args.camera}",
            "total_frames": total_frames,
            "filter_type": filter_cycle[filter_idx],
            "calibrated": calib_mgr.data.calibrated,
            "udp_enabled": sender is not None,
            "frameworks": [st.summary_dict() for st in states.values()],
        }
        summary_path = out_dir / "session_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[OK] Session summary: {summary_path}")

        # Print final comparison table
        print("\n" + "=" * 60)
        print("  SESSION SUMMARY")
        print("=" * 60)
        print(f"  Total frames: {total_frames}")
        print(f"  Filter: {filter_cycle[filter_idx]}")
        print(f"  Output dir: {out_dir}")
        if sender is not None:
            print(f"  UDP transmitted: {sender.packets_sent} packets "
                  f"to {args.host}:{args.port} ({sender.send_errors} errors)")
        print("-" * 60)
        for st in states.values():
            print(f"  {st.display_name:20s}  "
                  f"FPS: {st.avg_fps:5.1f}  "
                  f"Det: {st.detection_rate*100:5.1f}%  "
                  f"Avg ms: {st.total_inference_ms / max(st.frame_count, 1):.1f}")
        print("=" * 60)

        # Teardown
        if sender is not None:
            sender.stop()
        for st in states.values():
            st.runner.close()
        cap.release()
        cv2.destroyAllWindows()
        print("[OK] Pipeline shutdown complete.\n")


if __name__ == "__main__":
    main()
