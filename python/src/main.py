"""
main.py — Stage 6: MonoArm Live Integration Pipeline
=====================================================

Ties together all pipeline stages into a unified, high-performance execution loop:
  Camera Capture → MediaPipe Pose → Anatomical Angle Solver → Kalman Filter → UDP Streaming

Features:
  - Configurable update rate (30 Hz target)
  - Real-time video window overlay with skeleton landmarks and computed angles
  - Textual dashboard showing filtered vs raw joint angles in the terminal
  - Optional CSV logging of capture sessions
  - Graceful teardown on keyboard interrupts (Ctrl+C or 'q')

Usage
-----
    python src/main.py
    python src/main.py --no_viz --log_session --filter kalman
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np



from config.config import Config, ANGLES_DIR
from src.pose.mediapipe_runner import MediaPipeRunner
from src.pose import angle_landmarks
from src.processing.angle_solver import compute_bilateral_angles, ArmAngles, BilateralArmAngles
from src.processing.angle_filter import BilateralFilterBank
from src.streaming.udp_streamer import UdpAngleSender


def draw_debug_info(
    frame:   np.ndarray,
    raw:     BilateralArmAngles | None,
    filt:    BilateralArmAngles | None,
    fps:     float,
    port:    int,
) -> None:
    """Draw dashboard HUD and values (both arms) directly onto the camera display frame."""
    # Semi-transparent overlay box for dashboard
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (360, 300), (26, 26, 38), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Info text
    cv2.putText(frame, "MonoArm Live Controller (bilateral)", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 220, 100), 2)
    cv2.putText(frame, f"FPS: {fps:.1f} | Port: {port}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    joints = [
        ("Shoulder Flex",   lambda a: a.shoulder_flexion),
        ("Shoulder Abd",    lambda a: a.shoulder_abduction),
        ("Shoulder Rot",    lambda a: a.shoulder_rotation),
        ("Elbow Flexion",   lambda a: a.elbow_flexion),
    ]

    y_offset = 80
    for side_label, raw_side, filt_side in (
        ("RIGHT", raw.right if raw else None, filt.right if filt else None),
        ("LEFT",  raw.left  if raw else None, filt.left  if filt else None),
    ):
        cv2.putText(frame, side_label, (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 255), 1)
        y_offset += 22
        for label, getter in joints:
            raw_val = getter(raw_side) if raw_side else 0.0
            filt_val = getter(filt_side) if filt_side else 0.0

            cv2.putText(frame, f"{label}:", (20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1)
            cv2.putText(frame, f"{filt_val:6.1f} deg (raw: {raw_val:6.1f})", (140, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 200, 255), 1)
            y_offset += 25
        y_offset += 8

    # Simple connection status light
    detected = bool(raw and (raw.right or raw.left))
    color = (100, 255, 100) if detected else (100, 100, 255)
    status_txt = "STREAMING ACTIVE" if detected else "NO POSE DETECTED"
    cv2.circle(frame, (25, y_offset + 10), 6, color, -1)
    cv2.putText(frame, status_txt, (40, y_offset + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description="MonoArm Live Integration Pipeline")
    ap.add_argument("--camera",     type=int, default=Config.CAMERA_ID,
                    help="Camera device index")
    ap.add_argument("--filter",     default=Config.DEFAULT_FILTER_TYPE,
                    choices=["kalman", "ma", "sg"],
                    help="Temporal filter to use (default: kalman)")
    ap.add_argument("--no_viz",     action="store_true",
                    help="Run headless without showing OpenCV camera window")
    ap.add_argument("--log_session", action="store_true",
                    help="Save CSV logs of the captured joint angles")
    args = ap.parse_args()

    Config.ensure_directories()

    # ── Initialize Components ─────────────────────────────────────────────────
    print("\n=======================================================")
    print("  Initializing MonoArm Live Control Pipeline...")
    print("=======================================================")
    
    runner = MediaPipeRunner(
        detection_confidence=Config.MIN_DETECTION_CONFIDENCE,
        tracking_confidence=Config.MIN_TRACKING_CONFIDENCE,
        model_complexity=Config.MODEL_COMPLEXITY,
    )
    
    # Choose 30.0 Hz target loop rate for matching the Unity avatar update rate
    filt_bank = BilateralFilterBank(filter_type=args.filter, stream_hz=Config.STREAM_HZ)
    udp_sender = UdpAngleSender(host=Config.UDP_IP, port=Config.UDP_PORT, hz=Config.STREAM_HZ,
                                 bilateral=True)
    
    cap = cv2.VideoCapture(args.camera, cv2.CAP_MSMF)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)

    if not cap.isOpened():
        print(f"[FAIL] Error: Cannot open camera index {args.camera}")
        runner.close()
        sys.exit(1)

    # ── CSV Logging Setup ─────────────────────────────────────────────────────
    log_file = None
    csv_writer = None
    if args.log_session:
        Config.ensure_directories()
        log_path = ANGLES_DIR / f"session_{int(time.time())}.csv"
        log_file = open(log_path, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(log_file)
        # Header row
        csv_writer.writerow([
            "timestamp", "elapsed_s", "pose_detected",
            "right_shoulder_flexion_raw", "right_shoulder_flexion_filt",
            "right_shoulder_abduction_raw", "right_shoulder_abduction_filt",
            "right_shoulder_rotation_raw", "right_shoulder_rotation_filt",
            "right_elbow_flexion_raw", "right_elbow_flexion_filt",
            "left_shoulder_flexion_raw", "left_shoulder_flexion_filt",
            "left_shoulder_abduction_raw", "left_shoulder_abduction_filt",
            "left_shoulder_rotation_raw", "left_shoulder_rotation_filt",
            "left_elbow_flexion_raw", "left_elbow_flexion_filt",
        ])
        print(f"[OK] Logging session data to: {log_path}")

    # ── Execution Loop ────────────────────────────────────────────────────────
    udp_sender.start()
    print(f"[OK] UDP sending active: {Config.UDP_IP}:{Config.UDP_PORT} at {Config.STREAM_HZ}Hz")
    print("\n>>> PIPELINE RUNNING. Press 'Ctrl+C' in console or 'q' in video window to exit.")
    print("---------------------------------------------------------------------------------")

    start_time = time.perf_counter()
    last_fps_time = time.perf_counter()
    frame_count = 0
    fps = Config.STREAM_HZ

    # Target loop spacing: 33.3ms for 30 Hz
    loop_interval = 1.0 / Config.STREAM_HZ
    next_loop_time = time.perf_counter()

    try:
        while True:
            t_now = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                print("[!] Frame acquisition drop.")
                continue

            frame_count += 1
            if frame_count % 30 == 0:
                now = time.perf_counter()
                fps = 30.0 / (now - last_fps_time)
                last_fps_time = now

            # Process Pose Estimation
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            landmarks = runner.process(rgb)

            raw_angles = None
            filt_angles = None

            if landmarks is not None:
                # Solve joint angles for both arms
                raw_angles = compute_bilateral_angles(
                    angle_landmarks(runner, landmarks, rgb.shape[1], rgb.shape[0]))

                if raw_angles is not None:
                    # Apply filter (per side, per DOF)
                    filt_angles = filt_bank.update(raw_angles)

                    # Update UDP payload
                    udp_sender.update_bilateral(filt_angles)

            # ── Session Logging ───────────────────────────────────────────────────
            if csv_writer and log_file:
                elapsed = time.perf_counter() - start_time
                pose_detected = int(bool(raw_angles and (raw_angles.right or raw_angles.left)))
                if pose_detected:
                    def _fields(raw_side: ArmAngles | None, filt_side: ArmAngles | None) -> list:
                        if raw_side is None or filt_side is None:
                            return [0.0] * 8
                        return [
                            f"{raw_side.shoulder_flexion:.3f}", f"{filt_side.shoulder_flexion:.3f}",
                            f"{raw_side.shoulder_abduction:.3f}", f"{filt_side.shoulder_abduction:.3f}",
                            f"{raw_side.shoulder_rotation:.3f}", f"{filt_side.shoulder_rotation:.3f}",
                            f"{raw_side.elbow_flexion:.3f}", f"{filt_side.elbow_flexion:.3f}",
                        ]
                    csv_writer.writerow(
                        [time.time(), f"{elapsed:.3f}", 1]
                        + _fields(raw_angles.right, filt_angles.right)
                        + _fields(raw_angles.left,  filt_angles.left)
                    )
                else:
                    csv_writer.writerow([time.time(), f"{elapsed:.3f}", 0] + [0.0] * 16)

            # ── Video Overlay Display ─────────────────────────────────────────────
            if not args.no_viz:
                draw_debug_info(frame, raw_angles, filt_angles, fps, Config.UDP_PORT)
                cv2.imshow("MonoArm Controller", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # Print quick dashboard in console (carriage return to overwrite line)
            if frame_count % 15 == 0:
                r = filt_angles.right if filt_angles else None
                l = filt_angles.left if filt_angles else None
                if r or l:
                    def _fmt(a: ArmAngles | None) -> str:
                        if a is None:
                            return "no track"
                        return (f"Flex:{a.shoulder_flexion:5.1f}° Abd:{a.shoulder_abduction:5.1f}° "
                                f"Rot:{a.shoulder_rotation:5.1f}° Elb:{a.elbow_flexion:5.1f}°")
                    sys.stdout.write(
                        f"\r[OK] R[{_fmt(r)}] | L[{_fmt(l)}] | FPS: {fps:4.1f}"
                    )
                else:
                    sys.stdout.write("\r[WAIT] Searching for pose...")
                sys.stdout.flush()

            # ── Loop Rate Control ─────────────────────────────────────────────────
            next_loop_time += loop_interval
            sleep_time = next_loop_time - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # Loop is running behind: reset scheduler
                next_loop_time = time.perf_counter()

    except KeyboardInterrupt:
        print("\n\n[->] Stopping pipeline gracefully...")
    finally:
        # Clean teardown
        cap.release()
        udp_sender.stop()
        runner.close()
        if log_file:
            log_file.close()
        cv2.destroyAllWindows()
        print("[OK] Pipeline shutdown complete.\n")


if __name__ == "__main__":
    main()
