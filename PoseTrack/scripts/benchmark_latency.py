"""
benchmark_latency.py — End-to-End Pipeline Latency Profiler
============================================================

Measures per-component and end-to-end latency for the MonoArm pipeline:

    Frame capture → MediaPipe inference → Angle solver
    → Filter update → UDP send

Produces a per-component breakdown table with mean, P50, P95, and max
values (milliseconds) and saves a latency histogram.

The 100 ms end-to-end spec corresponds to a 10 Hz minimum update rate,
which is the minimum for real-time robot control feedback loops.
For avatar animation, 30+ FPS requires < 33 ms total.

Usage
-----
    python scripts/benchmark_latency.py
    python scripts/benchmark_latency.py --frames 300 --camera 0 --filter kalman
    python scripts/benchmark_latency.py --frames 300 --no_udp
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from src.pose.mediapipe_runner import MediaPipeRunner
from src.pose import angle_landmarks
from src.processing.angle_solver import compute_arm_angles
from src.processing.angle_filter import AngleFilterBank
from src.streaming.udp_streamer import UdpAngleSender


# ── Latency spec ─────────────────────────────────────────────────────────────
SPEC_MS = 100.0       # end-to-end latency requirement (ms)
TARGET_FPS = 30.0     # target frame rate for avatar animation


def benchmark(
    n_frames:  int  = 200,
    camera_id: int  = 0,
    filter_t:  str  = "kalman",
    send_udp:  bool = True,
) -> None:
    print(f"\n{'='*58}")
    print(f"  MonoArm Latency Benchmark")
    print(f"  Frames: {n_frames}   Camera: {camera_id}   Filter: {filter_t}")
    print(f"{'='*58}\n")

    # ── Setup ─────────────────────────────────────────────────────────────────
    runner   = MediaPipeRunner(model_complexity=1)
    filt     = AngleFilterBank(filter_type=filter_t, stream_hz=30.0)
    sender   = None
    if send_udp:
        sender = UdpAngleSender(host="127.0.0.1", port=9000, hz=30.0)
        sender.start()

    cap = cv2.VideoCapture(camera_id, cv2.CAP_MSMF)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("[✗] Cannot open camera. Provide a connected camera or use --no_udp.")
        if sender:
            sender.stop()
        runner.close()
        sys.exit(1)

    # ── Warm-up (discard first 20 frames to let auto-exposure settle) ─────────
    print("  Warming up (20 frames)...")
    for _ in range(20):
        cap.read()
    time.sleep(0.3)

    # ── Benchmark loop ────────────────────────────────────────────────────────
    buckets: dict[str, list[float]] = {
        "capture":   [],
        "mediapipe": [],
        "angles":    [],
        "filter":    [],
        "udp_send":  [],
        "total":     [],
    }
    detected = 0

    print(f"  Running {n_frames} frames...\n")
    for i in range(n_frames):
        t0 = time.perf_counter()

        # 1. Capture
        ret, frame = cap.read()
        t1 = time.perf_counter()
        if not ret:
            continue

        # 2. MediaPipe inference
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        lms = runner.process(rgb)
        t2  = time.perf_counter()

        if lms is None:
            continue
        detected += 1

        # 3. Angle solver
        angles = compute_arm_angles(
            angle_landmarks(runner, lms, rgb.shape[1], rgb.shape[0]))
        t3     = time.perf_counter()

        if angles is None:
            continue

        # 4. Filter
        filtered = filt.update(angles)
        t4       = time.perf_counter()

        # 5. UDP send
        if sender:
            sender.update(filtered)
        t5 = time.perf_counter()

        buckets["capture"].append(  (t1 - t0) * 1000)
        buckets["mediapipe"].append((t2 - t1) * 1000)
        buckets["angles"].append(   (t3 - t2) * 1000)
        buckets["filter"].append(   (t4 - t3) * 1000)
        buckets["udp_send"].append( (t5 - t4) * 1000)
        buckets["total"].append(    (t5 - t0) * 1000)

    # ── Teardown ──────────────────────────────────────────────────────────────
    cap.release()
    if sender:
        sender.stop()
    runner.close()

    if not buckets["total"]:
        print("[!] No poses detected. Point the camera at a person and retry.")
        return

    # ── Report ────────────────────────────────────────────────────────────────
    detection_rate = detected / n_frames * 100
    n_valid        = len(buckets["total"])

    print(f"  Detection rate : {detection_rate:.1f}%  ({detected}/{n_frames} frames)")
    print(f"  Valid samples  : {n_valid}\n")

    header = f"  {'Component':<14}  {'Mean':>7}  {'P50':>7}  {'P95':>7}  {'Max':>7}  ms"
    print(header)
    print("  " + "─" * (len(header) - 2))

    stats: dict[str, dict] = {}
    for name, vals in buckets.items():
        arr    = np.array(vals)
        label  = "END-TO-END" if name == "total" else name
        mean   = float(arr.mean())
        p50    = float(np.percentile(arr, 50))
        p95    = float(np.percentile(arr, 95))
        maxv   = float(arr.max())
        prefix = ">>> " if name == "total" else "    "
        print(f"{prefix}{label:<14}  {mean:>7.2f}  {p50:>7.2f}  {p95:>7.2f}  {maxv:>7.2f}")
        stats[name] = {"mean": mean, "p50": p50, "p95": p95, "max": maxv, "n": n_valid}

    print()
    total_arr  = np.array(buckets["total"])
    fps_mean   = 1000.0 / float(total_arr.mean())
    pct_pass   = float(np.mean(total_arr < SPEC_MS) * 100)
    pct_30fps  = float(np.mean(total_arr < (1000.0 / TARGET_FPS)) * 100)

    print(f"  Mean throughput   : {fps_mean:.1f} FPS")
    print(f"  < {SPEC_MS:.0f} ms (spec)    : {pct_pass:.1f}% of frames pass")
    print(f"  < {1000/TARGET_FPS:.0f} ms ({TARGET_FPS:.0f} FPS)  : {pct_30fps:.1f}% of frames pass")

    if total_arr.mean() < SPEC_MS:
        print(f"\n  ✅  Mean {total_arr.mean():.1f} ms — WITHIN {SPEC_MS:.0f} ms spec")
    else:
        print(f"\n  ⚠️   Mean {total_arr.mean():.1f} ms — EXCEEDS {SPEC_MS:.0f} ms spec")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out_dir  = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_out = out_dir / "latency_benchmark.json"
    report   = {
        "n_frames":       n_frames,
        "n_detected":     detected,
        "detection_rate": detection_rate,
        "filter_type":    filter_t,
        "fps_mean":       fps_mean,
        "spec_pass_pct":  pct_pass,
        "components":     stats,
    }
    with open(json_out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  [✓] JSON report → {json_out}")

    # ── Latency histogram ─────────────────────────────────────────────────────
    _save_histogram(total_arr, out_dir / "latency_histogram.png")


def _save_histogram(total_ms: np.ndarray, out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(9, 4), facecolor="#0e0e18")
    ax.set_facecolor("#1a1a26")

    ax.hist(total_ms, bins=40, color="#64dc64", edgecolor="#333", linewidth=0.5,
            alpha=0.85, label="Frame latency")
    ax.axvline(total_ms.mean(),              color="white",   linewidth=1.5,
               linestyle="--", label=f"Mean {total_ms.mean():.1f} ms")
    ax.axvline(np.percentile(total_ms, 95),  color="#ffb450", linewidth=1.5,
               linestyle="--", label=f"P95  {np.percentile(total_ms, 95):.1f} ms")
    ax.axvline(SPEC_MS,                      color="#ff4444", linewidth=1.2,
               linestyle=":",  label=f"{SPEC_MS:.0f} ms spec")

    ax.set_xlabel("End-to-End Latency (ms)", fontsize=10, color="#ccc")
    ax.set_ylabel("Frame Count",             fontsize=10, color="#ccc")
    ax.set_title("MonoArm Pipeline Latency Distribution",
                 color="#aacfff", fontsize=11)
    ax.tick_params(colors="#aaa")
    ax.spines[:].set_color("#333")
    ax.legend(fontsize=9, labelcolor="white", facecolor="#1a1a26", framealpha=0.8)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [✓] Histogram → {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="MonoArm end-to-end latency benchmark.")
    ap.add_argument("--frames",  type=int, default=200,    help="Number of benchmark frames")
    ap.add_argument("--camera",  type=int, default=0,      help="Camera device index")
    ap.add_argument("--filter",  default="kalman",
                    choices=["kalman", "ma", "sg"])
    ap.add_argument("--no_udp",  action="store_true",      help="Skip UDP send step")
    args = ap.parse_args()
    benchmark(args.frames, args.camera, args.filter, send_udp=not args.no_udp)
