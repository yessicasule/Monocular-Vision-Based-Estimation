"""
compare_filters.py — Stage 5: Temporal Filter Comparison & Calibration
========================================================================

Evaluates Moving Average, Savitzky-Golay, and 2-state Kalman filters
against the Month-4 target (±3° variance on static poses) and dynamic lag.

Generates:
  1. Filter response plot (outputs/filter_comparison.png)
  2. Performance summary table printed to console
  3. Tuned Kalman parameters recommendations

Usage
-----
    python scripts/compare_filters.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from src.processing.angle_filter import (
    MovingAverageFilter,
    SavitzkyGolayFilter,
    KalmanFilter2State,
)


def generate_benchmark_signals(
    n_samples: int = 400,
    hz:        float = 30.0,
    seed:      int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic signals modeling both static hold and dynamic motion.

    - Frames 0 - 150:  Static hold at 45.0° (to measure Month-4 variance spec)
    - Frames 150 - 250: Step input to 90.0° (to measure step/lag response)
    - Frames 250 - 400: Sinusoidal tracking at 1.0 Hz (to measure phase delay)
    """
    rng = np.random.default_rng(seed)
    t   = np.arange(n_samples) / hz

    gt = np.zeros(n_samples)
    # Static hold
    gt[0:150] = 45.0
    # Step response
    gt[150:250] = 90.0
    # Sinusoid
    gt[250:400] = 60.0 + 30.0 * np.sin(2.0 * np.pi * 1.0 * (t[250:400] - t[250]))

    # Add Gaussian noise (modeling MediaPipe/MoveNet landmark jitter)
    noise = rng.normal(0.0, 2.5, size=n_samples)

    # Add sporadic outlier spikes (modeling self-occlusions)
    spikes = rng.choice([0.0, 15.0, -15.0], size=n_samples, p=[0.96, 0.02, 0.02])

    noisy = gt + noise + spikes
    return gt, noisy, t


def run_filter(filt, noisy_signal: np.ndarray) -> np.ndarray:
    filt.reset()
    return np.array([filt.update(float(v)) for v in noisy_signal])


def evaluate_filter_metrics(
    name:     str,
    gt:       np.ndarray,
    noisy:    np.ndarray,
    filtered: np.ndarray,
    hz:       float = 30.0,
) -> dict:
    """
    Compute specific validation metrics:
      - Static standard deviation (target < 3.0° for Month-4 spec)
      - Peak phase delay during sinusoidal motion (ms)
      - Total Root Mean Squared Error (RMSE) vs ground truth
    """
    # 1. Static Hold (frames 20 - 140 to avoid filter initialization boundary)
    static_filt  = filtered[20:140]
    static_noisy = noisy[20:140]
    static_std   = float(np.std(static_filt))
    static_noise_reduct = float(1.0 - (static_std / np.std(static_noisy)))

    # 2. Step response phase lag (frames 150 - 180)
    # Measure time to reach 90% of step change (from 45 to 90 -> target = 85.5)
    target_val = 45.0 + 0.90 * 45.0
    step_idx = 150
    rise_idx_filt = np.where(filtered[step_idx:step_idx+30] >= target_val)[0]
    rise_idx_gt   = np.where(gt[step_idx:step_idx+30] >= target_val)[0]

    lag_ms = 0.0
    if len(rise_idx_filt) > 0 and len(rise_idx_gt) > 0:
        lag_frames = rise_idx_filt[0] - rise_idx_gt[0]
        lag_ms = (lag_frames / hz) * 1000.0

    # 3. Overall tracking error
    rmse = float(np.sqrt(np.mean((filtered - gt) ** 2)))

    return {
        "name":           name,
        "static_std":     static_std,
        "noise_reduct":   static_noise_reduct * 100.0,
        "step_lag_ms":    lag_ms,
        "rmse":           rmse,
    }


def main() -> None:
    print("\n=======================================================")
    print("  Stage 5: Temporal Filter Benchmarking & Tuning")
    print("=======================================================\n")

    hz = 30.0
    gt, noisy, t = generate_benchmark_signals(n_samples=400, hz=hz)

    # ── Define Filter Candidates ──────────────────────────────────────────────
    filters = {
        "Moving Average (W=5)": MovingAverageFilter(window_size=5),
        "Moving Average (W=9)": MovingAverageFilter(window_size=9),
        "Savitzky-Golay (W=11, p=3)": SavitzkyGolayFilter(window_length=11, polyorder=3),
        "Kalman 2-State (Responsive)": KalmanFilter2State(
            process_noise=0.1, measurement_noise=1.5, dt=1.0 / hz
        ),
        "Kalman 2-State (Balanced)": KalmanFilter2State(
            process_noise=0.01, measurement_noise=1.5, dt=1.0 / hz
        ),
        "Kalman 2-State (Smooth)": KalmanFilter2State(
            process_noise=0.001, measurement_noise=2.5, dt=1.0 / hz
        ),
    }

    # Run evaluation
    results = []
    filtered_signals = {}

    for name, filt in filters.items():
        filt_sig = run_filter(filt, noisy)
        filtered_signals[name] = filt_sig
        metrics = evaluate_filter_metrics(name, gt, noisy, filt_sig, hz=hz)
        results.append(metrics)

    # ── Print Console Summary ─────────────────────────────────────────────────
    print(f"{'Filter Configuration':<28} | {'Static Std (°)':<15} | {'Noise Red. (%)':<15} | {'Step Lag (ms)':<14} | {'RMSE (°)':<10}")
    print("-" * 92)
    for r in results:
        target_check = "PASS" if r["static_std"] <= 3.0 else "FAIL"
        print(f"{r['name']:<28} | {r['static_std']:>7.2f}° ({target_check}) | {r['noise_reduct']:>12.1f}% | {r['step_lag_ms']:>10.1f} ms | {r['rmse']:>8.2f}°")
    print("-" * 92)
    print("Month-4 Spec Requirement: Static Std <= 3.0° on static poses.\n")

    # ── Generate Plots ────────────────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), facecolor="#0e0e18")

        for ax in axes:
            ax.set_facecolor("#1a1a26")
            ax.tick_params(colors="#aaa", labelsize=8)
            ax.spines[:].set_color("#333")
            ax.grid(color="#2a2a3a", linewidth=0.5, linestyle="--")

        # Top Plot: Full overview
        axes[0].plot(t, noisy, color="#555", alpha=0.35, label="Noisy Input", linewidth=0.8)
        axes[0].plot(t, gt,    color="white",  linestyle="--", label="Ground Truth", linewidth=1.2)
        axes[0].plot(t, filtered_signals["Kalman 2-State (Balanced)"], color="#64dc64", label="Kalman (Balanced)", linewidth=1.5)
        axes[0].plot(t, filtered_signals["Moving Average (W=9)"],      color="#ffb450", label="MA (W=9)", linewidth=1.2)
        axes[0].set_title("Filter Response Overview", color="#aacfff", fontsize=11)
        axes[0].legend(fontsize=8, labelcolor="white", facecolor="#1a1a26", framealpha=0.8)

        # Middle Plot: Zoom on Static Hold and Step Lag
        axes[1].plot(t[120:200], gt[120:200],    color="white", linestyle="--", linewidth=1.2)
        axes[1].plot(t[120:200], noisy[120:200], color="#555", alpha=0.4, linewidth=0.8)
        axes[1].plot(t[120:200], filtered_signals["Kalman 2-State (Balanced)"][120:200], color="#64dc64", linewidth=1.6)
        axes[1].plot(t[120:200], filtered_signals["Moving Average (W=9)"][120:200],      color="#ffb450", linewidth=1.2)
        # Highlight ±3 deg target band around static mean
        axes[1].axhspan(45.0 - 3.0, 45.0 + 3.0, color="#64dc64", alpha=0.08, label="Month-4 target (±3°)")
        axes[1].set_title("Transient Step Response & Static Stability Target (Zoomed)", color="#aacfff", fontsize=11)

        # Bottom Plot: Zoom on Sinusoidal Tracking
        axes[2].plot(t[280:360], gt[280:360],    color="white", linestyle="--", linewidth=1.2)
        axes[2].plot(t[280:360], filtered_signals["Kalman 2-State (Balanced)"][280:360], color="#64dc64", linewidth=1.6)
        axes[2].plot(t[280:360], filtered_signals["Moving Average (W=9)"][280:360],      color="#ffb450", linewidth=1.2)
        axes[2].plot(t[280:360], filtered_signals["Savitzky-Golay (W=11, p=3)"][280:360], color="#64a0ff", linewidth=1.2, label="Savitzky-Golay")
        axes[2].set_title("1.0 Hz Sinusoidal Tracking / Phase Delay (Zoomed)", color="#aacfff", fontsize=11)
        axes[2].set_xlabel("Time (s)", color="#ccc")

        plt.tight_layout()
        out_path = Path("outputs/filter_comparison.png")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        print(f"[OK] Saved comparison figure to: {out_path}")
    except ImportError:
        print("[!] matplotlib not installed. Skipping plot generation.")

    # ── Kalman Tuning Recommendations ──────────────────────────────────────────
    print("\n=======================================================")
    print("  Kalman filter parameter tuning guide:")
    print("=======================================================")
    print("1. For extreme noise or low frame rate (<15 FPS):")
    print("   Set process_noise (q) = 0.001 and measurement_noise (R) = 2.5 (High smoothing).")
    print("2. For high-fidelity tracking (30+ FPS, good lighting):")
    print("   Set process_noise (q) = 0.01 and measurement_noise (R) = 1.5 (Balanced, <100ms lag).")
    print("3. For robotics / minimal latency control feedback loops:")
    print("   Set process_noise (q) = 0.1 and measurement_noise (R) = 1.0 (High responsiveness).\n")


if __name__ == "__main__":
    main()
