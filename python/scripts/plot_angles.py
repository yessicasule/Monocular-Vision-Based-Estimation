"""
Post-session joint angle plotter.

Reads a session CSV written by scripts/run_demo.py (CsvAngleLogger) or by
src/main.py, and produces a figure with:
  - Raw vs filtered overlay per joint
  - ±3 deg variance band around the filtered mean (the Month-4 target)
  - Per-joint statistics (mean, std, detection rate)
  - Optionally overlays the right arm's raw and filtered elbow angle on a single
    comparison panel for quick visual inspection

Usage:
    python scripts/plot_angles.py --csv data/sessions/arm_test_01/angles.csv
    python scripts/plot_angles.py --csv data/sessions/arm_test_01/angles.csv \
        --out outputs/arm_test_01_angles.png --show
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

# Graceful import — matplotlib may not be installed in every env
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

_JOINTS = [
    ("elbow_flexion",        "Elbow Flexion",        "#e05c5c"),
    ("shoulder_flexion",     "Shoulder Flexion",     "#5c9ee0"),
    ("shoulder_rotation",    "Shoulder Rotation",    "#5ce07a"),
    ("shoulder_abduction",   "Shoulder Abduction",   "#e0c05c"),
]


_FALLBACKS = {
    "shoulder_flexion":   "shoulder_elevation",
    "shoulder_rotation":  "shoulder_yaw",
    "shoulder_abduction": "shoulder_roll",
}


def _name_variants(key: str) -> list[str]:
    """Column base names for a joint, newest naming first."""
    names = [key, f"right_{key}"]
    if key in _FALLBACKS:
        names += [_FALLBACKS[key], f"right_{_FALLBACKS[key]}"]
    return names


def load_csv(csv_path: Path) -> dict:
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        raise RuntimeError(f"Empty CSV: {csv_path}")

    header = rows[0]

    time_col = next((c for c in ("elapsed_s", "timestamp_s") if c in header), None)
    if time_col is None:
        raise RuntimeError(
            f"{csv_path} has no 'elapsed_s' or 'timestamp_s' column. "
            f"Columns present: {', '.join(header)}"
        )
    elapsed = np.array([float(r[time_col]) for r in rows])

    # CsvAngleLogger only writes rows for frames where a pose was detected.
    if "pose_detected" in header:
        detected = np.array([int(r["pose_detected"]) for r in rows])
    else:
        detected = np.ones(len(rows), dtype=int)

    data = {"elapsed": elapsed, "detected": detected, "n": len(rows)}

    has_raw = False
    for key, *_ in _JOINTS:
        raw_key = filt_key = None
        for name in _name_variants(key):
            if f"{name}_raw" in header and f"{name}_filt" in header:
                raw_key, filt_key = f"{name}_raw", f"{name}_filt"
                has_raw = True
                break

        if raw_key is None:
            # Single already-filtered series (CsvAngleLogger).
            col = next((n for n in _name_variants(key) if n in header), None)
            if col is None:
                raise RuntimeError(
                    f"{csv_path} has no column for '{key}'. "
                    f"Columns present: {', '.join(header)}"
                )
            raw_key = filt_key = col

        data[f"{key}_raw"]  = np.array([float(r[raw_key])  for r in rows])
        data[f"{key}_filt"] = np.array([float(r[filt_key]) for r in rows])

    data["has_raw"] = has_raw
    return data


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_VARIANCE_TARGET_DEG = 3.0   # Month-4 spec: ±3 deg on static pose


def _plot_joint(ax, elapsed, raw, filt, colour, title, detected, has_raw=True):
    mask = detected.astype(bool)

    if has_raw:
        # Raw trace — muted
        ax.plot(elapsed, raw, color=colour, alpha=0.25, linewidth=0.8, label="Raw")

    # Filtered trace — solid
    ax.plot(elapsed, filt, color=colour, linewidth=1.6,
            label="Filtered" if has_raw else "Logged")

    # ±3 deg band around filtered mean (detected frames only)
    if mask.any():
        mean_filt = float(np.mean(filt[mask]))
        ax.axhspan(mean_filt - _VARIANCE_TARGET_DEG,
                   mean_filt + _VARIANCE_TARGET_DEG,
                   color=colour, alpha=0.08, label=f"±{_VARIANCE_TARGET_DEG}° target")
        ax.axhline(mean_filt, color=colour, linewidth=0.7, linestyle="--", alpha=0.5)

        std_filt = float(np.std(filt[mask]))
        det_rate = float(mask.sum()) / len(mask) * 100
        if has_raw:
            std_raw = float(np.std(raw[mask]))
            std_txt = f"std raw {std_raw:.1f}°  filt {std_filt:.1f}°"
        else:
            std_txt = f"std {std_filt:.1f}°"
        subtitle = (f"mean {mean_filt:.1f}°  |  {std_txt}  |  "
                    f"detect {det_rate:.0f}%")
        ax.set_title(f"{title}\n{subtitle}", fontsize=9, pad=4)
    else:
        ax.set_title(title, fontsize=9, pad=4)

    ax.set_ylabel("Angle (deg)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(15))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(5))
    ax.grid(True, which="major", alpha=0.25)
    ax.grid(True, which="minor", alpha=0.10)
    ax.legend(fontsize=7, loc="upper right", framealpha=0.6)


def make_figure(data: dict, session_name: str) -> "plt.Figure":
    fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
    fig.suptitle(f"Joint Angles — {session_name}\n"
                 f"(grey band = ±{_VARIANCE_TARGET_DEG}° target around filtered mean)",
                 fontsize=11, y=0.995)

    elapsed  = data["elapsed"]
    detected = data["detected"]

    for ax, (key, title, colour) in zip(axes, _JOINTS):
        _plot_joint(ax, elapsed,
                    data[f"{key}_raw"], data[f"{key}_filt"],
                    colour, title, detected, data["has_raw"])

    axes[-1].set_xlabel("Time (s)", fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def make_comparison_figure(data: dict, session_name: str) -> "plt.Figure":
    """Single panel: raw vs filtered for all four joints on overlaid axes."""
    has_raw = data["has_raw"]
    n_panels = 2 if has_raw else 1
    fig, axes = plt.subplots(n_panels, 1, figsize=(13, 3 * n_panels + 0.5), sharex=True,
                             squeeze=False)
    axes = axes[:, 0]
    ax_filt = axes[-1]
    elapsed  = data["elapsed"]

    for key, title, colour in _JOINTS:
        if has_raw:
            axes[0].plot(elapsed, data[f"{key}_raw"],
                         color=colour, linewidth=0.9, alpha=0.75, label=title)
        ax_filt.plot(elapsed, data[f"{key}_filt"],
                     color=colour, linewidth=1.4, label=title)

    labels = ["Raw", "Filtered (Kalman/EMA)"] if has_raw else ["Logged angles"]
    for ax, label in zip(axes, labels):
        ax.set_ylabel("Angle (deg)", fontsize=8)
        ax.set_title(label, fontsize=9)
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=7, loc="upper right", framealpha=0.6)
        ax.tick_params(labelsize=7)

    ax_filt.set_xlabel("Time (s)", fontsize=8)
    fig.suptitle(f"All Joints — {session_name}", fontsize=11)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Plot joint angles from session CSV")
    ap.add_argument("--csv",   required=True, help="Path to angles.csv")
    ap.add_argument("--out",   default=None,
                    help="Output PNG path (default: <csv_dir>/<session>_angles.png)")
    ap.add_argument("--show",  action="store_true",
                    help="Open figure in interactive window (requires display)")
    ap.add_argument("--comparison", action="store_true",
                    help="Also save an all-joints overlay comparison figure")
    args = ap.parse_args()

    if not _HAS_MPL:
        sys.exit("matplotlib not installed. Run: pip install matplotlib")

    csv_path     = Path(args.csv)
    session_name = csv_path.parent.name

    out_path = Path(args.out) if args.out else csv_path.parent / f"{session_name}_angles.png"

    print(f"Loading {csv_path} ...")
    data = load_csv(csv_path)
    print(f"  {data['n']} frames  |  "
          f"detection rate {data['detected'].mean()*100:.1f}%  |  "
          f"duration {data['elapsed'][-1]:.1f} s")

    fig = make_figure(data, session_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out_path}")

    if args.comparison:
        cmp_path = out_path.with_name(out_path.stem + "_comparison.png")
        fig2 = make_comparison_figure(data, session_name)
        fig2.savefig(cmp_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"Saved: {cmp_path}")

    if args.show:
        matplotlib.use("TkAgg")
        plt.show()

    plt.close("all")


if __name__ == "__main__":
    main()
