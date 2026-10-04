"""
angle_logger.py — Session Data Logging and Visualisation
=========================================================

Provides two complementary data recording tools:

1. CsvAngleLogger
   Writes one row per frame to a timestamped CSV file.
   Columns: timestamp_s, frame, shoulder_flexion, shoulder_abduction,
            shoulder_rotation, elbow_flexion, rotation_reliable, filter_type, calibrated

2. RollingPlot
   Maintains a rolling buffer of angle history and renders a live
   four-panel plot onto an OpenCV image for real-time visualisation.
   No matplotlib window is required; the plot is composited into the
   main video frame for zero-overhead display.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from collections import deque
from typing import Optional

import cv2
import numpy as np

from .angle_solver import ArmAngles, BilateralArmAngles


# ---------------------------------------------------------------------------
# CSV Logger
# ---------------------------------------------------------------------------

class CsvAngleLogger:
    """
    Timestamped CSV logger for joint angle sessions.

    Parameters
    ----------
    output_dir : str or Path
        Directory where log files are saved.
    filter_type : str
        Name of the active filter (for metadata).
    session_label : str
        Optional label appended to the filename.
    """

    COLUMNS = [
        "timestamp_s", "frame",
        "shoulder_flexion", "shoulder_abduction", "shoulder_rotation", "elbow_flexion",
        "rotation_reliable", "filter_type", "calibrated",
    ]

    def __init__(
        self,
        output_dir:    str | Path = "outputs/logs",
        filter_type:   str = "kalman",
        session_label: str = "",
    ) -> None:
        self._dir        = Path(output_dir)
        self._filter     = filter_type
        self._label      = session_label
        self._file       = None
        self._writer     = None
        self._start_time = 0.0
        self._frame      = 0
        self._calibrated = False

    def start(self, calibrated: bool = False) -> Path:
        """
        Open the CSV file and write the header row.

        Returns
        -------
        Path
            Path to the opened log file.
        """
        self._dir.mkdir(parents=True, exist_ok=True)
        ts    = time.strftime("%Y%m%d_%H%M%S")
        label = f"_{self._label}" if self._label else ""
        path  = self._dir / f"angles_{ts}{label}.csv"

        self._file       = open(path, "w", newline="")
        self._writer     = csv.DictWriter(self._file, fieldnames=self.COLUMNS)
        self._writer.writeheader()
        self._start_time = time.perf_counter()
        self._frame      = 0
        self._calibrated = calibrated
        return path

    def log(self, angles: ArmAngles) -> None:
        """Write one row for the current frame."""
        if self._writer is None:
            raise RuntimeError("Call start() before log().")
        self._writer.writerow({
            "timestamp_s":        round(time.perf_counter() - self._start_time, 4),
            "frame":              self._frame,
            "shoulder_flexion":   round(angles.shoulder_flexion,   2),
            "shoulder_abduction": round(angles.shoulder_abduction, 2),
            "shoulder_rotation":  round(angles.shoulder_rotation,  2),
            "elbow_flexion":      round(angles.elbow_flexion,      2),
            "rotation_reliable":  int(angles.rotation_reliable),
            "filter_type":        self._filter,
            "calibrated":         int(self._calibrated),
        })
        self._frame += 1

    def stop(self) -> None:
        """Flush and close the log file."""
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file   = None
            self._writer = None


_SIDES        = ["right", "left"]
_ANGLE_FIELDS = ["shoulder_flexion", "shoulder_abduction", "shoulder_rotation", "elbow_flexion"]


class BilateralCsvAngleLogger:
    """
    Timestamped CSV logger for bilateral (left + right arm) sessions.

    One row per frame with side-prefixed angle columns. A side that was
    not tracked in a given frame is logged as empty cells with its
    ``<side>_tracked`` flag set to 0, so downstream analysis can
    distinguish occlusion from a genuine 0-degree reading.
    """

    SIDES  = _SIDES
    ANGLES = _ANGLE_FIELDS

    COLUMNS = (
        ["timestamp_s", "frame"]
        + [f"{s}_{a}" for s in _SIDES for a in _ANGLE_FIELDS]
        + [f"{s}_rotation_reliable" for s in _SIDES]
        + [f"{s}_tracked" for s in _SIDES]
        + ["filter_type", "calibrated"]
    )

    def __init__(
        self,
        output_dir:    str | Path = "outputs/logs",
        filter_type:   str = "kalman",
        session_label: str = "",
    ) -> None:
        self._dir        = Path(output_dir)
        self._filter     = filter_type
        self._label      = session_label
        self._file       = None
        self._writer     = None
        self._start_time = 0.0
        self._frame      = 0
        self._calibrated = False

    def start(self, calibrated: bool = False) -> Path:
        """Open the CSV file and write the header row."""
        self._dir.mkdir(parents=True, exist_ok=True)
        ts    = time.strftime("%Y%m%d_%H%M%S")
        label = f"_{self._label}" if self._label else ""
        path  = self._dir / f"angles_bilateral_{ts}{label}.csv"

        self._file       = open(path, "w", newline="")
        self._writer     = csv.DictWriter(self._file, fieldnames=self.COLUMNS)
        self._writer.writeheader()
        self._start_time = time.perf_counter()
        self._frame      = 0
        self._calibrated = calibrated
        return path

    def log(self, bilateral: BilateralArmAngles) -> None:
        """Write one row for the current frame."""
        if self._writer is None:
            raise RuntimeError("Call start() before log().")

        row = {
            "timestamp_s": round(time.perf_counter() - self._start_time, 4),
            "frame":       self._frame,
            "filter_type": self._filter,
            "calibrated":  int(self._calibrated),
        }
        for side in self.SIDES:
            angles: ArmAngles | None = getattr(bilateral, side)
            row[f"{side}_tracked"] = int(angles is not None)
            if angles is None:
                for a in self.ANGLES:
                    row[f"{side}_{a}"] = ""
                row[f"{side}_rotation_reliable"] = ""
            else:
                for a in self.ANGLES:
                    row[f"{side}_{a}"] = round(getattr(angles, a), 2)
                row[f"{side}_rotation_reliable"] = int(angles.rotation_reliable)

        self._writer.writerow(row)
        self._frame += 1

    def stop(self) -> None:
        """Flush and close the log file."""
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file   = None
            self._writer = None


# ---------------------------------------------------------------------------
# Real-time rolling plot (OpenCV-based, no matplotlib dependency)
# ---------------------------------------------------------------------------

class RollingAnglePlot:
    """
    Renders a live four-panel strip plot of joint angles onto an OpenCV
    BGR image.  No external window is created; the image is returned and
    composited by the caller into the main video display.

    Parameters
    ----------
    width : int
        Width of the output image in pixels.
    height : int
        Height of the output image in pixels.
    history_len : int
        Number of past frames shown in each panel.
    y_range : tuple[float, float]
        Y-axis range (degrees) for shoulder panels.
    elbow_range : tuple[float, float]
        Y-axis range (degrees) for the elbow panel.
    """

    # ASCII only — cv2.putText uses Hershey fonts which render non-ASCII
    # characters (e.g. the degree symbol) as "??".
    _LABELS = ["Flex (deg)", "Abd (deg)", "Rot (deg)", "Elbow (deg)"]
    _COLORS = [
        (100, 220, 100),   # green  — flexion
        (100, 160, 255),   # blue   — abduction
        (255, 180,  80),   # orange — rotation
        (220,  80, 220),   # purple — elbow
    ]
    _BG_COLOR  = (20, 20, 30)
    _GRID_COLOR = (60, 60, 70)
    _TEXT_COLOR = (200, 200, 200)

    def __init__(
        self,
        width:        int = 640,
        height:       int = 200,
        history_len:  int = 150,
        y_range:      tuple[float, float] = (-120.0, 120.0),
        elbow_range:  tuple[float, float] = (0.0, 160.0),
    ) -> None:
        self._w    = width
        self._h    = height
        self._n    = history_len
        self._yr   = y_range
        self._er   = elbow_range

        # One deque per DOF
        self._bufs = [deque(maxlen=history_len) for _ in range(4)]

    def update(self, angles: ArmAngles) -> None:
        """Push new angle values into the rolling buffers."""
        self._bufs[0].append(angles.shoulder_flexion)
        self._bufs[1].append(angles.shoulder_abduction)
        self._bufs[2].append(angles.shoulder_rotation)
        self._bufs[3].append(angles.elbow_flexion)

    def render(self) -> np.ndarray:
        """
        Render the rolling plot and return a BGR image of shape (height, width, 3).
        """
        img    = np.full((self._h, self._w, 3), self._BG_COLOR, dtype=np.uint8)
        n_dof  = 4
        pw     = self._w // n_dof          # panel width
        margin = 6

        for i, (buf, label, color) in enumerate(
            zip(self._bufs, self._LABELS, self._COLORS)
        ):
            x0 = i * pw
            x1 = x0 + pw
            y_min, y_max = self._er if i == 3 else self._yr

            # Panel background + border
            cv2.rectangle(img, (x0, 0), (x1 - 1, self._h - 1), self._GRID_COLOR, 1)

            # Zero line
            zero_y = self._h - 1 - int((0 - y_min) / (y_max - y_min) * (self._h - 1))
            zero_y = int(np.clip(zero_y, 0, self._h - 1))
            cv2.line(img, (x0, zero_y), (x1, zero_y), self._GRID_COLOR, 1)

            # Data trace
            if len(buf) >= 2:
                xs = np.linspace(x0 + margin, x1 - margin, self._n)
                ys_raw = np.array(list(buf))
                ys = self._h - 1 - (
                    np.clip((ys_raw - y_min) / (y_max - y_min), 0, 1)
                    * (self._h - 1)
                ).astype(int)

                n_pts = len(ys)
                xs_trim = xs[self._n - n_pts:]
                pts = np.column_stack([xs_trim.astype(int), ys]).reshape(-1, 1, 2)
                cv2.polylines(img, [pts], isClosed=False, color=color, thickness=2, lineType=cv2.LINE_AA)

                # Current value dot
                cv2.circle(img, (int(xs_trim[-1]), int(ys[-1])), 4, color, -1, cv2.LINE_AA)

            # Label + current value text
            current = buf[-1] if buf else 0.0
            cv2.putText(img, label, (x0 + 4, 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, self._TEXT_COLOR, 1, cv2.LINE_AA)
            cv2.putText(img, f"{current:+.1f}", (x0 + 4, self._h - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

        return img