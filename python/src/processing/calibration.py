"""
calibration.py — Hybrid Joint Angle Calibration System
=======================================================

Implements a three-tier calibration pipeline that maps raw estimated
joint angles to the user's actual range of motion and to the Unity
avatar's muscle space.

Calibration Tiers
-----------------
1. Static Reference Pose Calibration (quick setup, ~30 seconds)
   User holds four reference poses while the system records angle readings.
   Reference poses:
       arm_down    → anatomical position; defines zero for all DOFs
       arm_forward → 90° flexion reference
       arm_side    → 90° abduction reference
       elbow_bent  → 90° elbow flexion reference
   From these, per-DOF (offset, scale) pairs are computed.

2. Dynamic Range-of-Motion Sweep (optional refinement, ~60 seconds)
   User slowly moves their arm through the full range of each DOF.
   The system records the observed min/max and uses this to set the
   linear mapping more accurately than the static method.

3. Auto-Adjustment (online, continuous)
   During normal operation, if a detected angle exceeds the calibrated
   range, the range is automatically expanded (one-sided). This ensures
   the avatar never hard-limits even for users with unusual flexibility.

Calibration Mapping
-------------------
The calibration maps raw estimated angles (degrees) to normalised
muscle values in Unity's HumanPoseHandler space [-1, +1]:

    muscle = (raw + offset) * scale

where:
    offset  = -raw_value_at_zero_pose
    scale   = 1.0 / (raw_range * 0.5)

The applied calibration converts raw angles into avatar-ready values,
stored in CalibrationData and reloaded between sessions via JSON.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from .angle_solver import ArmAngles


# ---------------------------------------------------------------------------
# Reference pose names
# ---------------------------------------------------------------------------
POSE_ARM_DOWN    = "arm_down"
POSE_ARM_FORWARD = "arm_forward"
POSE_ARM_SIDE    = "arm_side"
POSE_ELBOW_BENT  = "elbow_bent"

REQUIRED_POSES   = [POSE_ARM_DOWN, POSE_ARM_FORWARD, POSE_ARM_SIDE, POSE_ELBOW_BENT]

# ---------------------------------------------------------------------------
# Gain-fitting guards
# ---------------------------------------------------------------------------
# A calibration gain is fitted as expected_deg / observed_span. If the user
# held the reference pose poorly (e.g. arm_forward barely differs from
# arm_down), observed_span is tiny and the gain explodes — 90/5 = 18x — which
# would multiply every subsequent angle by that factor and drive the avatar
# (and any downstream exoskeleton reference signal) wildly out of range.
# We therefore require a minimum observed span before fitting a gain at all,
# and clamp the fitted gain to a plausible range.
MIN_REFERENCE_SPAN_DEG = 30.0   # below this, the pose was not held distinctly
MIN_GAIN               = 0.5
MAX_GAIN               = 2.0


def _fit_gain(
    observed_span: float,
    expected_deg: float = 90.0,
    dof_name: str = "",
    warnings: list[str] | None = None,
) -> float:
    """
    Fit a per-DOF calibration gain from one reference pose.

    Returns 1.0 (identity, i.e. trust the raw estimate) when the observed
    span is too small to fit a meaningful gain; otherwise returns
    expected_deg / observed_span clamped to [MIN_GAIN, MAX_GAIN]. Every
    rejection or clamp is printed and, when given, appended to ``warnings``
    so a UI can tell the operator which pose to repeat.
    """
    def _warn(msg: str) -> None:
        print(f"[calibration] {msg}")
        if warnings is not None:
            warnings.append(msg)

    if abs(observed_span) < MIN_REFERENCE_SPAN_DEG:
        _warn(f"Reference span for {dof_name or 'DOF'} was only "
              f"{observed_span:.1f} deg (< {MIN_REFERENCE_SPAN_DEG:.0f} deg); "
              f"the pose was likely not held distinctly. Left uncalibrated "
              f"(gain = 1.0) - repeat the pose more distinctly.")
        return 1.0

    gain = expected_deg / observed_span
    clamped = max(MIN_GAIN, min(MAX_GAIN, gain))
    if clamped != gain:
        _warn(f"Fitted gain {gain:.2f} for {dof_name or 'DOF'} is "
              f"outside [{MIN_GAIN}, {MAX_GAIN}]; clamped to {clamped:.2f}.")
    return clamped

# Expected angles at each reference pose (degrees, anatomical ground truth)
EXPECTED_ANGLES = {
    POSE_ARM_DOWN:    ArmAngles(shoulder_flexion=0,  shoulder_abduction=0,  shoulder_rotation=0,  elbow_flexion=0,  rotation_reliable=False),
    POSE_ARM_FORWARD: ArmAngles(shoulder_flexion=90, shoulder_abduction=0,  shoulder_rotation=0,  elbow_flexion=0,  rotation_reliable=False),
    POSE_ARM_SIDE:    ArmAngles(shoulder_flexion=0,  shoulder_abduction=90, shoulder_rotation=0,  elbow_flexion=0,  rotation_reliable=False),
    POSE_ELBOW_BENT:  ArmAngles(shoulder_flexion=0,  shoulder_abduction=0,  shoulder_rotation=0,  elbow_flexion=90, rotation_reliable=False),
}


@dataclass
class DOFCalibration:
    """Per-degree-of-freedom calibration parameters."""
    offset: float = 0.0    # degrees to add to raw value
    scale:  float = 1.0    # multiplier after offset
    min_raw: float = -180.0
    max_raw: float =  180.0


@dataclass
class CalibrationData:
    """
    Full calibration state for all four DOFs.
    Serialisable to/from JSON for persistence between sessions.
    """
    flexion:   DOFCalibration = field(default_factory=DOFCalibration)
    abduction: DOFCalibration = field(default_factory=DOFCalibration)
    rotation:  DOFCalibration = field(default_factory=DOFCalibration)
    elbow:     DOFCalibration = field(default_factory=DOFCalibration)
    calibrated: bool = False
    timestamp: str = ""

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @staticmethod
    def load(path: Path) -> "CalibrationData":
        with open(path, "r") as f:
            d = json.load(f)
        cal = CalibrationData()
        for dof in ("flexion", "abduction", "rotation", "elbow"):
            dof_d = d.get(dof, {})
            setattr(cal, dof, DOFCalibration(**dof_d))
        cal.calibrated = d.get("calibrated", False)
        cal.timestamp  = d.get("timestamp", "")
        return cal


class CalibrationManager:
    """
    Manages the three-tier calibration pipeline.

    Usage (static calibration)
    --------------------------
    >>> mgr = CalibrationManager()
    >>> mgr.begin_static()
    >>> for pose_name in mgr.next_pose():
    ...     # show user the pose instruction
    ...     # wait for user confirmation (e.g. key press)
    ...     mgr.capture_pose(current_angles)
    >>> mgr.finalise()
    >>> mgr.save("calibration.json")

    Usage (apply calibration)
    -------------------------
    >>> raw_angles = compute_arm_angles(landmarks)
    >>> cal_angles = mgr.apply(raw_angles)  # ready for UDP send
    """

    def __init__(self) -> None:
        self.data = CalibrationData()
        self._reference_captures: dict[str, ArmAngles] = {}
        self._sweep_samples: dict[str, list[float]] = {
            "flexion": [], "abduction": [], "rotation": [], "elbow": [],
        }
        self._current_pose: Optional[str] = None
        self._static_done  = False
        self._pose_queue   = list(REQUIRED_POSES)
        # Populated by finalise(): problems found while fitting the reference
        # poses. Empty means every axis was calibrated cleanly.
        self.warnings: list[str] = []

    # ------------------------------------------------------------------
    # Static calibration
    # ------------------------------------------------------------------

    def begin_static(self) -> None:
        """Start the static reference pose calibration sequence."""
        self._reference_captures.clear()
        self._pose_queue  = list(REQUIRED_POSES)
        self._static_done = False

    def next_pose(self) -> Optional[str]:
        """
        Return the next pose name to capture, or None if all are done.
        Advance the internal pointer.
        """
        if not self._pose_queue:
            return None
        self._current_pose = self._pose_queue.pop(0)
        return self._current_pose

    def capture_pose(self, angles: ArmAngles, n_samples: int = 1) -> None:
        """
        Record angle readings for the current reference pose.
        Accumulate multiple samples and average if called more than once.
        """
        if self._current_pose is None:
            raise RuntimeError("Call next_pose() before capture_pose().")

        prev = self._reference_captures.get(self._current_pose)
        if prev is None or n_samples == 1:
            self._reference_captures[self._current_pose] = angles
        else:
            # Running average for multi-sample capture
            w = 1.0 / (n_samples + 1)
            self._reference_captures[self._current_pose] = ArmAngles(
                shoulder_flexion   = (1 - w) * prev.shoulder_flexion   + w * angles.shoulder_flexion,
                shoulder_abduction = (1 - w) * prev.shoulder_abduction + w * angles.shoulder_abduction,
                shoulder_rotation  = (1 - w) * prev.shoulder_rotation  + w * angles.shoulder_rotation,
                elbow_flexion      = (1 - w) * prev.elbow_flexion      + w * angles.elbow_flexion,
                rotation_reliable  = angles.rotation_reliable,
            )

    def finalise(self) -> None:
        """
        Compute calibration parameters from the captured reference poses.

        For each DOF, the offset zeroes the arm_down reading, and the
        scale maps the observed range to the expected anatomical range.
        """
        if POSE_ARM_DOWN not in self._reference_captures:
            raise RuntimeError("arm_down pose must be captured before finalising.")

        down = self._reference_captures[POSE_ARM_DOWN]
        self.warnings = []

        # Flexion: arm_down = 0°, arm_forward = 90°
        self.data.flexion.offset = -down.shoulder_flexion
        if POSE_ARM_FORWARD in self._reference_captures:
            fwd_raw = (self._reference_captures[POSE_ARM_FORWARD].shoulder_flexion
                       + self.data.flexion.offset)
            self.data.flexion.scale = _fit_gain(fwd_raw, 90.0, "shoulder flexion", self.warnings)
        self.data.flexion.min_raw = down.shoulder_flexion - 30
        self.data.flexion.max_raw = self._reference_captures.get(
            POSE_ARM_FORWARD, down).shoulder_flexion + 10

        # Abduction: arm_down = 0°, arm_side = 90°
        self.data.abduction.offset = -down.shoulder_abduction
        if POSE_ARM_SIDE in self._reference_captures:
            side_raw = (self._reference_captures[POSE_ARM_SIDE].shoulder_abduction
                        + self.data.abduction.offset)
            self.data.abduction.scale = _fit_gain(side_raw, 90.0, "shoulder abduction", self.warnings)
        self.data.abduction.min_raw = down.shoulder_abduction - 15
        self.data.abduction.max_raw = self._reference_captures.get(
            POSE_ARM_SIDE, down).shoulder_abduction + 10

        # Rotation: zero at arm_down
        self.data.rotation.offset = -down.shoulder_rotation
        self.data.rotation.scale  = 1.0  # rotation range not directly calibrated

        # Elbow: arm_down = 0°, elbow_bent = 90°
        self.data.elbow.offset = -down.elbow_flexion
        if POSE_ELBOW_BENT in self._reference_captures:
            bent_raw = (self._reference_captures[POSE_ELBOW_BENT].elbow_flexion
                        + self.data.elbow.offset)
            self.data.elbow.scale = _fit_gain(bent_raw, 90.0, "elbow flexion", self.warnings)
        self.data.elbow.min_raw = down.elbow_flexion - 5
        self.data.elbow.max_raw = self._reference_captures.get(
            POSE_ELBOW_BENT, down).elbow_flexion + 10

        self.data.calibrated = True
        self.data.timestamp  = time.strftime("%Y-%m-%dT%H:%M:%S")
        self._static_done    = True

    # ------------------------------------------------------------------
    # Dynamic sweep calibration
    # ------------------------------------------------------------------

    def record_sweep_sample(self, angles: ArmAngles) -> None:
        """
        Record one angle sample during a dynamic range-of-motion sweep.
        Call this every frame while the user moves their arm freely.
        """
        self._sweep_samples["flexion"].append(angles.shoulder_flexion)
        self._sweep_samples["abduction"].append(angles.shoulder_abduction)
        self._sweep_samples["rotation"].append(angles.shoulder_rotation)
        self._sweep_samples["elbow"].append(angles.elbow_flexion)

    def finalise_sweep(self) -> None:
        """
        Refine calibration range using recorded sweep samples.
        Must be called after finalise() (static calibration must exist first).
        """
        if not self.data.calibrated:
            raise RuntimeError("Run static calibration before dynamic sweep.")

        for dof, key in (
            (self.data.flexion,   "flexion"),
            (self.data.abduction, "abduction"),
            (self.data.rotation,  "rotation"),
            (self.data.elbow,     "elbow"),
        ):
            samples = self._sweep_samples[key]
            if len(samples) < 10:
                continue
            dof.min_raw = min(samples) - 5.0
            dof.max_raw = max(samples) + 5.0

        # Clear for next sweep
        self._sweep_samples = {k: [] for k in self._sweep_samples}

    # ------------------------------------------------------------------
    # Apply calibration
    # ------------------------------------------------------------------

    def apply(self, angles: ArmAngles) -> ArmAngles:
        """
        Apply calibration to raw angles.

        Performs auto-range expansion if any angle exceeds the recorded
        calibrated range (prevents avatar freezing at limits).

        Parameters
        ----------
        angles : ArmAngles
            Raw estimated angles from the pipeline.

        Returns
        -------
        ArmAngles
            Calibrated angles. If not yet calibrated, returns unchanged.
        """
        if not self.data.calibrated:
            return angles

        def _apply_dof(raw: float, cal: DOFCalibration) -> float:
            # Auto-expand range
            if raw < cal.min_raw:
                cal.min_raw = raw - 5.0
            if raw > cal.max_raw:
                cal.max_raw = raw + 5.0
            return (raw + cal.offset) * cal.scale

        return ArmAngles(
            shoulder_flexion   = _apply_dof(angles.shoulder_flexion,   self.data.flexion),
            shoulder_abduction = _apply_dof(angles.shoulder_abduction, self.data.abduction),
            shoulder_rotation  = _apply_dof(angles.shoulder_rotation,  self.data.rotation),
            elbow_flexion      = _apply_dof(angles.elbow_flexion,      self.data.elbow),
            rotation_reliable  = angles.rotation_reliable,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save calibration data to a JSON file."""
        self.data.save(Path(path))

    def load(self, path: str | Path) -> None:
        """Load calibration data from a JSON file."""
        self.data = CalibrationData.load(Path(path))