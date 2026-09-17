"""
h36m_loader.py — Human3.6M Ground-Truth Angle Extractor
=========================================================

Parses Human3.6M skeleton files and computes ground-truth arm joint angles
using the SAME anatomical angle convention as the MonoArm vision pipeline
(coordinate_frame.py + angle_solver.py), ensuring a like-for-like comparison.

Human3.6M Dataset Format
--------------------------
Two distinct on-disk formats exist and must not be confused (the full
discussion, and the forward-kinematics conversion, live in h36m_skeleton.py):

    96 values/row — D3_Positions: 32 joints × 3 Cartesian coordinates in
                    millimetres, world frame. Ships with the registration-
                    gated release and pairs 1:1 with the video.

    99 values/row — exponential map: 3 root-translation values plus 32
                    joints × 3 axis-angle rotation vectors. This is the
                    mirrored ``h3.6m.zip`` used by the motion-prediction
                    literature; it has no video. These are rotations, not
                    positions, and are converted by forward kinematics
                    before any angle is computed.

``row_to_joint_positions`` dispatches on row width, so both are accepted and
both yield (32, 3) positions in millimetres.

H3.6M Right-Arm Joint Indices (0-indexed)
-------------------------------------------
H3.6M uses a different skeleton topology from MediaPipe. The mapping
to right-arm joints used for angle computation is:

    Index  1 → RHip        (right hip, for torso frame)
    Index  6 → LHip        (left hip, for torso frame)
    Index 14 → Spine/Chest (superior reference)
    Index 17 → LShoulder
    Index 25 → RShoulder   ← right arm origin
    Index 26 → RElbow      ← right arm mid
    Index 27 → RWrist      ← right arm end

Note: H3.6M joint indices vary by publication; the indices above follow
the standard 32-joint skeleton released at:
    http://vision.imar.ro/human3.6m/description.php

Coordinate Convention
---------------------
H3.6M coordinates are in millimetres with:
    X → lateral (right)
    Y → vertical (up)
    Z → anterior (toward camera)

This is a right-hand system with Y-up, analogous to the torso frame we
construct in coordinate_frame.py. We build the same ISB-aligned torso
frame from H3.6M joint positions so the angle decomposition is identical.

Anatomical Angle Computation
------------------------------
Rather than using the old unsigned pitch/roll/yaw approach from the
original build_h36m_dataset.py (which mixed ISB conventions and produced
unsigned angles), we compute angles using the same ZXY Euler decomposition
as angle_solver.py but operating directly on H3.6M 3D coordinates.

For the right arm:
    shoulder_flexion   = atan2(-vz_torso, -vy_torso)        [degrees]
    shoulder_abduction = arcsin(-vx_torso)                   [degrees, right arm]
    shoulder_rotation  = forearm proxy (when elbow bent)     [degrees]
    elbow_flexion      = arccos(û_arm · û_forearm)           [degrees]

References
----------
Human3.6M: Ionescu et al. (2014). Human3.6M: Large Scale Datasets and
    Predictive Methods for 3D Human Sensing in Natural Environments.
    IEEE TPAMI 36(7):1325-1339.
    http://vision.imar.ro/human3.6m/
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from .h36m_skeleton import N_JOINTS, forward_kinematics
from ..processing.coordinate_frame import MIN_TORSO_CONDITIONING

# --------------------------------------------------------------------------
# H3.6M arm joint indices (standard 32-joint skeleton)
# --------------------------------------------------------------------------
H36M_RSHOULDER = 25
H36M_RELBOW    = 26
H36M_RWRIST    = 27
H36M_LSHOULDER = 17
H36M_LELBOW    = 18
H36M_LWRIST    = 19
H36M_RHIP      = 1
H36M_LHIP      = 6
H36M_CHEST     = 14   # spine1 / chest — superior torso reference

# --------------------------------------------------------------------------
# Minimum elbow flexion (degrees) for shoulder rotation to be reliable
# --------------------------------------------------------------------------
ROT_RELIABLE_THRESHOLD = 25.0


@dataclass
class GTAngles:
    """
    Ground-truth bilateral arm joint angles from H3.6M (degrees).

    The unprefixed fields are the RIGHT arm (kept for backward
    compatibility); left-arm fields carry the ``left_`` prefix and use
    the mirrored anatomical convention (abduction + = away from midline,
    internal rotation + for both sides).
    """
    shoulder_flexion:   float
    shoulder_abduction: float
    shoulder_rotation:  float
    elbow_flexion:      float
    rotation_reliable:  bool
    frame_idx:          int = 0
    subject:            str = ""
    action:             str = ""
    left_shoulder_flexion:   float = float("nan")
    left_shoulder_abduction: float = float("nan")
    left_shoulder_rotation:  float = float("nan")
    left_elbow_flexion:      float = float("nan")
    left_rotation_reliable:  bool = False


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def _compute_side_gt(
    R: np.ndarray,
    p_shoulder: np.ndarray,
    p_elbow: np.ndarray,
    p_wrist: np.ndarray,
    mirror: bool,
) -> tuple[float, float, float, float, bool] | None:
    """
    Compute (flexion, abduction, rotation, elbow, rot_reliable) for one arm.

    `mirror=True` (left arm) reflects the torso-frame vectors across the
    sagittal plane so the right-arm formulas yield mirrored anatomical
    conventions — identical to angle_solver._compute_side_angles().
    """
    v_upper_arm_world = p_elbow - p_shoulder
    v_forearm_world   = p_wrist - p_elbow

    if np.linalg.norm(v_upper_arm_world) < 1e-9:
        return None

    v_ua_torso = R.T @ v_upper_arm_world      # [x_lat, y_sup, z_ant]
    v_fa_torso = R.T @ v_forearm_world

    if mirror:
        v_ua_torso = v_ua_torso * np.array([-1.0, 1.0, 1.0])
        v_fa_torso = v_fa_torso * np.array([-1.0, 1.0, 1.0])

    u_ua = _normalize(v_ua_torso)
    vx, vy, vz = u_ua[0], u_ua[1], u_ua[2]

    # Shoulder flexion-extension: atan2(-vz, -vy)
    flexion_deg = math.degrees(math.atan2(-vz, -vy))

    # Shoulder abduction-adduction (right-arm convention → negate vx)
    vx_clamped    = float(np.clip(-vx, -1.0, 1.0))
    abduction_deg = math.degrees(math.asin(vx_clamped))

    # Elbow flexion
    u_fa = _normalize(v_fa_torso) if np.linalg.norm(v_fa_torso) > 1e-9 else v_fa_torso
    cos_ang   = float(np.clip(np.dot(u_ua, u_fa), -1.0, 1.0))
    elbow_deg = max(0.0, math.degrees(math.acos(cos_ang)))

    # Shoulder rotation (forearm proxy)
    rotation_deg = 0.0
    rot_reliable = elbow_deg >= ROT_RELIABLE_THRESHOLD

    if rot_reliable and np.linalg.norm(v_fa_torso) > 1e-9:
        f_perp = u_fa - np.dot(u_fa, u_ua) * u_ua
        f_norm = np.linalg.norm(f_perp)
        if f_norm > 1e-9:
            f_perp /= f_norm
            ref = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(u_ua, ref)) > 0.9:
                ref = np.array([0.0, 0.0, 1.0])
            e1 = _normalize(np.cross(u_ua, ref))
            e2 = np.cross(u_ua, e1)
            rotation_deg = math.degrees(math.atan2(
                float(np.dot(f_perp, e1)),
                float(np.dot(f_perp, e2)),
            ))

    return flexion_deg, abduction_deg, rotation_deg, elbow_deg, rot_reliable


def _compute_gt_angles(joints: np.ndarray, frame_idx: int = 0) -> GTAngles | None:
    """
    Compute anatomically consistent arm joint angles from H3.6M joint positions.

    Mirrors the pipeline in angle_solver.compute_arm_angles() but operates
    directly on H3.6M 3D coordinates (metres or mm — unit doesn't matter
    since we only use directions).

    Parameters
    ----------
    joints : np.ndarray, shape (33, 3)
        3D joint positions for one frame in H3.6M convention.

    Returns
    -------
    GTAngles or None if skeleton is degenerate.
    """
    try:
        r_shoulder = joints[H36M_RSHOULDER]
        r_elbow    = joints[H36M_RELBOW]
        r_wrist    = joints[H36M_RWRIST]
        l_shoulder = joints[H36M_LSHOULDER]
        l_elbow    = joints[H36M_LELBOW]
        l_wrist    = joints[H36M_LWRIST]
        r_hip      = joints[H36M_RHIP]
        l_hip      = joints[H36M_LHIP]
    except IndexError:
        return None

    # ── Step 1: Build torso frame (same Gram-Schmidt as coordinate_frame.py) ──
    hip_mid      = 0.5 * (l_hip + r_hip)
    shoulder_mid = 0.5 * (l_shoulder + r_shoulder)

    # Y: hip → shoulder (superior)
    y_cand = _normalize(shoulder_mid - hip_mid)
    # X: right_shoulder → left_shoulder (to user's left)
    x_cand = _normalize(l_shoulder - r_shoulder)

    if np.linalg.norm(y_cand) < 1e-9 or np.linalg.norm(x_cand) < 1e-9:
        return None

    # Gram-Schmidt. ||x_orth|| before normalising is the sine of the angle
    # between the shoulder axis and the spine; when that collapses the two
    # candidates no longer span a plane and the frame degenerates into noise.
    # Rejecting here mirrors coordinate_frame.build_torso_frame() so ground
    # truth and predictions are discarded under identical conditions — the
    # dropped frame becomes NaN and is excluded per-joint, never fabricated.
    x_orth_raw = x_cand - np.dot(x_cand, y_cand) * y_cand
    if np.linalg.norm(x_orth_raw) < MIN_TORSO_CONDITIONING:
        return None

    x_orth = _normalize(x_orth_raw)
    z_axis = _normalize(np.cross(x_orth, y_cand))
    x_axis = _normalize(np.cross(y_cand, z_axis))

    # R: columns are [x_axis, y_axis, z_axis]
    R = np.column_stack([x_axis, y_cand, z_axis])

    # ── Step 2: Per-side angle computation (right required, left optional) ────
    right = _compute_side_gt(R, r_shoulder, r_elbow, r_wrist, mirror=False)
    if right is None:
        return None
    left = _compute_side_gt(R, l_shoulder, l_elbow, l_wrist, mirror=True)

    r_flex, r_abd, r_rot, r_elb, r_rel = right

    gt = GTAngles(
        shoulder_flexion   = round(r_flex, 4),
        shoulder_abduction = round(r_abd,  4),
        shoulder_rotation  = round(r_rot,  4),
        elbow_flexion      = round(r_elb,  4),
        rotation_reliable  = r_rel,
        frame_idx          = frame_idx,
    )
    if left is not None:
        l_flex, l_abd, l_rot, l_elb, l_rel = left
        gt.left_shoulder_flexion   = round(l_flex, 4)
        gt.left_shoulder_abduction = round(l_abd,  4)
        gt.left_shoulder_rotation  = round(l_rot,  4)
        gt.left_elbow_flexion      = round(l_elb,  4)
        gt.left_rotation_reliable  = l_rel

    return gt


def parse_h36m_file(txt_path: Path) -> list[GTAngles]:
    """
    Parse one H3.6M skeleton .txt file and extract right-arm GT angles.

    Two on-disk row widths occur in the wild and they mean different things
    (see h36m_skeleton.py):

        96 values — D3_Positions: 32 joints × 3 Cartesian coordinates (mm),
                    used directly.
        99 values — exponential map: 3 root-translation values plus 32 joints
                    × 3 axis-angle rotation vectors. These are rotations, not
                    positions, so the row is passed through forward kinematics
                    to recover 3D joint positions before any angle is computed.

    Parameters
    ----------
    txt_path : Path
        Path to the .txt skeleton file (e.g. S1/walking_1.txt).

    Returns
    -------
    list[GTAngles]
        One entry per successfully parsed frame. Malformed rows skipped.
    """
    results: list[GTAngles] = []
    action  = txt_path.stem
    subject = txt_path.parent.name

    lines = txt_path.read_text(encoding="utf-8").strip().split("\n")
    for frame_idx, line in enumerate(lines):
        vals = [v.strip() for v in line.strip().split(",") if v.strip()]
        try:
            row = np.array([float(v) for v in vals], dtype=np.float64)
            joints_xyz = row_to_joint_positions(row)
        except ValueError:
            continue

        gt = _compute_gt_angles(joints_xyz, frame_idx=frame_idx)
        if gt is not None:
            gt.subject = subject
            gt.action  = action
            results.append(gt)

    return results


def gt_angles_from_row(
    row:       np.ndarray,
    frame_idx: int = 0,
    subject:   str = "",
    action:    str = "",
) -> GTAngles | None:
    """
    Compute ground-truth angles from one raw H3.6M row (96 or 99 values).

    Returns None for degenerate skeletons, exactly as ``parse_h36m_file`` does.
    Callers that pair ground truth with video frames must drop the paired frame
    whenever this returns None, or the pairing silently shifts.
    """
    try:
        joints_xyz = row_to_joint_positions(row)
    except ValueError:
        return None

    gt = _compute_gt_angles(joints_xyz, frame_idx=frame_idx)
    if gt is not None:
        gt.subject = subject
        gt.action  = action
    return gt


def row_to_joint_positions(row: np.ndarray) -> np.ndarray:
    """
    Turn one raw H3.6M row into (32, 3) joint positions in millimetres.

    Dispatches on row width: 96 values are already Cartesian (D3_Positions),
    99 values are exponential-map rotations and need forward kinematics.

    Raises
    ------
    ValueError
        If the row is neither 96 nor 99 values wide.
    """
    row = np.asarray(row, dtype=np.float64).ravel()
    if row.size == 96:
        return row.reshape(N_JOINTS, 3)
    if row.size == 99:
        return forward_kinematics(row)
    raise ValueError(
        f"unrecognised H3.6M row width {row.size}; expected 96 (D3_Positions) "
        f"or 99 (exponential map)"
    )


def iter_h36m_dataset(
    h36m_dir:  Path,
    subjects:  list[str] | None = None,
    max_frames: int | None = None,
    per_subject: bool = True,
) -> Iterator[GTAngles]:
    """
    Iterate over all H3.6M skeleton files and yield GTAngles.

    Parameters
    ----------
    h36m_dir : Path
        Root directory containing S1/, S5/, S6/, S7/, S8/, S9/, S11/ folders.
    subjects : list[str], optional
        Which subjects to include. Defaults to all standard subjects.
    max_frames : int, optional
        Cap on total frames yielded. Useful for quick testing.
    per_subject : bool
        When True (default) the max_frames budget is divided evenly
        across subjects, so a capped run still covers EVERY requested
        subject — required for the test-split and LOSO protocols.
        When False the cap is global and fills from the first subject.

    Yields
    ------
    GTAngles
    """
    subjects = subjects or ["S1", "S5", "S6", "S7", "S8", "S9", "S11"]
    h36m_dir = Path(h36m_dir)
    yielded  = 0

    subject_budget = None
    if max_frames is not None and per_subject and subjects:
        subject_budget = max(1, max_frames // len(subjects))

    for subject in subjects:
        subj_dir = h36m_dir / subject
        if not subj_dir.exists():
            print(f"  [h36m_loader] SKIP: {subj_dir} not found")
            continue
        yielded_subject = 0
        for txt_path in sorted(subj_dir.glob("*.txt")):
            for gt in parse_h36m_file(txt_path):
                yield gt
                yielded += 1
                yielded_subject += 1
                if max_frames is not None and yielded >= max_frames:
                    return
                if subject_budget is not None and yielded_subject >= subject_budget:
                    break
            else:
                continue
            break
