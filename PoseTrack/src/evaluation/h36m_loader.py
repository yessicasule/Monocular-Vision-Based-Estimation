"""
h36m_loader.py — Human3.6M Ground-Truth Angle Extractor
=========================================================

Parses Human3.6M skeleton files and computes ground-truth arm joint angles
using the SAME anatomical angle convention as the MonoArm vision pipeline
(coordinate_frame.py + angle_solver.py), ensuring a like-for-like comparison.

Human3.6M Dataset Format
--------------------------
The release used here (``h3.6m/dataset/S*/<action>_<n>.txt``, the
motion-prediction distribution of Fragkiadaki et al. 2015 / Martinez et al.
2017) stores one frame per row as 99 comma-separated values:

    3 root position  +  32 joints × 3 exponential-map rotation values

These are joint ROTATIONS in radians, NOT 3D positions. Reading them as
xyz coordinates yields a "skeleton" whose bone lengths change every frame
(the upper arm of S1/directions_1 varies 2.0–5.9 units), so joint positions
are reconstructed with forward kinematics over the H3.6M bone hierarchy
(:func:`h36m_expmap_to_xyz`) before any angle is computed.

H3.6M Right-Arm Joint Indices (0-indexed, 32-joint FK skeleton)
----------------------------------------------------------------
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
Forward kinematics yields millimetres in the skeleton's own right-handed
frame (rest pose: X → subject's left, Y → up, Z → anterior). The common
``xyz[:, [0, 2, 1]]`` axis swap used for plotting is deliberately NOT
applied: swapping two axes is a reflection, which would mirror the subject
and flip the sign of flexion. The torso frame is rebuilt from the joints
exactly as coordinate_frame.py does, so the global orientation is irrelevant.

Anatomical Angle Computation
------------------------------
Rather than using the old unsigned pitch/roll/yaw approach from the
original build_h36m_dataset.py (which mixed ISB conventions and produced
unsigned angles), we compute angles using the same ZXY Euler decomposition
as angle_solver.py but operating directly on H3.6M 3D coordinates.

For the right arm:
    shoulder_flexion   = atan2(vz_torso, -vy_torso)         [degrees, + = forward]
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

# --------------------------------------------------------------------------
# H3.6M arm joint indices (standard 33-joint skeleton)
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

# --------------------------------------------------------------------------
# H3.6M bone hierarchy for forward kinematics (32 joints)
# Parent indices and rest-pose bone offsets (mm, in the parent's frame) as
# published with the exponential-map release (Martinez et al. 2017,
# src/forward_kinematics.py).
# --------------------------------------------------------------------------
_H36M_PARENT = np.array([
    0, 1, 2, 3, 4, 5, 1, 7, 8, 9, 10, 1, 12, 13, 14, 15, 13,
    17, 18, 19, 20, 21, 20, 23, 13, 25, 26, 27, 28, 29, 28, 31,
]) - 1

_H36M_OFFSET = np.array([
    0.000000, 0.000000, 0.000000,      -132.948591, 0.000000, 0.000000,
    0.000000, -442.894612, 0.000000,   0.000000, -454.206447, 0.000000,
    0.000000, 0.000000, 162.767078,    0.000000, 0.000000, 74.999437,
    132.948826, 0.000000, 0.000000,    0.000000, -442.894413, 0.000000,
    0.000000, -454.206590, 0.000000,   0.000000, 0.000000, 162.767426,
    0.000000, 0.000000, 74.999948,     0.000000, 0.100000, 0.000000,
    0.000000, 233.383263, 0.000000,    0.000000, 257.077681, 0.000000,
    0.000000, 121.134938, 0.000000,    0.000000, 115.002227, 0.000000,
    0.000000, 257.077681, 0.000000,    0.000000, 151.034226, 0.000000,
    0.000000, 278.882773, 0.000000,    0.000000, 251.733451, 0.000000,
    0.000000, 0.000000, 0.000000,      0.000000, 0.000000, 99.999627,
    0.000000, 100.000188, 0.000000,    0.000000, 0.000000, 0.000000,
    0.000000, 257.077681, 0.000000,    0.000000, 151.031437, 0.000000,
    0.000000, 278.892924, 0.000000,    0.000000, 251.728680, 0.000000,
    0.000000, 0.000000, 0.000000,      0.000000, 0.000000, 99.999888,
    0.000000, 137.499922, 0.000000,    0.000000, 0.000000, 0.000000,
]).reshape(32, 3)


def _expmap_to_rotmat(r: np.ndarray) -> np.ndarray:
    """Rodrigues' formula: exponential-map vector → 3×3 rotation matrix."""
    theta = float(np.linalg.norm(r))
    if theta < 1e-12:
        return np.eye(3)
    k = r / theta
    K = np.array([[0.0, -k[2], k[1]],
                  [k[2], 0.0, -k[0]],
                  [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)


def h36m_expmap_to_xyz(row: np.ndarray) -> np.ndarray:
    """
    Forward kinematics for one 99-value exponential-map frame.

    Parameters
    ----------
    row : np.ndarray, shape (99,)
        Root position (3) followed by 32 joints × 3 exp-map rotation values.

    Returns
    -------
    np.ndarray, shape (32, 3)
        Joint positions in millimetres (row-vector convention, matching the
        reference implementation: x_child = offset @ R_parent + x_parent).
    """
    row = np.asarray(row, dtype=np.float64)
    if row.shape[0] < 99:
        raise ValueError(f"expected 99 values per frame, got {row.shape[0]}")
    xyz = np.zeros((32, 3))
    rot = [None] * 32
    for i in range(32):
        R_local = _expmap_to_rotmat(row[3 + 3 * i: 6 + 3 * i])
        p = _H36M_PARENT[i]
        if p < 0:
            xyz[i] = _H36M_OFFSET[i] + row[0:3]
            rot[i] = R_local
        else:
            xyz[i] = _H36M_OFFSET[i] @ rot[p] + xyz[p]
            rot[i] = R_local @ rot[p]
    return xyz


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

    # Shoulder flexion-extension: atan2(vz, -vy)  (+ = forward)
    flexion_deg = math.degrees(math.atan2(vz, -vy))

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

    # Gram-Schmidt
    x_orth = _normalize(x_cand - np.dot(x_cand, y_cand) * y_cand)
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

    H3.6M .txt format: one row per frame, 99 comma-separated values — root
    position plus 32 exponential-map joint rotations. Joint positions are
    reconstructed by forward kinematics before angles are computed.

    Parameters
    ----------
    txt_path : Path
        Path to the .txt skeleton file (e.g. S1/Directions 1.txt).

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
        if len(vals) < 99:
            continue
        try:
            expmap     = np.array([float(v) for v in vals[:99]], dtype=np.float64)
            joints_xyz = h36m_expmap_to_xyz(expmap)
            gt = _compute_gt_angles(joints_xyz, frame_idx=frame_idx)
            if gt is not None:
                gt.subject = subject
                gt.action  = action
                results.append(gt)
        except (ValueError, Exception):
            continue

    return results


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
