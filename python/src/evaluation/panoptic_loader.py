"""
panoptic_loader.py — CMU Panoptic Studio Ground-Truth Angle Extractor
========================================================================

Parses CMU Panoptic Studio "coco19" 3D body keypoint files and computes
ground-truth BILATERAL (both arms) joint angles using the SAME anatomical
angle convention as the MonoArm vision pipeline (coordinate_frame.py +
angle_solver.py) and the same construction used for Human3.6M in
h36m_loader.py, so results are directly comparable. Left-arm angles reuse
h36m_loader._compute_side_gt(mirror=True) — the identical sagittal-plane
reflection angle_solver.compute_bilateral_angles() applies on the
prediction side.

Used as a real (non-registration-gated) substitute for Human3.6M when
Human3.6M account approval is unavailable. See:
    http://domedb.perception.cs.cmu.edu/
    https://github.com/CMU-Perceptual-Computing-Lab/panoptic-toolbox

Panoptic "coco19" Joint Order (0-indexed)
-------------------------------------------
    0: Neck            7: lKnee            14: rAnkle
    1: Nose             8: lAnkle            15: lEye
    2: BodyCenter        9: rShoulder        16: lEar
    3: lShoulder        10: rElbow           17: rEye
    4: lElbow           11: rWrist           18: rEar
    5: lWrist           12: rHip
    6: lHip             13: rKnee

Each frame's JSON file (body3DScene_<frame>.json) stores one or more
detected "bodies", each with a flat `joints19` array of
[x1,y1,z1,c1, x2,y2,z2,c2, ...] — 3D position (cm) + confidence per joint.

Coordinate Convention
----------------------
Panoptic world coordinates do not share Human3.6M's axis convention, but
the torso frame construction below is body-relative (built from hip/
shoulder positions themselves via Gram-Schmidt, exactly as in
coordinate_frame.py / h36m_loader.py), so it is invariant to the dataset's
absolute world-axis convention.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import numpy as np

from .h36m_loader import GTAngles, _normalize, _compute_side_gt

# --------------------------------------------------------------------------
# Panoptic coco19 bilateral-arm joint indices
# --------------------------------------------------------------------------
COCO19_NECK        = 0
COCO19_LSHOULDER   = 3
COCO19_LELBOW      = 4
COCO19_LWRIST      = 5
COCO19_LHIP        = 6
COCO19_RSHOULDER   = 9
COCO19_RELBOW      = 10
COCO19_RWRIST      = 11
COCO19_RHIP        = 12

MIN_JOINT_CONFIDENCE = 0.1


def _compute_gt_angles_coco19(
    joints: np.ndarray, frame_idx: int = 0, left_ok: bool = True,
) -> GTAngles | None:
    """
    Compute anatomically consistent bilateral arm angles from a coco19 skeleton.

    Builds the torso frame as in h36m_loader._compute_gt_angles() (just
    re-indexed for the Panoptic joint layout), then delegates the per-side
    angle math to h36m_loader._compute_side_gt() for both arms — the right
    arm is required (returns None if degenerate); the left arm is optional
    and simply left at its NaN default (via GTAngles) if degenerate.

    Parameters
    ----------
    joints : np.ndarray, shape (19, 3)
        3D joint positions (cm) for one detected body in one frame.
    """
    try:
        r_shoulder = joints[COCO19_RSHOULDER]
        r_elbow    = joints[COCO19_RELBOW]
        r_wrist    = joints[COCO19_RWRIST]
        l_shoulder = joints[COCO19_LSHOULDER]
        l_elbow    = joints[COCO19_LELBOW]
        l_wrist    = joints[COCO19_LWRIST]
        r_hip      = joints[COCO19_RHIP]
        l_hip      = joints[COCO19_LHIP]
    except IndexError:
        return None

    hip_mid      = 0.5 * (l_hip + r_hip)
    shoulder_mid = 0.5 * (l_shoulder + r_shoulder)

    y_cand = _normalize(shoulder_mid - hip_mid)
    x_cand = _normalize(l_shoulder - r_shoulder)

    if np.linalg.norm(y_cand) < 1e-9 or np.linalg.norm(x_cand) < 1e-9:
        return None

    x_orth = _normalize(x_cand - np.dot(x_cand, y_cand) * y_cand)
    z_axis = _normalize(np.cross(x_orth, y_cand))
    x_axis = _normalize(np.cross(y_cand, z_axis))

    R = np.column_stack([x_axis, y_cand, z_axis])

    right = _compute_side_gt(R, r_shoulder, r_elbow, r_wrist, mirror=False)
    if right is None:
        return None
    left = _compute_side_gt(R, l_shoulder, l_elbow, l_wrist, mirror=True) if left_ok else None

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


def parse_panoptic_frame(json_path: Path, body_id: int | None = None) -> GTAngles | None:
    """
    Parse one body3DScene_<frame>.json file and compute right-arm GT angles
    for a single body (the first detected body, or a specific `body_id`).
    """
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None

    bodies = data.get("bodies", [])
    if not bodies:
        return None

    body = bodies[0] if body_id is None else next(
        (b for b in bodies if b.get("id") == body_id), None
    )
    if body is None:
        return None

    flat = body.get("joints19")
    if not flat or len(flat) < 19 * 4:
        return None

    arr = np.array(flat, dtype=np.float64).reshape(19, 4)
    joints, confidence = arr[:, :3], arr[:, 3]

    # Right arm + torso landmarks are required (frame dropped if unreliable);
    # the left arm is optional — low left-side confidence degrades only the
    # left-arm fields (left at NaN) rather than dropping an otherwise-good
    # right-arm frame.
    needed = [COCO19_RSHOULDER, COCO19_RELBOW, COCO19_RWRIST,
              COCO19_LSHOULDER, COCO19_RHIP, COCO19_LHIP]
    if any(confidence[j] < MIN_JOINT_CONFIDENCE for j in needed):
        return None

    left_ok = (confidence[COCO19_LELBOW] >= MIN_JOINT_CONFIDENCE and
               confidence[COCO19_LWRIST] >= MIN_JOINT_CONFIDENCE)

    frame_idx = int(json_path.stem.split("_")[-1])
    return _compute_gt_angles_coco19(joints, frame_idx=frame_idx, left_ok=left_ok)


def iter_panoptic_sequence(
    pose_dir:  Path,
    body_id:   int | None = None,
    max_frames: int | None = None,
) -> Iterator[GTAngles]:
    """
    Iterate over all body3DScene_*.json files in a Panoptic
    hdPose3d_stage1_coco19/ directory in frame order, yielding GTAngles.
    """
    pose_dir = Path(pose_dir)
    yielded  = 0
    for json_path in sorted(pose_dir.glob("body3DScene_*.json")):
        gt = parse_panoptic_frame(json_path, body_id=body_id)
        if gt is not None:
            gt.subject = "panoptic"
            gt.action  = pose_dir.parent.name
            yield gt
            yielded += 1
            if max_frames is not None and yielded >= max_frames:
                return
