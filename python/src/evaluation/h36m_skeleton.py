"""
h36m_skeleton.py — Human3.6M Skeleton Definition and Forward Kinematics
=======================================================================

The Human3.6M distribution ships body motion in two different forms, and
they are NOT interchangeable:

    1. D3_Positions (.cdf)    — 32 joints x 3 Cartesian coordinates (mm)
                                in the world frame; 96 values per frame.
                                This is what ``MyPoseFeatures/D3_Positions``
                                holds in the registration-gated release from
                                http://vision.imar.ro/human3.6m/, and it is
                                the form that pairs 1:1 with the video.

    2. Exponential map (.txt) — 3 root-translation values followed by 32
                                joints x 3 exponential-map (axis-angle)
                                rotation vectors; 99 values per frame. This
                                is the widely-mirrored ``h3.6m.zip`` used by
                                the motion-prediction literature. It ships
                                with no video.

Rotation vectors are not positions. Reshaping a 99-value exponential-map row
to (33, 3) and reading joints out of it yields numerically valid but
anatomically meaningless coordinates. To get 3D positions out of the
exponential-map release you must run forward kinematics over the H3.6M
skeleton, which is what this module implements.

Skeleton
--------
32 joints, parent-indexed, with fixed bone offsets in millimetres. The offset
table and channel layout follow the original H3.6M export used by the
standard motion-prediction preprocessing pipeline.

Joint indices of interest to MonoArm (identical between D3_Positions and the
forward-kinematics output of this module):

    0  Hip (root)      11 Spine        17 LShoulder    25 RShoulder
    1  RHip            12 Spine1       18 LElbow       26 RElbow
    6  LHip            13 Thorax       19 LWrist       27 RWrist

References
----------
Ionescu et al. (2014). Human3.6M: Large Scale Datasets and Predictive Methods
    for 3D Human Sensing in Natural Environments. IEEE TPAMI 36(7):1325-1339.
"""

from __future__ import annotations

import numpy as np

N_JOINTS = 32

# Parent index of each joint (-1 for the root).
PARENT = np.array([
    0,  1,  2,  3,  4,  5,  1,  7,  8,  9, 10,  1, 12, 13, 14, 15,
   13, 17, 18, 19, 20, 21, 20, 23, 13, 25, 26, 27, 28, 29, 28, 31,
], dtype=np.int64) - 1

# Fixed bone offset of each joint from its parent, in millimetres.
OFFSET = np.array([
       0.000000,    0.000000,   0.000000,
    -132.948591,    0.000000,   0.000000,
       0.000000, -442.894612,   0.000000,
       0.000000, -454.206447,   0.000000,
       0.000000,    0.000000, 162.767078,
       0.000000,    0.000000,  74.999437,
     132.948826,    0.000000,   0.000000,
       0.000000, -442.894413,   0.000000,
       0.000000, -454.206590,   0.000000,
       0.000000,    0.000000, 162.767426,
       0.000000,    0.000000,  74.999948,
       0.000000,    0.100000,   0.000000,
       0.000000,  233.383263,   0.000000,
       0.000000,  257.077681,   0.000000,
       0.000000,  121.134938,   0.000000,
       0.000000,  115.002227,   0.000000,
       0.000000,  257.077681,   0.000000,
       0.000000,  151.034226,   0.000000,
       0.000000,  278.882773,   0.000000,
       0.000000,  251.733451,   0.000000,
       0.000000,    0.000000,   0.000000,
       0.000000,    0.000000,  99.999627,
       0.000000,  100.000188,   0.000000,
       0.000000,    0.000000,   0.000000,
       0.000000,  257.077681,   0.000000,
       0.000000,  151.031437,   0.000000,
       0.000000,  278.892924,   0.000000,
       0.000000,  251.728680,   0.000000,
       0.000000,    0.000000,   0.000000,
       0.000000,    0.000000,  99.999888,
       0.000000,  137.499922,   0.000000,
       0.000000,    0.000000,   0.000000,
], dtype=np.float64).reshape(N_JOINTS, 3)

# Exponential-map channel triplets (0-indexed into the 99-value row). Values
# 0..2 are the root translation; joint i's rotation occupies 3 + 3i .. 5 + 3i.
EXPMAP_IND = np.split(np.arange(4, 100) - 1, N_JOINTS)

# Root translation channels (0-indexed).
ROOT_TRANSLATION_IND = np.array([0, 1, 2])

# NOTE ON PER-JOINT TRANSLATION CHANNELS
# --------------------------------------
# The widely-copied reference implementation of this forward-kinematics routine
# carries a `rotInd` table giving a translation triplet for almost every joint,
# and adds that translation to the bone offset. That table describes a BVH-style
# channel layout, not the 99-value format, and its indices collide with the
# rotation channels: rotInd[19] is [53, 54, 52], which 0-indexed is
# [52, 53, 51] — exactly the rotation channels of joint 16. Applying it feeds
# one joint's rotation values in as another joint's bone translation.
#
# The effect is small but systematic: on S1/walking_1 it stretches bones by up
# to 2.13 mm and gives them a per-bone standard deviation of 0.20 mm across
# frames, in a skeleton that is rigid by construction. Using the root
# translation alone leaves bone lengths constant to 2.3e-13 mm.
#
# So only the root translates here. Every other joint contributes rotation
# only, and the skeleton stays exactly rigid.

# Below this rotation magnitude the axis is numerically undefined and the
# rotation is the identity to well within float64 precision. Guarding with an
# explicit branch rather than the float32 epsilon used by the common reference
# implementation keeps identity exact: adding ~1.2e-7 to the denominator
# perturbs every reconstructed joint position at the 1e-7 level, which shows up
# as spurious bone-length drift in a supposedly rigid skeleton.
_SMALL_ANGLE = 1e-12


def expmap_to_rotmat(r: np.ndarray) -> np.ndarray:
    """
    Convert one exponential-map (axis-angle) vector to a 3x3 rotation matrix
    via Rodrigues' formula.

        theta = norm(r),  k = r / theta
        R = I + sin(theta) * K + (1 - cos(theta)) * K @ K

    where K is the skew-symmetric cross-product matrix of the unit axis k.
    """
    theta = float(np.linalg.norm(r))
    if theta < _SMALL_ANGLE:
        return np.eye(3)
    k = r / theta
    K = np.array([
        [  0.0, -k[2],  k[1]],
        [ k[2],   0.0, -k[0]],
        [-k[1],  k[0],   0.0],
    ], dtype=np.float64)
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def forward_kinematics(angles: np.ndarray) -> np.ndarray:
    """
    Run forward kinematics on one 99-value exponential-map frame.

    Each joint's world rotation is its local rotation composed with its
    parent's world rotation; its world position is the parent's position plus
    the fixed bone offset rotated into the parent frame:

        R_i = R_local(i) @ R_parent(i)
        p_i = offset_i @ R_parent(i) + p_parent(i)

    Only the root translates (see the note on per-joint translation channels
    above), so every bone keeps exactly the length given by the offset table.

    Parameters
    ----------
    angles : np.ndarray, shape (99,)
        One row of an exponential-map .txt file.

    Returns
    -------
    np.ndarray, shape (32, 3)
        Joint positions in millimetres.
    """
    angles = np.asarray(angles, dtype=np.float64)
    if angles.shape[-1] != 99:
        raise ValueError(f"expected 99 exponential-map values, got {angles.shape[-1]}")

    xyz = np.zeros((N_JOINTS, 3), dtype=np.float64)
    rot = [np.eye(3) for _ in range(N_JOINTS)]

    for i in range(N_JOINTS):
        local_rot = expmap_to_rotmat(angles[EXPMAP_IND[i]])
        parent = PARENT[i]

        if parent == -1:
            rot[i] = local_rot
            xyz[i] = OFFSET[i] + angles[ROOT_TRANSLATION_IND]
        else:
            xyz[i] = OFFSET[i] @ rot[parent] + xyz[parent]
            rot[i] = local_rot @ rot[parent]

    return xyz


def forward_kinematics_batch(angles: np.ndarray) -> np.ndarray:
    """
    Forward kinematics over a whole sequence.

    Parameters
    ----------
    angles : np.ndarray, shape (n_frames, 99)

    Returns
    -------
    np.ndarray, shape (n_frames, 32, 3)
    """
    angles = np.asarray(angles, dtype=np.float64)
    if angles.ndim != 2 or angles.shape[1] != 99:
        raise ValueError(f"expected (n_frames, 99), got {angles.shape}")
    return np.stack([forward_kinematics(row) for row in angles], axis=0)


def bone_length_report(xyz: np.ndarray) -> dict[str, float]:
    """
    Self-check for a forward-kinematics result.

    A correct rigid-skeleton reconstruction has bone lengths that are constant
    across frames and equal to the norms of the offset table. Drift here means
    the channel layout or the rotation composition is wrong, so this is the
    cheapest way to catch a broken FK implementation before it silently
    corrupts ground-truth angles.

    Parameters
    ----------
    xyz : np.ndarray, shape (n_frames, 32, 3)

    Returns
    -------
    dict with the worst absolute deviation (mm) from the expected bone length,
    the worst per-bone standard deviation across frames, and the reconstructed
    right upper-arm / forearm lengths for an anatomical sanity read.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    expected = np.linalg.norm(OFFSET, axis=1)

    max_dev = 0.0
    max_std = 0.0
    for i in range(N_JOINTS):
        parent = PARENT[i]
        if parent == -1:
            continue
        lengths = np.linalg.norm(xyz[:, i, :] - xyz[:, parent, :], axis=1)
        max_dev = max(max_dev, float(np.max(np.abs(lengths - expected[i]))))
        max_std = max(max_std, float(np.std(lengths)))

    return {
        "max_abs_deviation_mm": max_dev,
        "max_length_std_mm":    max_std,
        "right_upper_arm_mm":   float(np.mean(np.linalg.norm(xyz[:, 26] - xyz[:, 25], axis=1))),
        "right_forearm_mm":     float(np.mean(np.linalg.norm(xyz[:, 27] - xyz[:, 26], axis=1))),
    }
