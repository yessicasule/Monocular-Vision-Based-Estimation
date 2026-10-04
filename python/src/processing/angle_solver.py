"""
angle_solver.py — Anatomically Consistent Arm Joint Angle Estimation
=====================================================================

Computes four joint angles from MediaPipe pseudo-3D landmarks using
an ISB-inspired hybrid coordinate frame approach:

    1. Shoulder Flexion-Extension  [degrees]
    2. Shoulder Abduction-Adduction [degrees]
    3. Shoulder Internal-External Rotation [degrees]  (forearm proxy)
    4. Elbow Flexion               [degrees]

Mathematical Framework
----------------------
All shoulder angles are decomposed using a ZXY Euler sequence applied to
the upper-arm orientation vector expressed in the torso coordinate frame.
This avoids gimbal lock at the anatomical rest position (arm hanging down),
which occurs under ISB's recommended YXY sequence.

Coordinate Frame Convention (see coordinate_frame.py):
    X — lateral (pointing to user's left)
    Y — superior (pointing up along spine)
    Z — anterior (pointing out of the chest toward the camera)

Sign Conventions:
    Flexion   (+) : arm forward in the sagittal plane
    Extension (−) : arm backward
    Abduction (+) : arm raised away from the midline (out to the side)
    Adduction (−) : arm crossing toward the midline
    Int. Rot. (+) : palm rotating inward (pronation direction)
    Ext. Rot. (−) : palm rotating outward (supination direction)
    Elbow Flex (+) : forearm bending toward the upper arm; 0° = fully extended

Shoulder Angle Derivation
--------------------------
Let v = [vx, vy, vz]^T be the unit upper-arm vector in the torso frame.

    Flexion-Extension:
        θ_flex = atan2(vz, −vy)                  [rad]

        When the arm hangs down: vy ≈ −1, vz ≈ 0  →  θ_flex = atan2(0, 1)  = 0°
        When the arm points forward: vz ≈ +1        →  θ_flex = atan2(1, 0)  = +90°
        When the arm points backward: vz ≈ −1       →  θ_flex = atan2(−1, 0) = −90°

    Abduction-Adduction:
        θ_abd = arcsin(vx / ‖v‖)                 [rad]

        When the arm hangs down: vx ≈ 0            →  θ_abd ≈ 0°
        When arm is fully abducted: vx ≈ +1        →  θ_abd ≈ +90°

    Internal-External Rotation (forearm proxy):
        Computed from the component of the forearm vector that is
        perpendicular to the upper-arm axis. Only reliable when the
        elbow is flexed ≥ ROTATION_RELIABLE_THRESHOLD degrees.

        Let u = unit upper-arm vector (torso frame).
        Let f = unit forearm vector (torso frame).
        Project f onto the plane perpendicular to u:
            f_perp = f − (f·u)u
        Build two reference vectors in that plane:
            e1 = unit(cross(u, Y))  where Y = [0,1,0]^T in torso frame
            e2 = unit(cross(u, e1))
        θ_rot = atan2(f_perp · e1,  f_perp · e2)

Elbow Flexion Derivation
--------------------------
    θ_elbow = arccos(û_arm · û_forearm)

    where û_arm is the unit vector from shoulder to elbow (direction of
    upper arm), and û_forearm is the unit vector from elbow to wrist.
    Both run proximal→distal, so a straight arm makes them co-directional.
    Result: 0° = fully extended, ~150° = fully flexed.

Input Space
-----------
The landmarks passed in must be isotropic — one unit along every axis.
Use src.pose.angle_landmarks() to obtain metric world landmarks (MediaPipe)
or pixel-scaled image landmarks. Raw normalised image coordinates divide x
by the width and y by the height, which distorts every angle.

MediaPipe Right Arm Landmarks:
    RIGHT_SHOULDER = 12
    RIGHT_ELBOW    = 14
    RIGHT_WRIST    = 16

References:
    Wu et al. (2005). ISB recommendation on definitions of joint coordinate
    systems. J Biomechanics 38(5):981-992.
    Biryukova et al. (2000). Kinematics of human arm reconstructed from
    spatial tracking. J Biomechanics 33(8):985-995.
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass

from .coordinate_frame import TorsoFrame, build_torso_frame, world_to_torso, _to_array, _normalize

# ---------------------------------------------------------------------------
# MediaPipe landmark indices — both arms
# ---------------------------------------------------------------------------
LM_RIGHT_SHOULDER = 12
LM_RIGHT_ELBOW    = 14
LM_RIGHT_WRIST    = 16
LM_LEFT_SHOULDER  = 11
LM_LEFT_ELBOW     = 13
LM_LEFT_WRIST     = 15

# (shoulder, elbow, wrist) landmark indices per side
_SIDE_LANDMARKS = {
    "right": (LM_RIGHT_SHOULDER, LM_RIGHT_ELBOW, LM_RIGHT_WRIST),
    "left":  (LM_LEFT_SHOULDER,  LM_LEFT_ELBOW,  LM_LEFT_WRIST),
}

# Minimum elbow flexion (degrees) for rotation estimate to be reliable.
# Below this, the forearm is nearly collinear with the upper arm, making
# rotation unobservable from the forearm orientation.
ROTATION_RELIABLE_THRESHOLD = 25.0


@dataclass
class ArmAngles:
    """
    Complete set of estimated arm joint angles.

    Attributes
    ----------
    shoulder_flexion : float
        Shoulder flexion-extension in degrees.
        Positive = flexion (forward), negative = extension (backward).
    shoulder_abduction : float
        Shoulder abduction-adduction in degrees.
        Positive = abduction (away from body), negative = adduction.
    shoulder_rotation : float
        Shoulder internal-external rotation in degrees.
        Estimated via forearm proxy; reliable only when elbow_flexion >= 25°.
        Positive = internal rotation, negative = external rotation.
    elbow_flexion : float
        Elbow flexion in degrees. 0° = fully extended arm.
    rotation_reliable : bool
        True when elbow_flexion is sufficient for rotation estimation.
    """
    shoulder_flexion: float
    shoulder_abduction: float
    shoulder_rotation: float
    elbow_flexion: float
    rotation_reliable: bool
    side: str = "right"


@dataclass
class BilateralArmAngles:
    """
    Joint angles for both upper limbs computed from a single frame.

    Either side may be None when its landmarks are missing or degenerate
    (e.g. one arm fully occluded) while the other remains trackable.
    """
    right: ArmAngles | None
    left:  ArmAngles | None


def _compute_shoulder_angles(
    v_upper_arm_torso: np.ndarray,
) -> tuple[float, float]:
    """
    Compute shoulder flexion and abduction from the upper-arm vector
    expressed in the torso coordinate frame, using ZXY Euler decomposition.

    Parameters
    ----------
    v_upper_arm_torso : np.ndarray, shape (3,)
        Unit vector from shoulder to elbow expressed in torso frame.
        Components: [x_lateral, y_superior, z_anterior].

    Returns
    -------
    flexion_deg : float
        Shoulder flexion-extension angle in degrees.
    abduction_deg : float
        Shoulder abduction-adduction angle in degrees.
    """
    vx = v_upper_arm_torso[0]  # lateral component
    vy = v_upper_arm_torso[1]  # superior component
    vz = v_upper_arm_torso[2]  # anterior component

    # Flexion-Extension
    # θ_flex = atan2(vz, −vy)
    # Derivation: in the sagittal plane (Y-Z plane of torso frame),
    # the rest position (arm down) corresponds to vy = −1, vz = 0,
    # giving θ = atan2(0, +1) = 0°.  Arm forward: vy→0, vz→+1 (anterior),
    # giving θ = atan2(+1, 0) = +90°.
    flexion_rad = math.atan2(vz, -vy)
    flexion_deg = math.degrees(flexion_rad)

    # Abduction-Adduction
    # θ_abd = arcsin(−vx)  for the RIGHT arm.
    #
    # Derivation: the torso X-axis points to the user's LEFT
    # (constructed as left_shoulder − right_shoulder, see coordinate_frame.py).
    # Therefore, raising the RIGHT arm outward (to the right / away from midline)
    # moves the elbow in the NEGATIVE-X direction of the torso frame (vx < 0).
    # We negate to obtain the anatomical sign convention where abduction is +.
    vx_clamped   = float(np.clip(-vx, -1.0, 1.0))   # note negation for right arm
    abduction_deg = math.degrees(math.asin(vx_clamped))

    return flexion_deg, abduction_deg


def _compute_shoulder_rotation(
    v_upper_arm_torso: np.ndarray,
    v_forearm_torso: np.ndarray,
    elbow_flexion_deg: float,
) -> tuple[float, bool]:
    """
    Estimate shoulder internal-external rotation using the forearm as a proxy.

    The forearm vector, projected onto the plane perpendicular to the upper
    arm, encodes the axial rotation of the whole limb about the humeral axis.
    This is only geometrically observable when the elbow is flexed.

    Parameters
    ----------
    v_upper_arm_torso : np.ndarray, shape (3,)
        Unit upper-arm vector in torso frame.
    v_forearm_torso : np.ndarray, shape (3,)
        Unit forearm vector in torso frame (elbow → wrist direction).
    elbow_flexion_deg : float
        Current elbow flexion angle (degrees).

    Returns
    -------
    rotation_deg : float
        Estimated rotation angle in degrees; 0 when unreliable.
    reliable : bool
        Whether the estimate is geometrically reliable.
    """
    reliable = elbow_flexion_deg >= ROTATION_RELIABLE_THRESHOLD

    if not reliable:
        return 0.0, False

    u = _normalize(v_upper_arm_torso)

    # Project forearm onto the plane perpendicular to the upper arm
    # f_perp = f − (f·u)u
    f_perp = v_forearm_torso - np.dot(v_forearm_torso, u) * u
    f_perp_norm = np.linalg.norm(f_perp)

    if f_perp_norm < 1e-9:
        return 0.0, False

    f_perp = f_perp / f_perp_norm

    # Build an orthonormal basis (e1, e2) for the perpendicular plane.
    # e1 is perpendicular to u and as close as possible to the global
    # lateral axis [1, 0, 0] (or [0, 0, 1] if u is nearly lateral).
    reference = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(u, reference)) > 0.9:
        reference = np.array([0.0, 0.0, 1.0])

    e1 = np.cross(u, reference)
    e1_norm = np.linalg.norm(e1)
    if e1_norm < 1e-9:
        return 0.0, False
    e1 = e1 / e1_norm
    e2 = np.cross(u, e1)

    # atan2 gives signed rotation angle in the perpendicular plane
    rotation_deg = math.degrees(math.atan2(
        float(np.dot(f_perp, e1)),
        float(np.dot(f_perp, e2)),
    ))

    return rotation_deg, True


def _compute_elbow_flexion(
    v_upper_arm_world: np.ndarray,
    v_forearm_world: np.ndarray,
) -> float:
    """
    Compute elbow flexion angle from upper-arm and forearm direction vectors.

    θ_elbow = arccos(û_upper · û_forearm)

    where û_upper = shoulder→elbow  and  û_forearm = elbow→wrist.

    Both vectors run proximal→distal, so a fully extended arm makes them
    co-directional (dot = +1) and θ_elbow = 0°; a 90° bend makes them
    perpendicular (dot = 0); a fully folded arm makes them anti-parallel
    (dot = −1, θ_elbow = 180°).

    This matches the ground-truth convention in
    evaluation/h36m_loader._compute_side_gt, so predictions and GT are
    directly comparable — keep the two in sync.

    Parameters
    ----------
    v_upper_arm_world : np.ndarray, shape (3,)
        Vector from shoulder to elbow (direction of upper arm).
    v_forearm_world : np.ndarray, shape (3,)
        Vector from elbow to wrist (direction of forearm).

    Returns
    -------
    float
        Elbow flexion in degrees. 0° = fully extended.
    """
    u_arm    = _normalize(v_upper_arm_world)
    u_fore   = _normalize(v_forearm_world)
    cos_ang = float(np.clip(np.dot(u_arm, u_fore), -1.0, 1.0))
    return max(0.0, math.degrees(math.acos(cos_ang)))


def _compute_side_angles(landmarks, frame: TorsoFrame, side: str) -> ArmAngles | None:
    """
    Compute the four joint angles for one arm given a prebuilt torso frame.

    Left-arm handling: after transforming the segment vectors into the torso
    frame, the lateral (X) component is negated — a reflection across the
    sagittal (YZ) plane. The right-arm ZXY decomposition then yields
    anatomically mirrored conventions for the left arm:
        abduction (+) = away from the midline for BOTH arms
        flexion   (+) = forward for BOTH arms
        rotation  (+) = internal rotation for BOTH arms
    Elbow flexion uses only a dot product and is reflection-invariant.
    """
    idx_shoulder, idx_elbow, idx_wrist = _SIDE_LANDMARKS[side]

    # Extract world-space arm segment vectors
    try:
        p_shoulder = _to_array(landmarks[idx_shoulder])
        p_elbow    = _to_array(landmarks[idx_elbow])
        p_wrist    = _to_array(landmarks[idx_wrist])
    except (IndexError, TypeError):
        return None

    v_upper_arm_world = p_elbow - p_shoulder   # shoulder → elbow
    v_forearm_world   = p_wrist - p_elbow       # elbow → wrist

    if np.linalg.norm(v_upper_arm_world) < 1e-9:
        return None

    # Transform to torso frame
    v_upper_arm_torso = world_to_torso(v_upper_arm_world, frame)
    v_forearm_torso   = world_to_torso(v_forearm_world,   frame)

    # Mirror the left arm across the sagittal plane so the right-arm
    # formulas apply with mirrored anatomical sign conventions.
    if side == "left":
        v_upper_arm_torso = v_upper_arm_torso * np.array([-1.0, 1.0, 1.0])
        v_forearm_torso   = v_forearm_torso   * np.array([-1.0, 1.0, 1.0])

    # Normalise (needed for arcsin / atan2)
    v_upper_arm_unit  = _normalize(v_upper_arm_torso)

    # Shoulder flexion and abduction (ZXY decomposition)
    flexion, abduction = _compute_shoulder_angles(v_upper_arm_unit)

    # Elbow flexion (world vectors; dot product is reflection-invariant)
    elbow = _compute_elbow_flexion(v_upper_arm_world, v_forearm_world)

    # Shoulder rotation (forearm proxy)
    rotation, rot_reliable = _compute_shoulder_rotation(
        v_upper_arm_unit,
        _normalize(v_forearm_torso) if np.linalg.norm(v_forearm_torso) > 1e-9
        else v_forearm_torso,
        elbow,
    )

    return ArmAngles(
        shoulder_flexion=flexion,
        shoulder_abduction=abduction,
        shoulder_rotation=rotation,
        elbow_flexion=elbow,
        rotation_reliable=rot_reliable,
        side=side,
    )


def compute_arm_angles(landmarks, side: str = "right") -> ArmAngles | None:
    """
    Compute all four arm joint angles from a MediaPipe landmark list.

    This is the main entry point for single-arm angle computation.

    Parameters
    ----------
    landmarks : list
        MediaPipe 33-landmark list with pseudo-3D (.x, .y, .z) coordinates.
    side : str
        "right" (default) or "left".

    Returns
    -------
    ArmAngles or None
        Returns None if required landmarks are missing or the torso frame
        cannot be constructed (e.g., full-body occlusion).

    Pipeline
    --------
    1. Build torso coordinate frame from hip/shoulder landmarks.
    2. Extract arm segment vectors (shoulder→elbow, elbow→wrist).
    3. Transform arm vectors into the torso frame
       (left arm: reflect across the sagittal plane).
    4. Decompose shoulder orientation using ZXY Euler angles.
    5. Estimate shoulder rotation from forearm proxy.
    6. Compute elbow flexion from segment dot product.
    """
    if side not in _SIDE_LANDMARKS:
        raise ValueError(f"side must be one of {sorted(_SIDE_LANDMARKS)}, got {side!r}")

    frame = build_torso_frame(landmarks)
    if frame is None:
        return None

    return _compute_side_angles(landmarks, frame, side)


def compute_bilateral_angles(landmarks) -> BilateralArmAngles | None:
    """
    Compute joint angles for BOTH arms from one landmark list.

    The torso frame is built once and shared by both sides, so this is
    cheaper than two compute_arm_angles() calls.

    Returns
    -------
    BilateralArmAngles or None
        None only when the torso frame itself cannot be constructed.
        Otherwise each side is an ArmAngles or None independently.
    """
    frame = build_torso_frame(landmarks)
    if frame is None:
        return None

    return BilateralArmAngles(
        right=_compute_side_angles(landmarks, frame, "right"),
        left=_compute_side_angles(landmarks, frame, "left"),
    )
