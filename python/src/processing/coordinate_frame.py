"""
coordinate_frame.py — Torso-relative Anatomical Coordinate Frame Construction
==============================================================================

Constructs an orthonormal coordinate frame attached to the torso from MediaPipe
pseudo-3D body landmarks. All downstream shoulder angle computations are performed
relative to this frame, making them independent of camera orientation and user
position/lean.

Coordinate Axes (right-hand system, ISB-inspired):
    Y-axis : vertical / superior direction (hip → shoulder)
    X-axis : lateral direction (pointing to user's left, right shoulder → left shoulder)
    Z-axis : anterior / forward direction (out of the chest, toward the camera)

Body-relative computation ensures that:
    - Shoulder flexion is detected regardless of camera angle
    - Torso lean or rotation is automatically compensated
    - Both frontal and sagittal camera views produce consistent results

MediaPipe landmark indices used:
    11 — Left shoulder
    12 — Right shoulder
    23 — Left hip
    24 — Right hip

References:
    ISB recommendation on definitions of joint coordinate systems:
    Wu et al. (2005), J Biomech 38(5):981-992.
    Gram-Schmidt orthogonalization: Golub & Van Loan, Matrix Computations, 4th ed.
"""

import numpy as np
from dataclasses import dataclass

# MediaPipe landmark indices for upper body landmarks
LM_LEFT_SHOULDER  = 11
LM_RIGHT_SHOULDER = 12
LM_LEFT_HIP       = 23
LM_RIGHT_HIP      = 24

# Minimum sine of the angle between the shoulder axis and the spine for the
# torso frame to be well-conditioned. See build_torso_frame() for why a frame
# below this threshold must be rejected rather than normalised.
MIN_TORSO_CONDITIONING = 0.1


@dataclass
class TorsoFrame:
    """
    Orthonormal reference frame attached to the torso.

    Attributes
    ----------
    origin : np.ndarray, shape (3,)
        World-space origin of the frame (mid-shoulder point).
    R : np.ndarray, shape (3, 3)
        Rotation matrix whose columns are [x_axis, y_axis, z_axis].
        Transforms a world-space vector v into torso-space via:
            v_torso = R.T @ v_world
    x_axis : np.ndarray, shape (3,)
        Lateral axis (pointing to user's left).
    y_axis : np.ndarray, shape (3,)
        Superior axis (pointing upward along spine).
    z_axis : np.ndarray, shape (3,)
        Anterior axis (pointing forward out of the chest).
    """
    origin: np.ndarray
    R: np.ndarray
    x_axis: np.ndarray
    y_axis: np.ndarray
    z_axis: np.ndarray


def _to_array(landmark) -> np.ndarray:
    """Convert a landmark (attribute or index access) to a numpy array."""
    try:
        return np.array([landmark.x, landmark.y, landmark.z], dtype=np.float64)
    except AttributeError:
        return np.array([landmark[0], landmark[1], landmark[2]], dtype=np.float64)


def _normalize(v: np.ndarray) -> np.ndarray:
    """Return a unit vector. Returns v unchanged if near-zero norm."""
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def build_torso_frame(landmarks) -> TorsoFrame | None:
    """
    Construct the torso coordinate frame from four body landmarks.

    Algorithm
    ---------
    1. Extract 3D positions of left/right shoulders and left/right hips.
    2. Compute torso centre (hip midpoint) and shoulder centre.
    3. Primary Y-axis candidate: hip_mid → shoulder_mid (superior direction).
    4. Primary X-axis candidate: right_shoulder → left_shoulder (lateral).
    5. Z-axis: cross(X_candidate, Y) — anterior/forward.
    6. Re-derive X: cross(Y, Z) to enforce right-hand orthogonality.
    7. Gram-Schmidt re-orthogonalise to remove residual non-orthogonality
       from noisy keypoints.

    Parameters
    ----------
    landmarks : list
        MediaPipe pose landmark list (33 landmarks), each with .x, .y, .z
        in normalised image coordinates.  The z component is MediaPipe's
        pseudo-depth estimate (relative to the hip).

    Returns
    -------
    TorsoFrame or None
        Returns None if the required landmarks are missing or degenerate.

    Notes
    -----
    MediaPipe's z coordinate is in the same scale as x (normalised image
    width), with the hip as z = 0.  It is a relative depth estimate from a
    monocular image and therefore noisy, especially for lateral camera views.
    """
    try:
        l_shoulder = _to_array(landmarks[LM_LEFT_SHOULDER])
        r_shoulder = _to_array(landmarks[LM_RIGHT_SHOULDER])
        l_hip      = _to_array(landmarks[LM_LEFT_HIP])
        r_hip      = _to_array(landmarks[LM_RIGHT_HIP])
    except (IndexError, TypeError):
        return None

    # --- Step 1: Midpoints ---
    hip_mid      = 0.5 * (l_hip + r_hip)
    shoulder_mid = 0.5 * (l_shoulder + r_shoulder)

    # --- Step 2: Primary axis candidates ---
    # Y: superior direction (hip → shoulder centre, along the spine)
    y_cand = shoulder_mid - hip_mid

    # X: lateral direction (right shoulder → left shoulder)
    x_cand = l_shoulder - r_shoulder

    if np.linalg.norm(y_cand) < 1e-9 or np.linalg.norm(x_cand) < 1e-9:
        return None  # Degenerate configuration

    y_cand = _normalize(y_cand)
    x_cand = _normalize(x_cand)

    # --- Step 3: Gram-Schmidt orthogonalisation ---
    # Make X orthogonal to Y, then derive Z from their cross product.
    #
    # Gram-Schmidt ensures a proper orthonormal basis despite keypoint noise:
    #   x' = x - (x·y)y      (remove y-component from x)
    #   z  = x' × y          (right-hand forward axis)
    x_orth = x_cand - np.dot(x_cand, y_cand) * y_cand

    # Reject a torso whose lateral axis is (near-)collinear with the spine.
    # ||x_orth|| here is sin(angle between the shoulder axis and the spine), so
    # it collapses toward zero exactly when the two candidates stop spanning a
    # plane — a subject seen edge-on, or a frame where both shoulders project
    # to nearly the same point. Normalising through that collapse divides by
    # noise and yields a frame that looks valid but is arbitrary, so every
    # angle derived from it would be meaningless. An anatomically intact torso
    # sits near 1.0; 0.1 (5.7 degrees) rejects only genuine degeneracy.
    if np.linalg.norm(x_orth) < MIN_TORSO_CONDITIONING:
        return None

    x_orth = _normalize(x_orth)

    z_axis = np.cross(x_orth, y_cand)
    z_axis = _normalize(z_axis)

    # Rebuild X one final time from Y×Z for numerical cleanliness
    x_axis = np.cross(y_cand, z_axis)
    x_axis = _normalize(x_axis)

    # Column-wise rotation matrix: each column is one axis
    R = np.column_stack([x_axis, y_cand, z_axis])  # shape (3, 3)

    return TorsoFrame(
        origin=shoulder_mid,
        R=R,
        x_axis=x_axis,
        y_axis=y_cand,
        z_axis=z_axis,
    )


def world_to_torso(v_world: np.ndarray, frame: TorsoFrame) -> np.ndarray:
    """
    Express a world-space vector in the torso coordinate frame.

    Parameters
    ----------
    v_world : np.ndarray, shape (3,)
        Vector in world (camera-normalised) coordinates.
    frame : TorsoFrame
        Torso frame from build_torso_frame().

    Returns
    -------
    np.ndarray, shape (3,)
        Vector expressed in the torso frame:
            [lateral (X), superior (Y), anterior (Z)]
    """
    return frame.R.T @ v_world
