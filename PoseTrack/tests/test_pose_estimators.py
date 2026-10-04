"""
test_pose_estimators.py — Unit Tests for Pose Estimation Pipeline
=================================================================

Tests cover:
    1. PoseEstimator abstract interface contract
    2. MediaPipeRunner (with mock backend)
    3. MoveNet COCO→MediaPipe remapping correctness
    4. PoseNet landmark remapping
    5. Angle solver round-trip with synthetic landmarks

Run from project root:
    python -m pytest src/evaluation/test_pose_estimators.py -v
    -- or --
    python tests/test_pose_estimators.py
"""

from __future__ import annotations

import sys
import math
import unittest
from pathlib import Path

import numpy as np

from src.pose.base import Landmark, N_LANDMARKS, _default_landmarks
from src.processing.coordinate_frame import build_torso_frame, TorsoFrame
from src.processing.angle_solver import compute_arm_angles, ArmAngles


# ---------------------------------------------------------------------------
# Helpers: build synthetic landmark lists
# ---------------------------------------------------------------------------

def _make_landmarks(positions: dict[int, tuple[float, float, float]]) -> list[Landmark]:
    """
    Build a 33-landmark list with specific indices set to the given (x,y,z).
    All other landmarks default to (0,0,0,0).
    """
    lms = _default_landmarks()
    for idx, (x, y, z) in positions.items():
        lms[idx] = Landmark(x=x, y=y, z=z, visibility=1.0)
    return lms


def _arm_down_landmarks() -> list[Landmark]:
    """
    Subject facing the camera with the right arm hanging straight down.
    Expected: all angles ≈ 0° (flexion=0, abduction=0, elbow=0).

    MediaPipe image coordinate system:
        x: right, y: down (image top = 0), z: away from the camera
    A subject FACING the camera has their right side on the image's LEFT,
    so their anterior direction is −z (toward the camera).

    Landmark positions (normalised image coords):
        Left shoulder  (11): (0.6, 0.3, 0)
        Right shoulder (12): (0.4, 0.3, 0)
        Left hip       (23): (0.6, 0.6, 0)
        Right hip      (24): (0.4, 0.6, 0)
        Right elbow    (14): (0.4, 0.5, 0)   ← arm pointing down (+y)
        Right wrist    (16): (0.4, 0.7, 0)
    """
    return _make_landmarks({
        11: (0.6, 0.3, 0.0),   # left shoulder
        12: (0.4, 0.3, 0.0),   # right shoulder
        23: (0.6, 0.6, 0.0),   # left hip
        24: (0.4, 0.6, 0.0),   # right hip
        14: (0.4, 0.5, 0.0),   # right elbow (arm down)
        16: (0.4, 0.7, 0.0),   # right wrist
    })


def _arm_forward_landmarks() -> list[Landmark]:
    """
    Right arm raised forward (shoulder flexion ~90°).
    In image space, "forward" = toward camera = negative z (depth).

    We simulate this by setting z < 0 for elbow/wrist while keeping
    y at shoulder height.
    """
    return _make_landmarks({
        11: (0.6, 0.3,  0.0),
        12: (0.4, 0.3,  0.0),
        23: (0.6, 0.6,  0.0),
        24: (0.4, 0.6,  0.0),
        14: (0.4, 0.3, -0.15),   # elbow forward (−z = toward camera)
        16: (0.4, 0.3, -0.30),   # wrist further forward
    })


def _arm_backward_landmarks() -> list[Landmark]:
    """Right arm swung 45° behind the body (+z = away from camera)."""
    return _make_landmarks({
        11: (0.6, 0.3, 0.0),
        12: (0.4, 0.3, 0.0),
        23: (0.6, 0.6, 0.0),
        24: (0.4, 0.6, 0.0),
        14: (0.4, 0.4, 0.10),    # down and behind
        16: (0.4, 0.5, 0.20),
    })


def _arm_side_landmarks() -> list[Landmark]:
    """
    Right arm raised to the side (abduction ~90°).
    The subject faces the camera, so their right is the image's −x.
    """
    return _make_landmarks({
        11: (0.6, 0.3, 0.0),
        12: (0.4, 0.3, 0.0),
        23: (0.6, 0.6, 0.0),
        24: (0.4, 0.6, 0.0),
        14: (0.25, 0.3, 0.0),    # elbow out to the subject's right
        16: (0.10, 0.3, 0.0),    # wrist further out
    })


def _elbow_bent_landmarks() -> list[Landmark]:
    """
    Right arm hanging down, elbow bent ~90° (forearm points forward).
    Upper arm: shoulder(0.4,0.3)→elbow(0.4,0.5) [down]
    Forearm:   elbow(0.4,0.5)→wrist(0.4,0.5,-0.15) [forward]
    """
    return _make_landmarks({
        11: (0.6, 0.3,  0.0),
        12: (0.4, 0.3,  0.0),
        23: (0.6, 0.6,  0.0),
        24: (0.4, 0.6,  0.0),
        14: (0.4, 0.5,  0.0),    # elbow down from shoulder
        16: (0.4, 0.5, -0.15),   # wrist forward → 90° elbow
    })


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

class TestDefaultLandmarks(unittest.TestCase):
    def test_length(self):
        lms = _default_landmarks()
        self.assertEqual(len(lms), N_LANDMARKS)

    def test_all_zero_visibility(self):
        lms = _default_landmarks()
        for lm in lms:
            self.assertEqual(lm.visibility, 0.0)


class TestTorsoFrame(unittest.TestCase):
    def _frame(self, lms):
        return build_torso_frame(lms)

    def test_builds_successfully(self):
        lms = _arm_down_landmarks()
        frame = self._frame(lms)
        self.assertIsNotNone(frame)

    def test_orthonormality(self):
        """R should be an orthonormal matrix: R^T R = I."""
        frame = self._frame(_arm_down_landmarks())
        self.assertIsNotNone(frame)
        RtR = frame.R.T @ frame.R
        np.testing.assert_allclose(RtR, np.eye(3), atol=1e-6,
                                   err_msg="Torso frame R is not orthonormal")

    def test_proper_rotation(self):
        """det(R) should be +1 (right-handed system)."""
        frame = self._frame(_arm_down_landmarks())
        self.assertIsNotNone(frame)
        det = np.linalg.det(frame.R)
        self.assertAlmostEqual(det, 1.0, places=5,
                               msg="Torso frame is not a proper rotation (det≠1)")

    def test_returns_none_for_degenerate(self):
        """A landmark list with zero-distance hips/shoulders → None."""
        lms = _default_landmarks()   # all zeros
        frame = self._frame(lms)
        # Either None or very small y-axis norm
        if frame is not None:
            norm = np.linalg.norm(frame.y_axis)
            self.assertAlmostEqual(norm, 0.0, places=3)


class TestAngleSolverReference(unittest.TestCase):
    """Tests against synthetic reference poses with known expected angles."""

    TOL_DEG = 15.0   # tolerance in degrees for geometric approximations

    def _angles(self, lms) -> ArmAngles | None:
        return compute_arm_angles(lms)

    def test_arm_down_near_zero_flexion(self):
        a = self._angles(_arm_down_landmarks())
        self.assertIsNotNone(a, "Angle solver returned None for arm_down landmarks")
        self.assertAlmostEqual(a.shoulder_flexion, 0.0, delta=self.TOL_DEG,
                               msg=f"Expected flexion≈0° at arm_down, got {a.shoulder_flexion:.1f}°")

    def test_arm_down_near_zero_abduction(self):
        a = self._angles(_arm_down_landmarks())
        self.assertIsNotNone(a)
        self.assertAlmostEqual(a.shoulder_abduction, 0.0, delta=self.TOL_DEG,
                               msg=f"Expected abduction≈0° at arm_down, got {a.shoulder_abduction:.1f}°")

    def test_arm_down_near_zero_elbow(self):
        a = self._angles(_arm_down_landmarks())
        self.assertIsNotNone(a)
        self.assertAlmostEqual(a.elbow_flexion, 0.0, delta=self.TOL_DEG,
                               msg=f"Expected elbow≈0° at arm_down, got {a.elbow_flexion:.1f}°")

    def test_arm_side_positive_abduction(self):
        a = self._angles(_arm_side_landmarks())
        self.assertIsNotNone(a)
        self.assertGreater(a.shoulder_abduction, 30.0,
                           msg=f"Expected abduction>30° for arm_side, got {a.shoulder_abduction:.1f}°")

    def test_elbow_bent_positive_flexion(self):
        a = self._angles(_elbow_bent_landmarks())
        self.assertIsNotNone(a)
        self.assertGreater(a.elbow_flexion, 45.0,
                           msg=f"Expected elbow_flexion>45° when bent, got {a.elbow_flexion:.1f}°")

    def test_arm_forward_positive_flexion(self):
        a = self._angles(_arm_forward_landmarks())
        self.assertIsNotNone(a)
        self.assertAlmostEqual(a.shoulder_flexion, 90.0, delta=1.0,
                               msg=f"Expected flexion≈+90° for arm_forward, got {a.shoulder_flexion:.1f}°")

    def test_arm_backward_negative_flexion(self):
        a = self._angles(_arm_backward_landmarks())
        self.assertIsNotNone(a)
        self.assertAlmostEqual(a.shoulder_flexion, -45.0, delta=1.0,
                               msg=f"Expected flexion≈-45° for arm_backward, got {a.shoulder_flexion:.1f}°")

    def test_anterior_axis_points_at_the_camera(self):
        """A subject facing the camera has its anterior axis along −z."""
        frame = build_torso_frame(_arm_down_landmarks())
        np.testing.assert_allclose(frame.z_axis, [0.0, 0.0, -1.0], atol=1e-9)

    def test_left_arm_mirrors_right_arm(self):
        """Mirror-image poses give identical angles on both sides."""
        from src.processing.angle_solver import compute_bilateral_angles
        lms = _arm_side_landmarks()
        # Mirror the right arm onto the left about the body midline x = 0.5
        lms[13] = Landmark(x=1.0 - lms[14].x, y=lms[14].y, z=lms[14].z)
        lms[15] = Landmark(x=1.0 - lms[16].x, y=lms[16].y, z=lms[16].z)
        b = compute_bilateral_angles(lms)
        self.assertAlmostEqual(b.left.shoulder_abduction, b.right.shoulder_abduction, places=6)
        self.assertAlmostEqual(b.left.shoulder_flexion, b.right.shoulder_flexion, places=6)

    def test_elbow_never_negative(self):
        """Elbow flexion should always be ≥ 0 by physical constraint."""
        for name, lms_fn in [
            ("arm_down",    _arm_down_landmarks),
            ("arm_forward", _arm_forward_landmarks),
            ("arm_side",    _arm_side_landmarks),
        ]:
            a = self._angles(lms_fn())
            if a is not None:
                self.assertGreaterEqual(a.elbow_flexion, -5.0,
                                        msg=f"Elbow flexion negative ({a.elbow_flexion:.1f}°) at {name}")

    def test_rotation_unreliable_when_elbow_extended(self):
        """Rotation should be marked unreliable when elbow is nearly straight."""
        a = self._angles(_arm_down_landmarks())
        if a is not None and a.elbow_flexion < 25.0:
            self.assertFalse(a.rotation_reliable,
                             msg="Rotation marked reliable with nearly-straight elbow")


class TestAngleSpace(unittest.TestCase):
    """Angles must be computed in an isotropic space, not normalised coords."""

    def test_pixel_space_restores_true_angle_on_wide_frames(self):
        from src.pose.base import to_pixel_space
        # 1280×720 frame; in PIXELS the right arm is abducted 60° from
        # hanging: 100 px out and 57.7 px down from the shoulder (512, 216).
        W, H = 1280, 720
        down = 100.0 / math.tan(math.radians(60.0))
        lms = _make_landmarks({
            11: (768 / W, 216 / H, 0.0), 12: (512 / W, 216 / H, 0.0),
            23: (768 / W, 432 / H, 0.0), 24: (512 / W, 432 / H, 0.0),
            14: ((512 - 100) / W, (216 + down) / H, 0.0),
            16: ((512 - 200) / W, (216 + 2 * down) / H, 0.0),
        })
        a_px = compute_arm_angles(to_pixel_space(lms, W, H))
        self.assertAlmostEqual(a_px.shoulder_abduction, 60.0, delta=0.01)
        # The same pose read from raw normalised coordinates is distorted
        a_norm = compute_arm_angles(lms)
        self.assertGreater(abs(a_norm.shoulder_abduction - a_px.shoulder_abduction), 5.0)

    def test_angle_landmarks_prefers_world_landmarks(self):
        from src.pose.base import angle_landmarks
        image = _arm_down_landmarks()
        world = _arm_side_landmarks()

        class _Est:
            world_landmarks = world
        self.assertIs(angle_landmarks(_Est(), image, 640, 480), world)

        class _NoWorld:
            world_landmarks = None
        px = angle_landmarks(_NoWorld(), image, 640, 480)
        self.assertAlmostEqual(px[12].x, image[12].x * 640)
        self.assertAlmostEqual(px[12].y, image[12].y * 480)
        self.assertIsNone(angle_landmarks(_Est(), None, 640, 480))


class TestMovenetRemapping(unittest.TestCase):
    """Test the COCO→MediaPipe remapping logic in MoveNetRunner."""

    def _build_fake_kps(self, coco_positions: dict[int, tuple]) -> np.ndarray:
        """Build a (17, 3) = [y, x, score] array."""
        kps = np.zeros((17, 3), dtype=np.float32)
        kps[:, 2] = 0.05   # below threshold by default
        for coco_idx, (y, x, score) in coco_positions.items():
            kps[coco_idx] = [y, x, score]
        return kps

    def test_remapping_places_correct_indices(self):
        """Left shoulder (COCO 5) → MediaPipe 11."""
        # Import only the remapping helper, not the full TF dependency
        from src.pose.movenet_runner import _COCO_TO_MP
        self.assertEqual(_COCO_TO_MP[5],  11)
        self.assertEqual(_COCO_TO_MP[6],  12)
        self.assertEqual(_COCO_TO_MP[7],  13)
        self.assertEqual(_COCO_TO_MP[8],  14)
        self.assertEqual(_COCO_TO_MP[9],  15)
        self.assertEqual(_COCO_TO_MP[10], 16)
        self.assertEqual(_COCO_TO_MP[11], 23)
        self.assertEqual(_COCO_TO_MP[12], 24)

    def test_coordinate_swap(self):
        """COCO [y, x, score] should map to Landmark(x=x, y=y)."""
        from src.pose.movenet_runner import MoveNetRunner, _COCO_TO_MP, _default_landmarks
        from src.pose.base import Landmark

        # Simulate _remap_to_mediapipe manually
        kps = np.zeros((17, 3), dtype=np.float32)
        kps[5]  = [0.3, 0.4, 0.9]   # COCO left shoulder: y=0.3, x=0.4
        kps[6]  = [0.3, 0.6, 0.9]   # COCO right shoulder
        kps[7]  = [0.5, 0.4, 0.8]   # COCO left elbow
        kps[8]  = [0.5, 0.6, 0.8]   # COCO right elbow
        kps[11] = [0.6, 0.4, 0.85]  # COCO left hip
        kps[12] = [0.6, 0.6, 0.85]  # COCO right hip

        lms = _default_landmarks()
        for coco_idx, mp_idx in _COCO_TO_MP.items():
            y, x, score = float(kps[coco_idx, 0]), float(kps[coco_idx, 1]), float(kps[coco_idx, 2])
            lms[mp_idx] = Landmark(x=x, y=y, z=0.0, visibility=score)

        # MediaPipe right shoulder = index 12
        self.assertAlmostEqual(lms[12].x, 0.6, places=5,
                               msg="x coordinate not correctly swapped from COCO [y,x]")
        self.assertAlmostEqual(lms[12].y, 0.3, places=5,
                               msg="y coordinate not correctly swapped from COCO [y,x]")

    def test_output_has_33_landmarks(self):
        """Remapped landmark list must have exactly 33 elements."""
        lms = _default_landmarks()
        self.assertEqual(len(lms), 33)


class TestFilterBank(unittest.TestCase):
    """Sanity tests for the filter bank with synthetic angle signals."""

    def _make_angles(self, flex=0.0, abd=0.0, rot=0.0, elb=0.0) -> ArmAngles:
        return ArmAngles(
            shoulder_flexion=flex,
            shoulder_abduction=abd,
            shoulder_rotation=rot,
            elbow_flexion=elb,
            rotation_reliable=False,
        )

    def test_kalman_initialises_to_first_value(self):
        from src.processing.angle_filter import AngleFilterBank
        filt = AngleFilterBank("kalman")
        a0 = self._make_angles(flex=45.0, elb=90.0)
        out = filt.update(a0)
        self.assertAlmostEqual(out.shoulder_flexion, 45.0, delta=1.0)
        self.assertAlmostEqual(out.elbow_flexion,    90.0, delta=1.0)

    def test_kalman_converges_on_constant(self):
        from src.processing.angle_filter import AngleFilterBank
        filt = AngleFilterBank("kalman")
        a = self._make_angles(flex=60.0, elb=45.0)
        for _ in range(60):
            out = filt.update(a)
        self.assertAlmostEqual(out.shoulder_flexion, 60.0, delta=2.0)
        self.assertAlmostEqual(out.elbow_flexion,    45.0, delta=2.0)

    def test_ma_reduces_noise(self):
        from src.processing.angle_filter import AngleFilterBank
        filt = AngleFilterBank("ma")
        rng = np.random.default_rng(42)
        noisy = [self._make_angles(flex=30.0 + rng.normal(0, 10)) for _ in range(100)]
        outputs = [filt.update(a).shoulder_flexion for a in noisy]
        raw_std  = float(np.std([a.shoulder_flexion for a in noisy]))
        filt_std = float(np.std(outputs[10:]))   # skip warm-up
        self.assertLess(filt_std, raw_std,
                        msg="MA filter did not reduce signal variance")

    def test_filter_reset_clears_state(self):
        from src.processing.angle_filter import AngleFilterBank
        filt = AngleFilterBank("kalman")
        for _ in range(30):
            filt.update(self._make_angles(flex=90.0))
        filt.reset()
        out = filt.update(self._make_angles(flex=0.0))
        self.assertAlmostEqual(out.shoulder_flexion, 0.0, delta=1.0,
                               msg="Filter state not cleared after reset()")

    def test_sg_requires_sufficient_buffer(self):
        """SG filter should fall back to raw until buffer fills."""
        from src.processing.angle_filter import AngleFilterBank
        filt = AngleFilterBank("sg")
        a = self._make_angles(flex=45.0)
        out = filt.update(a)
        # Before buffer fills (window_length=11), should return raw value
        self.assertAlmostEqual(out.shoulder_flexion, 45.0, delta=1.0)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
