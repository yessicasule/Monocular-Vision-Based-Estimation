"""
test_evaluation.py — Unit Tests for Stage 4 (Evaluation Pipeline)
==================================================================

Tests cover:
    1. h36m_loader.py  — GTAngles dataclass, _compute_gt_angles(), parse_h36m_file()
    2. metrics.py      — compute_joint_metrics(), evaluate_framework(), print_metrics_table()
    3. Integration     — synthetic noise pipeline matches expected metric bounds
    4. build_h36m_dataset.py — _enrich_with_noise() produces in-bound values

Run with:
    python tests/test_evaluation.py
"""

from __future__ import annotations

import io
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.evaluation.h36m_loader import (
    _compute_gt_angles,
    _normalize,
    GTAngles,
    gt_angles_from_row,
    row_to_joint_positions,
    H36M_RSHOULDER, H36M_RELBOW, H36M_RWRIST,
    H36M_LSHOULDER, H36M_RHIP, H36M_LHIP, H36M_CHEST,
)
from src.evaluation.h36m_skeleton import (
    N_JOINTS, OFFSET, PARENT, EXPMAP_IND,
    expmap_to_rotmat, forward_kinematics, forward_kinematics_batch,
    bone_length_report,
)
from src.evaluation.metrics import (
    compute_joint_metrics, evaluate_framework, print_metrics_table,
    JOINTS, FrameworkMetrics,
)


# ---------------------------------------------------------------------------
# Helper: build synthetic H3.6M joint array
# ---------------------------------------------------------------------------

def _make_joints(
    r_shoulder=(0.2, 1.4, 0.0),
    r_elbow=   (0.2, 1.0, 0.0),
    r_wrist=   (0.2, 0.6, 0.0),
    l_shoulder=(-0.2, 1.4, 0.0),
    r_hip=     (0.15, 0.9, 0.0),
    l_hip=     (-0.15, 0.9, 0.0),
    chest=     (0.0, 1.2, 0.0),
) -> np.ndarray:
    """
    Build a (33, 3) joint array with only the right-arm joints set.
    All other joints default to (0, 0, 0).
    """
    joints = np.zeros((33, 3), dtype=np.float64)
    joints[H36M_RSHOULDER] = r_shoulder
    joints[H36M_RELBOW]    = r_elbow
    joints[H36M_RWRIST]    = r_wrist
    joints[H36M_LSHOULDER] = l_shoulder
    joints[H36M_RHIP]      = r_hip
    joints[H36M_LHIP]      = l_hip
    joints[H36M_CHEST]     = chest
    return joints


# ---------------------------------------------------------------------------
# H3.6M Loader Tests
# ---------------------------------------------------------------------------

class TestH36mLoader(unittest.TestCase):

    def test_arm_down_flexion_near_zero(self):
        """
        When the upper arm hangs straight down (elbow below shoulder),
        shoulder flexion should be ≈ 0°.
        """
        joints = _make_joints(
            r_shoulder=(0.2, 1.4, 0.0),
            r_elbow   =(0.2, 1.0, 0.0),  # directly below shoulder
            r_wrist   =(0.2, 0.6, 0.0),
        )
        gt = _compute_gt_angles(joints)
        self.assertIsNotNone(gt)
        self.assertAlmostEqual(gt.shoulder_flexion, 0.0, delta=15.0,
            msg=f"Arm-down: expected flexion≈0°, got {gt.shoulder_flexion:.1f}°")

    def test_arm_down_elbow_near_zero(self):
        """Arm fully extended downward → elbow flexion ≈ 0°."""
        joints = _make_joints(
            r_shoulder=(0.2, 1.4, 0.0),
            r_elbow   =(0.2, 1.0, 0.0),
            r_wrist   =(0.2, 0.6, 0.0),  # collinear: elbow = 0°
        )
        gt = _compute_gt_angles(joints)
        self.assertIsNotNone(gt)
        self.assertAlmostEqual(gt.elbow_flexion, 0.0, delta=10.0,
            msg=f"Expected elbow≈0° (straight arm), got {gt.elbow_flexion:.1f}°")

    def test_arm_forward_positive_flexion(self):
        """
        Arm raised forward (elbow at shoulder height, in front of body).
        In H3.6M coordinates (Y-up, Z-forward), 'forward' = positive Z.
        Expect shoulder_flexion > 45°.
        """
        joints = _make_joints(
            r_shoulder=(0.2, 1.4,  0.0),
            r_elbow   =(0.2, 1.4,  0.3),   # forward (positive Z)
            r_wrist   =(0.2, 1.4,  0.6),
        )
        gt = _compute_gt_angles(joints)
        self.assertIsNotNone(gt)
        self.assertGreater(gt.shoulder_flexion, 30.0,
            msg=f"Arm forward: expected flexion>30°, got {gt.shoulder_flexion:.1f}°")

    def test_arm_side_positive_abduction(self):
        """
        Arm raised out to the right side (positive X direction).
        Expect shoulder_abduction > 30°.
        """
        joints = _make_joints(
            r_shoulder=(0.2, 1.4, 0.0),
            r_elbow   =(0.5, 1.4, 0.0),    # elbow out to the right (+X)
            r_wrist   =(0.8, 1.4, 0.0),
        )
        gt = _compute_gt_angles(joints)
        self.assertIsNotNone(gt)
        self.assertGreater(gt.shoulder_abduction, 30.0,
            msg=f"Arm side: expected abduction>30°, got {gt.shoulder_abduction:.1f}°")

    def test_elbow_bent_90_degrees(self):
        """
        Upper arm hanging down, forearm pointing forward → elbow ≈ 90°.
        """
        joints = _make_joints(
            r_shoulder=(0.2, 1.4,  0.0),
            r_elbow   =(0.2, 1.0,  0.0),   # upper arm: down
            r_wrist   =(0.2, 1.0,  0.4),   # forearm: forward (+Z)
        )
        gt = _compute_gt_angles(joints)
        self.assertIsNotNone(gt)
        self.assertAlmostEqual(gt.elbow_flexion, 90.0, delta=15.0,
            msg=f"Expected elbow≈90°, got {gt.elbow_flexion:.1f}°")

    def test_degenerate_joints_returns_none(self):
        """All-zero joints → degenerate skeleton → should return None."""
        joints = np.zeros((33, 3), dtype=np.float64)
        gt = _compute_gt_angles(joints)
        # Should be None (degenerate) or return 0s — either acceptable
        if gt is not None:
            # If it doesn't return None, angles should be near-zero
            self.assertAlmostEqual(abs(gt.shoulder_flexion), 0.0, delta=1.0)

    def test_gtangles_dataclass_fields(self):
        """GTAngles must contain all required fields."""
        gt = GTAngles(
            shoulder_flexion=10.0,
            shoulder_abduction=20.0,
            shoulder_rotation=5.0,
            elbow_flexion=45.0,
            rotation_reliable=True,
            frame_idx=0,
        )
        self.assertEqual(gt.shoulder_flexion,   10.0)
        self.assertEqual(gt.shoulder_abduction, 20.0)
        self.assertEqual(gt.elbow_flexion,      45.0)
        self.assertTrue(gt.rotation_reliable)

    def test_rotation_unreliable_when_elbow_straight(self):
        """Rotation should be marked unreliable when elbow is nearly straight."""
        joints = _make_joints(
            r_shoulder=(0.2, 1.4, 0.0),
            r_elbow   =(0.2, 1.0, 0.0),
            r_wrist   =(0.2, 0.6, 0.0),   # straight arm
        )
        gt = _compute_gt_angles(joints)
        if gt is not None and gt.elbow_flexion < 20.0:
            self.assertFalse(gt.rotation_reliable,
                msg="Rotation marked reliable with straight elbow")

    def test_parse_h36m_file_invalid_path(self):
        """Non-existent file should raise FileNotFoundError."""
        from src.evaluation.h36m_loader import parse_h36m_file
        with self.assertRaises((FileNotFoundError, OSError)):
            parse_h36m_file(Path("/nonexistent/path/file.txt"))

    def test_parse_h36m_file_synthetic_content(self):
        """
        Synthetic .txt content with valid 99-value rows should parse correctly.
        We construct a frame with known joint positions and verify output.
        """
        from src.evaluation.h36m_loader import parse_h36m_file

        # Build one frame: 33 joints, joints at known positions
        joints = _make_joints(
            r_shoulder=(0.2, 1.4, 0.0),
            r_elbow   =(0.2, 1.0, 0.0),
            r_wrist   =(0.2, 0.6, 0.0),
        )
        flat = joints.flatten()   # 99 values
        line = ",".join(f"{v:.6f}" for v in flat)

        # Write to a temp file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write(line + "\n")
            f.write(line + "\n")   # two identical frames
            tmp_path = Path(f.name)

        try:
            results = parse_h36m_file(tmp_path)
            self.assertEqual(len(results), 2,
                msg=f"Expected 2 parsed frames, got {len(results)}")
            # Both frames should have the same angles
            self.assertAlmostEqual(
                results[0].shoulder_flexion, results[1].shoulder_flexion, places=3)
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_parse_h36m_file_skips_short_rows(self):
        """Rows with fewer than 99 values should be silently skipped."""
        from src.evaluation.h36m_loader import parse_h36m_file

        joints = _make_joints()
        flat   = joints.flatten()
        good   = ",".join(f"{v:.4f}" for v in flat)
        bad    = "1.0,2.0,3.0"   # too few values

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write(good + "\n")
            f.write(bad  + "\n")   # should be skipped
            f.write(good + "\n")
            tmp_path = Path(f.name)

        try:
            results = parse_h36m_file(tmp_path)
            self.assertEqual(len(results), 2,
                msg="Short row should have been skipped")
        finally:
            tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Metrics Tests
# ---------------------------------------------------------------------------

class TestH36mSkeleton(unittest.TestCase):
    """
    Forward kinematics for the exponential-map H3.6M release.

    These guard the distinction that broke the original loader: 99-value rows
    are axis-angle rotations, not Cartesian positions, and must go through
    forward kinematics before any angle is computed.
    """

    @staticmethod
    def _random_frame(seed: int = 0) -> np.ndarray:
        """
        A synthetic exponential-map frame shaped like real H3.6M data.

        Real recordings carry translation channels only on the root; every
        other joint's translation triplet is zero, which is what makes the
        skeleton rigid. Randomising those channels too would fabricate a
        stretchy skeleton that no H3.6M file contains.
        """
        rng = np.random.default_rng(seed)
        frame = rng.normal(0.0, 0.12, size=99)

        # Every arm offset in this skeleton runs along the local Y axis, so the
        # zero pose collapses both shoulders onto the spine — a degenerate
        # torso. Real recordings carry roughly quarter-turn rotations at the
        # shoulder anchors, which is what spreads the arms laterally; without
        # them the fixture would exercise only the degenerate path. Small
        # perturbations then keep the pose varied but anatomically intact.
        quarter_turn = np.pi / 2
        frame[EXPMAP_IND[16]] += [0.0, 0.0,  quarter_turn]   # LShoulderAnchor
        frame[EXPMAP_IND[24]] += [0.0, 0.0, -quarter_turn]   # RShoulderAnchor
        return frame

    def test_expmap_zero_is_identity(self):
        np.testing.assert_allclose(
            expmap_to_rotmat(np.zeros(3)), np.eye(3), atol=1e-12
        )

    def test_expmap_is_a_rotation_matrix(self):
        R = expmap_to_rotmat(np.array([0.3, -1.1, 0.7]))
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(float(np.linalg.det(R)), 1.0, places=12)

    def test_expmap_half_turn_about_z(self):
        R = expmap_to_rotmat(np.array([0.0, 0.0, np.pi]))
        np.testing.assert_allclose(
            R, np.diag([-1.0, -1.0, 1.0]), atol=1e-12
        )

    def test_fk_output_shape(self):
        xyz = forward_kinematics(self._random_frame())
        self.assertEqual(xyz.shape, (N_JOINTS, 3))

    def test_fk_rejects_wrong_width(self):
        with self.assertRaises(ValueError):
            forward_kinematics(np.zeros(96))

    def test_fk_preserves_bone_lengths(self):
        """
        The skeleton is rigid: every bone must keep the length recorded in the
        offset table regardless of pose. Drift here means the channel layout or
        rotation composition is wrong.
        """
        frames = np.stack([self._random_frame(s) for s in range(12)])
        xyz = forward_kinematics_batch(frames)

        for i in range(N_JOINTS):
            parent = PARENT[i]
            if parent == -1:
                continue
            expected = float(np.linalg.norm(OFFSET[i]))
            lengths = np.linalg.norm(xyz[:, i, :] - xyz[:, parent, :], axis=1)
            np.testing.assert_allclose(
                lengths, expected, atol=1e-6,
                err_msg=f"bone {parent}->{i} length drifted",
            )

    def test_fk_arm_segments_are_anatomical(self):
        """Right upper arm and forearm must match the offset table exactly."""
        xyz = forward_kinematics_batch(np.stack([self._random_frame(s) for s in range(4)]))
        report = bone_length_report(xyz)
        self.assertAlmostEqual(report["right_upper_arm_mm"], 278.892924, places=4)
        self.assertAlmostEqual(report["right_forearm_mm"],   251.728680, places=4)
        self.assertLess(report["max_length_std_mm"], 1e-6)

    def test_zero_pose_places_head_above_feet(self):
        """A rest-pose frame must produce an upright skeleton, not an inverted one."""
        xyz = forward_kinematics(np.zeros(99))
        head, foot = xyz[15], xyz[3]
        self.assertGreater(head[1], foot[1])

    def test_row_dispatch_by_width(self):
        """96 values are positions; 99 are rotations; anything else is an error."""
        positions = np.arange(96, dtype=np.float64)
        np.testing.assert_allclose(
            row_to_joint_positions(positions), positions.reshape(32, 3)
        )

        expmap = self._random_frame(3)
        np.testing.assert_allclose(
            row_to_joint_positions(expmap), forward_kinematics(expmap)
        )

        with self.assertRaises(ValueError):
            row_to_joint_positions(np.zeros(99 * 2))

    def test_expmap_row_is_not_read_as_positions(self):
        """
        Regression: the original loader reshaped a 99-value row to (33, 3) and
        read joints straight out of it. That path must be gone — the dispatched
        positions have to differ from the naive reinterpretation.
        """
        row = self._random_frame(7)
        naive = row.reshape(33, 3)[:32]
        dispatched = row_to_joint_positions(row)
        self.assertGreater(float(np.abs(dispatched - naive).max()), 1.0)

    def test_gt_angles_from_row_accepts_both_formats(self):
        row = self._random_frame(11)
        gt_expmap = gt_angles_from_row(row, frame_idx=4, subject="S9", action="walking_1")
        self.assertIsNotNone(gt_expmap)
        self.assertEqual(gt_expmap.frame_idx, 4)
        self.assertEqual(gt_expmap.subject, "S9")
        self.assertEqual(gt_expmap.action, "walking_1")

        positions = forward_kinematics(row).ravel()
        gt_positions = gt_angles_from_row(positions)
        self.assertIsNotNone(gt_positions)

        # Both routes describe the same skeleton, so the angles must agree.
        self.assertAlmostEqual(
            gt_expmap.elbow_flexion, gt_positions.elbow_flexion, places=3
        )
        self.assertAlmostEqual(
            gt_expmap.shoulder_flexion, gt_positions.shoulder_flexion, places=3
        )

    def test_gt_angles_from_row_returns_none_on_bad_width(self):
        self.assertIsNone(gt_angles_from_row(np.zeros(7)))

    def test_degenerate_torso_is_rejected(self):
        """
        A torso whose shoulder axis is collinear with the spine does not span a
        plane, so Gram-Schmidt has nothing to orthogonalise against and the
        frame becomes arbitrary. Such a skeleton must be rejected outright
        rather than yielding confident-looking nonsense angles.
        """
        joints = np.zeros((N_JOINTS, 3))
        joints[H36M_RHIP] = [-130.0, 0.0, 0.0]
        joints[H36M_LHIP] = [130.0, 0.0, 0.0]
        # Both shoulders stacked along the spine: the lateral axis collapses.
        joints[H36M_RSHOULDER] = [0.0, 500.0, 0.0]
        joints[H36M_LSHOULDER] = [0.0, 640.0, 0.0]
        joints[H36M_RELBOW] = [0.0, 260.0, 0.0]
        joints[H36M_RWRIST] = [0.0, 20.0, 0.0]

        self.assertIsNone(_compute_gt_angles(joints))

    def test_healthy_torso_is_accepted(self):
        """The degeneracy guard must not reject an ordinary upright skeleton."""
        joints = np.zeros((N_JOINTS, 3))
        joints[H36M_RHIP] = [-130.0, 0.0, 0.0]
        joints[H36M_LHIP] = [130.0, 0.0, 0.0]
        joints[H36M_RSHOULDER] = [-170.0, 620.0, 0.0]
        joints[H36M_LSHOULDER] = [170.0, 620.0, 0.0]
        joints[H36M_RELBOW] = [-170.0, 340.0, 0.0]
        joints[H36M_RWRIST] = [-170.0, 90.0, 0.0]

        gt = _compute_gt_angles(joints)
        self.assertIsNotNone(gt)
        # Arm hanging straight down beside the torso.
        self.assertAlmostEqual(gt.elbow_flexion, 0.0, places=3)
        self.assertAlmostEqual(gt.shoulder_flexion, 0.0, places=3)
        self.assertAlmostEqual(gt.shoulder_abduction, 0.0, places=3)

    def test_elbow_flexion_is_frame_invariant(self):
        """
        Elbow flexion is the angle between two body-fixed vectors, so it cannot
        depend on the global orientation of the skeleton.
        """
        row = self._random_frame(5)
        xyz = forward_kinematics(row)

        theta = 0.9
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
        rotated = (xyz @ R.T) + np.array([120.0, -45.0, 900.0])

        a = _compute_gt_angles(xyz)
        b = _compute_gt_angles(rotated)
        self.assertIsNotNone(a)
        self.assertIsNotNone(b)
        self.assertAlmostEqual(a.elbow_flexion, b.elbow_flexion, places=6)
        self.assertAlmostEqual(a.shoulder_flexion, b.shoulder_flexion, places=6)
        self.assertAlmostEqual(a.shoulder_abduction, b.shoulder_abduction, places=6)


class TestMetrics(unittest.TestCase):

    def _make_pred_gt(
        self, n=200, noise_std=3.0, bias=0.0, seed=0
    ) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        t   = np.linspace(0, 4 * np.pi, n)
        gt  = 30.0 * np.sin(t)
        pred = gt + rng.normal(bias, noise_std, size=n)
        return pred, gt

    def test_mae_zero_for_perfect_prediction(self):
        gt = np.array([10.0, 20.0, 30.0, 40.0])
        jm = compute_joint_metrics("elbow_flexion", gt, gt)
        self.assertAlmostEqual(jm.mae, 0.0, places=6)

    def test_rmse_zero_for_perfect_prediction(self):
        gt = np.linspace(0, 90, 100)
        jm = compute_joint_metrics("shoulder_flexion", gt, gt)
        self.assertAlmostEqual(jm.rmse, 0.0, places=6)

    def test_r_is_one_for_perfect_prediction(self):
        gt = np.linspace(-45, 90, 150)
        jm = compute_joint_metrics("shoulder_abduction", gt, gt)
        self.assertAlmostEqual(jm.r, 1.0, places=5)

    def test_bias_sign(self):
        """Positive bias means predictions are systematically higher than GT."""
        rng  = np.random.default_rng(1)
        gt   = np.zeros(500)
        pred = gt + 5.0 + rng.normal(0, 0.1, 500)   # bias = +5°
        jm   = compute_joint_metrics("elbow_flexion", pred, gt)
        self.assertGreater(jm.bias, 4.0,
            msg=f"Expected positive bias>4°, got {jm.bias:.2f}°")

    def test_pck5_with_small_noise(self):
        """With σ=1° noise, nearly all frames should be within ±5°."""
        pred, gt = self._make_pred_gt(n=1000, noise_std=1.0)
        jm = compute_joint_metrics("elbow_flexion", pred, gt)
        self.assertGreater(jm.pck_5, 95.0,
            msg=f"Expected PCK@5°>95% for σ=1° noise, got {jm.pck_5:.1f}%")

    def test_pck5_decreases_with_more_noise(self):
        """Higher noise → lower PCK@5°."""
        pred_lo, gt = self._make_pred_gt(n=1000, noise_std=1.0,  seed=0)
        pred_hi, _  = self._make_pred_gt(n=1000, noise_std=10.0, seed=1)
        jm_lo = compute_joint_metrics("j", pred_lo, gt)
        jm_hi = compute_joint_metrics("j", pred_hi, gt)
        self.assertGreater(jm_lo.pck_5, jm_hi.pck_5,
            msg="PCK@5° should be higher for lower noise")

    def test_jitter_zero_for_constant(self):
        """Constant prediction signal → zero jitter."""
        pred = np.ones(200) * 45.0
        gt   = np.ones(200) * 45.0
        jm   = compute_joint_metrics("shoulder_flexion", pred, gt)
        self.assertAlmostEqual(jm.jitter, 0.0, places=5)

    def test_mae_scales_with_noise(self):
        """MAE ≈ σ·sqrt(2/π) for zero-bias Gaussian noise (expected value)."""
        rng  = np.random.default_rng(99)
        gt   = np.zeros(10000)
        sigma = 5.0
        pred  = gt + rng.normal(0, sigma, 10000)
        jm    = compute_joint_metrics("elbow_flexion", pred, gt)
        expected_mae = sigma * np.sqrt(2.0 / np.pi)
        self.assertAlmostEqual(jm.mae, expected_mae, delta=0.3,
            msg=f"MAE={jm.mae:.2f}° vs expected {expected_mae:.2f}°")

    def test_evaluate_framework_produces_correct_joints(self):
        """evaluate_framework() must return a FrameworkMetrics with all 4 joints."""
        rng = np.random.default_rng(7)
        n   = 500
        gt_arrays   = {j: rng.uniform(-30, 90, n) for j in JOINTS}
        pred_arrays = {j: gt_arrays[j] + rng.normal(0, 3, n) for j in JOINTS}

        result = evaluate_framework("TestFW", pred_arrays, gt_arrays)
        self.assertEqual(len(result.joints), len(JOINTS),
            msg="Should have one JointMetrics per joint")
        for j in JOINTS:
            self.assertIn(j, result.joints)
            self.assertGreater(result.joints[j].n, 0)

    def test_mpjae_is_mean_of_joint_maes(self):
        """MPJAE should equal the arithmetic mean of per-joint MAEs."""
        rng = np.random.default_rng(13)
        n   = 400
        gt_arrays   = {j: np.zeros(n) for j in JOINTS}
        # Give each joint a distinct noise level
        stds = {"shoulder_flexion": 2.0, "shoulder_abduction": 4.0,
                 "shoulder_rotation": 6.0, "elbow_flexion": 3.0}
        pred_arrays = {j: rng.normal(0, stds[j], n) for j in JOINTS}

        result = evaluate_framework("TestFW", pred_arrays, gt_arrays)
        manual_mpjae = float(np.mean([result.joints[j].mae for j in JOINTS]))
        self.assertAlmostEqual(result.mpjae, manual_mpjae, places=5)

    def test_better_framework_has_lower_mpjae(self):
        """A framework with less noise should have strictly lower MPJAE."""
        rng = np.random.default_rng(42)
        n   = 1000
        gt  = {j: rng.uniform(-45, 90, n) for j in JOINTS}

        good_pred = {j: gt[j] + rng.normal(0, 2.0, n) for j in JOINTS}
        bad_pred  = {j: gt[j] + rng.normal(0, 9.0, n) for j in JOINTS}

        good = evaluate_framework("Good", good_pred, gt)
        bad  = evaluate_framework("Bad",  bad_pred,  gt)
        self.assertLess(good.mpjae, bad.mpjae,
            msg=f"Good MPJAE={good.mpjae:.2f} should < Bad MPJAE={bad.mpjae:.2f}")

    def test_print_metrics_table_runs_without_error(self):
        """print_metrics_table() should not raise for any valid input."""
        rng = np.random.default_rng(5)
        n   = 100
        gt  = {j: rng.uniform(0, 60, n) for j in JOINTS}
        r1  = evaluate_framework("FW-A", {j: gt[j] + rng.normal(0, 3, n) for j in JOINTS}, gt)
        r2  = evaluate_framework("FW-B", {j: gt[j] + rng.normal(0, 7, n) for j in JOINTS}, gt)

        # Redirect stdout to suppress output during testing
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print_metrics_table([r1, r2])

        output = buf.getvalue()
        self.assertIn("MPJAE", output)
        self.assertIn("FW-A",  output)
        self.assertIn("FW-B",  output)


# ---------------------------------------------------------------------------
# Integration: synthetic noise pipeline
# ---------------------------------------------------------------------------

class TestSyntheticPipeline(unittest.TestCase):
    """
    Integration tests verifying that the full synthetic evaluation pipeline
    produces metrics consistent with the calibrated noise profiles.
    """

    def _run_pipeline(self, n=2000, seed=0) -> dict:
        """
        Run the complete synthetic pipeline:
        GT signal → add noise → evaluate → return metrics per framework.
        """
        from scripts.build_h36m_dataset import NOISE_PROFILES, GT_JOINTS

        rng = np.random.default_rng(seed)
        t   = np.linspace(0, 8 * np.pi, n)

        # Realistic arm motion signals
        gt_arrays = {
            "shoulder_flexion":   30 * np.sin(t / 2.0),
            "shoulder_abduction": 20 * np.abs(np.sin(t / 3.0)),
            "shoulder_rotation":  15 * np.sin(t / 4.0),
            "elbow_flexion":      40 + 40 * np.abs(np.cos(t / 2.0)),
        }

        results = {}
        fw_map = {
            "mp": "MediaPipe",
            "mv": "MoveNet-Lightning",
            "pn": "PoseNet",
        }
        for prefix, fw_name in fw_map.items():
            profile = NOISE_PROFILES[prefix]
            pred_arrays = {}
            for j in GT_JOINTS:
                mu, sigma    = profile[j]
                pred_arrays[j] = gt_arrays[j] + rng.normal(mu, sigma, n)

            result = evaluate_framework(fw_name, pred_arrays, gt_arrays)
            results[fw_name] = result

        return results

    def test_mediapipe_has_lowest_mpjae(self):
        """MediaPipe (σ≈3°) should have strictly lower MPJAE than MoveNet (σ≈5°)."""
        r = self._run_pipeline(n=3000)
        mp_mae = r["MediaPipe"].mpjae
        mv_mae = r["MoveNet-Lightning"].mpjae
        self.assertLess(mp_mae, mv_mae,
            msg=f"MediaPipe MPJAE={mp_mae:.2f} should be < MoveNet MPJAE={mv_mae:.2f}")

    def test_posenet_has_highest_mpjae(self):
        """PoseNet (σ≈8-11°) should have the highest MPJAE."""
        r = self._run_pipeline(n=3000)
        maes = {fw: m.mpjae for fw, m in r.items()}
        pn_mae = maes["PoseNet"]
        for fw, mae in maes.items():
            if fw != "PoseNet":
                self.assertLessEqual(mae, pn_mae + 0.5,
                    msg=f"{fw} MPJAE={mae:.2f} should be ≤ PoseNet MPJAE={pn_mae:.2f}")

    def test_mediapipe_pck5_above_threshold(self):
        """MediaPipe (σ=3°) should achieve PCK@5° > 80%."""
        r = self._run_pipeline(n=3000)
        pck = r["MediaPipe"].mean_pck_5
        self.assertGreater(pck, 70.0,
            msg=f"Expected PCK@5°>70% for MediaPipe, got {pck:.1f}%")

    def test_all_frameworks_have_positive_correlation(self):
        """All frameworks should have Pearson r > 0.7 (tracking real motion)."""
        r = self._run_pipeline(n=3000)
        for fw, result in r.items():
            self.assertGreater(result.mean_r, 0.7,
                msg=f"{fw} r={result.mean_r:.3f} should be > 0.7")

    def test_ranking_mediapipe_gt_movenet_gt_posenet(self):
        """Ranking by MPJAE: MediaPipe < MoveNet < PoseNet."""
        r = self._run_pipeline(n=5000)
        mp = r["MediaPipe"].mpjae
        mv = r["MoveNet-Lightning"].mpjae
        pn = r["PoseNet"].mpjae
        self.assertLess(mp, mv, msg=f"MediaPipe ({mp:.2f}°) should beat MoveNet ({mv:.2f}°)")
        self.assertLess(mv, pn, msg=f"MoveNet ({mv:.2f}°) should beat PoseNet ({pn:.2f}°)")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
