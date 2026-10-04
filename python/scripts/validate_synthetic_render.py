"""
validate_synthetic_render.py -- Harness Verification and Render-Domain Measurement
==================================================================================

Two measurements against EXACT ground truth, using the repository's rigged
"X Bot" FBX posed in headless Blender. Neither requires a registration-gated
or network-restricted motion-capture dataset.

Ground truth is exact by construction: the rig is posed, and the TRUE 3D joint
positions are read straight out of the armature -- no annotation error, no
multi-view reconstruction error. Ground-truth angles are then computed with
`h36m_loader._compute_side_gt`, the SAME function used for Human3.6M and CMU
Panoptic ground truth, so the angle convention is identical.

Stage 1 -- HARNESS VERIFICATION (`ideal` landmarks, no rendering)
    The true 3D joints are projected through the known pinhole camera into
    MediaPipe's landmark convention, and the solver runs on those. Because the
    inputs are perfect, any error here is the harness's own: a non-zero result
    would mean the coordinate conventions, the mirror logic, or the GT
    comparison are wrong. This is the meaningful, transferable result -- it
    validates the whole solver + GT-comparison chain end to end, which the
    unit tests (which check components in isolation) do not.

Stage 2 -- RENDER-DOMAIN MEASUREMENT (real MediaPipe on rendered frames)
    The same poses are rendered and the real MediaPipe landmarker runs on the
    images. Reported alongside Stage 1 and alongside raw 2D landmark error in
    pixels.

    READ THIS BEFORE QUOTING STAGE 2. These numbers are NOT an estimate of the
    system's real-world accuracy and must never be reported as one. MediaPipe
    is trained on photographs of people; this is an untextured mannequin under
    studio lighting, and measured 2D landmark error on it is large enough
    (tens of pixels, with catastrophic outliers on wrists) that the resulting
    angle errors characterise the rendering domain gap, not the estimator's
    performance on real video. Real-world accuracy requires
    `evaluate_panoptic.py` on real captured footage -- see
    `fetch_panoptic_sample.sh`.

Usage
-----
    python -m scripts.validate_synthetic_render --stage ideal              # fast, no rendering
    python -m scripts.validate_synthetic_render --stage both --n_poses 40
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_PYTHON = Path(__file__).resolve().parent.parent
if str(REPO_PYTHON) not in sys.path:
    sys.path.insert(0, str(REPO_PYTHON))

from src.evaluation.h36m_loader import _compute_side_gt, _normalize
from src.evaluation.metrics import circular_diff

DEFAULT_FBX = REPO_PYTHON.parent / "unity" / "UnityMedia" / "Assets" / "X Bot.fbx"

BONES = {
    "r_sh": "mixamorig:RightArm",    "r_el": "mixamorig:RightForeArm", "r_wr": "mixamorig:RightHand",
    "l_sh": "mixamorig:LeftArm",     "l_el": "mixamorig:LeftForeArm",  "l_wr": "mixamorig:LeftHand",
    "r_hip": "mixamorig:RightUpLeg", "l_hip": "mixamorig:LeftUpLeg",
}
LANDMARK_IDX = {"l_sh": 11, "r_sh": 12, "l_el": 13, "r_el": 14,
                "l_wr": 15, "r_wr": 16, "l_hip": 23, "r_hip": 24}
DOFS = ["shoulder_flexion", "shoulder_abduction", "shoulder_rotation", "elbow_flexion"]

LENS_MM, SENSOR_MM = 50.0, 36.0
CAMERA_AXIS = np.array([0.0, 1.0, 0.0])   # build_scene puts the camera on -Y looking along +Y
SINGULAR_ABD_DEG = 75.0
IN_PLANE_DEG = 30.0


# ── Scene ────────────────────────────────────────────────────────────────────

def build_scene(fbx_path: Path, resolution: int, samples: int):
    import bpy
    from mathutils import Vector

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=str(fbx_path))

    arm = next(o for o in bpy.data.objects if o.type == 'ARMATURE')
    missing = [b for b in BONES.values() if b not in arm.pose.bones]
    if missing:
        raise RuntimeError(f"Rig is missing expected bones: {missing}")

    mn = Vector((1e9,) * 3); mx = Vector((-1e9,) * 3)
    for m in (o for o in bpy.data.objects if o.type == 'MESH'):
        for c in m.bound_box:
            w = m.matrix_world @ Vector(c)
            mn = Vector((min(mn[i], w[i]) for i in range(3)))
            mx = Vector((max(mx[i], w[i]) for i in range(3)))
    center, size = (mn + mx) / 2, mx - mn

    # Frame the worst case (arms fully out) with margin: a wrist clipped at the
    # frame edge silently corrupts the elbow angle.
    fov = 2 * math.atan(SENSOR_MM / (2 * LENS_MM))
    dist = (max(size.x, size.z) * 1.35 / 2) / math.tan(fov / 2)

    bpy.ops.object.camera_add(location=(center.x, center.y - dist, center.z))
    cam = bpy.context.object
    cam.rotation_euler = (math.radians(90), 0, 0)
    cam.data.lens = LENS_MM
    bpy.context.scene.camera = cam

    bpy.ops.object.light_add(type='SUN',
                             location=(center.x + size.z, center.y - dist, center.z + size.z))
    bpy.context.object.data.energy = 4.0
    bpy.context.object.rotation_euler = (math.radians(50), 0, math.radians(30))

    world = bpy.data.worlds.new("W")
    bpy.context.scene.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[0].default_value = (0.6, 0.62, 0.68, 1)
    world.node_tree.nodes["Background"].inputs[1].default_value = 1.2

    sc = bpy.context.scene
    sc.render.engine = 'CYCLES'
    sc.cycles.device = 'CPU'
    sc.cycles.samples = samples
    sc.render.resolution_x = sc.render.resolution_y = resolution
    sc.render.image_settings.file_format = 'PNG'
    return arm, np.array(cam.location, dtype=np.float64)


def joint_xyz(arm, key) -> np.ndarray:
    return np.array(arm.matrix_world @ arm.pose.bones[BONES[key]].head, dtype=np.float64)


def ground_truth_angles(arm, side: str):
    """GT anatomical angles from true 3D joints (same math as H3.6M/Panoptic GT)."""
    r_sh, l_sh = joint_xyz(arm, "r_sh"), joint_xyz(arm, "l_sh")
    r_hip, l_hip = joint_xyz(arm, "r_hip"), joint_xyz(arm, "l_hip")
    y = _normalize(0.5 * (l_sh + r_sh) - 0.5 * (l_hip + r_hip))
    x = _normalize(l_sh - r_sh)
    x_orth = _normalize(x - np.dot(x, y) * y)
    z = _normalize(np.cross(x_orth, y))
    R = np.column_stack([_normalize(np.cross(y, z)), y, z])
    if side == "right":
        return _compute_side_gt(R, r_sh, joint_xyz(arm, "r_el"), joint_xyz(arm, "r_wr"), mirror=False)
    return _compute_side_gt(R, l_sh, joint_xyz(arm, "l_el"), joint_xyz(arm, "l_wr"), mirror=True)


def pose_rig(arm, rot: dict):
    import bpy
    for pb in arm.pose.bones:
        pb.rotation_mode = 'XYZ'
        pb.rotation_euler = (0, 0, 0)
    for key, (rx, ry, rz) in rot.items():
        pb = arm.pose.bones[BONES[key]]
        pb.rotation_mode = 'XYZ'
        pb.rotation_euler = (math.radians(rx), math.radians(ry), math.radians(rz))
    bpy.context.view_layer.update()


def sample_rotation(rng):
    """
    Sample plausible shoulder/elbow bone rotations for both arms.

    Axis roles were established empirically on this rig: from the T-pose bind,
    shoulder `rx` sweeps the arm within the coronal plane (driving abduction),
    `rz` swings it out of that plane toward/away from the camera, and `ry`
    spins it about its own axis.

    `rz` is kept narrow on purpose. A monocular frontal camera cannot resolve
    an arm pointing along its viewing axis -- shoulder and wrist project to
    nearly the same pixel -- so sampling it uniformly would fill the set with
    physically unobservable poses. A moderate spread is kept so the
    foreshortening strata are populated and that degradation is measured
    rather than assumed.
    """
    def one_arm(sign):
        return {"sh": (rng.uniform(-100, 20), rng.uniform(-35, 35), sign * rng.uniform(-35, 35)),
                "el": (sign * rng.uniform(0, 110), 0.0, 0.0)}
    r, l = one_arm(+1), one_arm(-1)
    return {"r_sh": r["sh"], "r_el": r["el"], "l_sh": l["sh"], "l_el": l["el"]}


def axis_deviation_deg(arm, side: str) -> float:
    """Upper-arm tilt out of the image plane: 0 = in-plane (best), 90 = along camera axis."""
    sh = joint_xyz(arm, "r_sh" if side == "right" else "l_sh")
    el = joint_xyz(arm, "r_el" if side == "right" else "l_el")
    v = _normalize(el - sh)
    return 90.0 - math.degrees(math.acos(float(np.clip(abs(np.dot(v, CAMERA_AXIS)), -1.0, 1.0))))


def project(P: np.ndarray, cam_loc: np.ndarray, depth_ref: float):
    """True 3D point -> ideal landmark in MediaPipe's normalized convention."""
    tan_half = (SENSOR_MM / 2) / LENS_MM
    dx, depth, dz = P[0] - cam_loc[0], P[1] - cam_loc[1], P[2] - cam_loc[2]
    return (0.5 + 0.5 * (dx / depth) / tan_half,
            0.5 - 0.5 * (dz / depth) / tan_half,          # image y grows downward
            (depth - depth_ref) / (2.0 * depth_ref * tan_half))   # negative = nearer camera


# ── Measurement ──────────────────────────────────────────────────────────────

def collect(arm, cam_loc, rng, n_poses, stage, out_dir, keep_renders):
    from src.pose.base import _default_landmarks, Landmark
    from src.processing.angle_solver import compute_bilateral_angles

    do_render = stage in ("render", "both")
    if do_render:
        import bpy, cv2
        from src.pose.mediapipe_runner import MediaPipeRunner

    rows, px_errs, n_undetected = [], {k: [] for k in LANDMARK_IDX}, 0

    for i in range(n_poses):
        pose_rig(arm, sample_rotation(rng))

        gt = {}
        for side in ("right", "left"):
            g = ground_truth_angles(arm, side)
            if g is not None:
                gt[side] = dict(zip(DOFS, g[:4]), rotation_reliable=bool(g[4]))
        if len(gt) != 2:
            continue
        dev = {s: axis_deviation_deg(arm, s) for s in ("right", "left")}

        pts = {k: joint_xyz(arm, k) for k in LANDMARK_IDX}
        depth_ref = 0.5 * (pts["l_hip"][1] + pts["r_hip"][1]) - cam_loc[1]
        ideal = {k: project(pts[k], cam_loc, depth_ref) for k in LANDMARK_IDX}

        ideal_lms = _default_landmarks()
        for k, idx in LANDMARK_IDX.items():
            x, y, z = ideal[k]
            ideal_lms[idx] = Landmark(x=x, y=y, z=z, visibility=1.0)
        preds = {"ideal": compute_bilateral_angles(ideal_lms)}

        if do_render:
            img = out_dir / "renders" / f"pose_{i:04d}.png"
            bpy.context.scene.render.filepath = str(img)
            bpy.ops.render.render(write_still=True)
            # Fresh estimator per pose: independent stills, so no temporal
            # tracking state may carry across unrelated poses.
            runner = MediaPipeRunner()
            real = runner.process(cv2.cvtColor(cv2.imread(str(img)), cv2.COLOR_BGR2RGB))
            runner.close()
            if not keep_renders:
                img.unlink(missing_ok=True)
            if real is None:
                n_undetected += 1
                continue
            res = bpy.context.scene.render.resolution_x
            for k, idx in LANDMARK_IDX.items():
                px_errs[k].append(float(np.hypot((real[idx].x - ideal[k][0]) * res,
                                                 (real[idx].y - ideal[k][1]) * res)))
            preds["render"] = compute_bilateral_angles(real)

        for side in ("right", "left"):
            row = {"pose": i, "side": side, "axis_deviation_deg": dev[side],
                   "gt_rotation_reliable": gt[side]["rotation_reliable"]}
            for dof in DOFS:
                row[f"gt_{dof}"] = gt[side][dof]
            ok = True
            for tag, b in preds.items():
                p = getattr(b, side) if b else None
                if p is None:
                    ok = False
                    break
                for dof in DOFS:
                    row[f"{tag}_{dof}"] = getattr(p, dof)
            if ok:
                rows.append(row)

        if (i + 1) % 10 == 0:
            print(f"  [{i+1:3d}/{n_poses}] {len(rows)} arm-observations")

    return rows, px_errs, n_undetected


def error_table(rows, tag):
    out = {}
    for dof in DOFS:
        rs = rows
        if dof == "shoulder_rotation":
            rs = [r for r in rows if r["gt_rotation_reliable"]]
        if not rs:
            continue
        err = circular_diff(np.array([r[f"{tag}_{dof}"] for r in rs]),
                            np.array([r[f"gt_{dof}"] for r in rs]))
        out[dof] = {"n": len(err), "mae_deg": float(np.mean(np.abs(err))),
                    "rmse_deg": float(np.sqrt(np.mean(err ** 2))),
                    "p90_abs_err_deg": float(np.percentile(np.abs(err), 90))}
    maes = [v["mae_deg"] for v in out.values()]
    return {"per_dof": out, "mpjae_deg": float(np.mean(maes)) if maes else float("nan")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fbx", default=str(DEFAULT_FBX))
    ap.add_argument("--stage", choices=["ideal", "render", "both"], default="both")
    ap.add_argument("--n_poses", type=int, default=40)
    ap.add_argument("--resolution", type=int, default=640)
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", default="outputs/synthetic_validation")
    ap.add_argument("--keep_renders", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    (out_dir / "renders").mkdir(parents=True, exist_ok=True)

    print(f"[->] Building scene from {args.fbx}")
    arm, cam_loc = build_scene(Path(args.fbx), args.resolution, args.samples)
    rng = np.random.default_rng(args.seed)

    rows, px_errs, n_undetected = collect(
        arm, cam_loc, rng, args.n_poses, args.stage, out_dir, args.keep_renders)
    if not rows:
        print("[FAIL] No usable observations."); sys.exit(1)

    in_plane = [r for r in rows if r["axis_deviation_deg"] <= IN_PLANE_DEG]
    well = [r for r in in_plane if abs(r["gt_shoulder_abduction"]) < SINGULAR_ABD_DEG]
    near = [r for r in in_plane if abs(r["gt_shoulder_abduction"]) >= SINGULAR_ABD_DEG]

    report = {
        "n_arm_observations": len(rows),
        "n_poses_undetected": n_undetected,
        "seed": args.seed, "resolution": args.resolution, "cycles_samples": args.samples,
        "stage1_harness_verification": {
            "description": "solver on IDEAL landmarks projected from exact 3D truth; "
                           "non-zero error here would indicate a harness/convention bug",
            "all": error_table(rows, "ideal"),
            "well_conditioned": error_table(well, "ideal") if well else None,
        },
    }
    if args.stage in ("render", "both"):
        report["stage2_render_domain"] = {
            "publication_grade": False,
            "caveat": "Rendered untextured mannequin. NOT an estimate of real-world "
                      "accuracy and must not be reported as one; characterises the "
                      "rendering domain gap. Use evaluate_panoptic.py on real footage.",
            "all": error_table(rows, "render"),
            "in_plane": error_table(in_plane, "render") if in_plane else None,
            "in_plane_well_conditioned": error_table(well, "render") if well else None,
            "in_plane_near_singular": error_table(near, "render") if near else None,
            "landmark_px_error": {
                k: {"median": float(np.median(v)), "mean": float(np.mean(v)),
                    "p90": float(np.percentile(v, 90))}
                for k, v in px_errs.items() if v
            },
        }

    print_report(report, args)
    (out_dir / "synthetic_validation.json").write_text(json.dumps(report, indent=2))
    print(f"\n[OK] Report -> {out_dir}/synthetic_validation.json")


def print_report(rep: dict, args) -> None:
    def table(title, tbl):
        if not tbl or not tbl["per_dof"]:
            return
        print(f"\n  {title}")
        print(f"    {'DOF':<22}{'n':>5}{'MAE':>9}{'RMSE':>9}{'p90':>9}")
        print("    " + "-" * 54)
        for dof, m in tbl["per_dof"].items():
            print(f"    {dof:<22}{m['n']:>5}{m['mae_deg']:>9.2f}{m['rmse_deg']:>9.2f}{m['p90_abs_err_deg']:>9.2f}")
        print(f"    {'MPJAE':<22}{'':>5}{tbl['mpjae_deg']:>9.2f}")

    print("\n" + "=" * 76)
    print("  STAGE 1 -- HARNESS VERIFICATION (ideal landmarks, exact 3D truth)")
    print("=" * 76)
    print("  Near-zero error here means the solver, mirror logic and GT-comparison")
    print("  conventions are correct end to end. Residual is perspective projection.")
    s1 = rep["stage1_harness_verification"]
    table("all observations", s1["all"])
    table("in-plane & well-conditioned", s1.get("well_conditioned"))

    s2 = rep.get("stage2_render_domain")
    if s2:
        print("\n" + "=" * 76)
        print("  STAGE 2 -- RENDERED-IMAGE MEASUREMENT (real MediaPipe)")
        print("=" * 76)
        table("all observations", s2["all"])
        table("in-plane (arm near image plane)", s2.get("in_plane"))
        table("in-plane & well-conditioned", s2.get("in_plane_well_conditioned"))
        table("in-plane & near ZXY abduction pole", s2.get("in_plane_near_singular"))
        px = s2["landmark_px_error"]
        if px:
            print(f"\n  2D landmark localization error vs projected truth "
                  f"({args.resolution}x{args.resolution} render)")
            print(f"    {'joint':<8}{'median':>10}{'mean':>10}{'p90':>10}")
            print("    " + "-" * 38)
            for k, m in px.items():
                print(f"    {k:<8}{m['median']:>10.1f}{m['mean']:>10.1f}{m['p90']:>10.1f}")
        print("\n  " + s2["caveat"])
    print("=" * 76)


if __name__ == "__main__":
    main()
