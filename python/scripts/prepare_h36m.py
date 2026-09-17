"""
prepare_h36m.py — Human3.6M Video + 3D Pose Preparation
========================================================

Converts the official (registration-gated) Human3.6M release into the layout
that ``scripts/evaluate_h36m.py --mode live`` consumes, and — critically —
writes an explicit **manifest** recording which extracted video frame
corresponds to which ground-truth pose row.

Why the manifest matters
------------------------
Frame/GT correspondence cannot be recovered by sorting file paths. Ground-truth
rows are dropped when a skeleton is degenerate, subjects and actions iterate in
loader order rather than lexicographic order, and video and pose streams can
differ in length by a few frames at the tail. Any of those silently shifts the
pairing, and a shifted pairing produces plausible-looking but meaningless error
numbers. The manifest pins (subject, action, camera, gt_row, frame_file)
per sample so the evaluation never has to guess.

Expected input layout
---------------------
This is what you get after extracting the per-subject archives downloaded from
http://vision.imar.ro/human3.6m/ (see docs/DATA_H36M.md for how to obtain them):

    <h36m_root>/
        S1/
            Videos/
                Directions 1.54138969.mp4
                Directions 1.55011271.mp4
                ...
            MyPoseFeatures/
                D3_Positions/
                    Directions 1.cdf
                    ...
        S5/ ...

The four camera IDs are 54138969, 55011271, 58860488 and 60457274.

Output layout
-------------
    <out_dir>/
        positions/<subject>/<action>.txt          96-value CSV rows (mm),
                                                  readable by h36m_loader
        frames/<subject>/<action>/<camera>/frame_%06d.jpg
        manifest.json                             sample-level pairing record

Usage
-----
    # One camera, every 5th frame (50 Hz -> 10 Hz), test subjects only
    python scripts/prepare_h36m.py \
        --h36m_root  data/dataset/h3.6m/raw \
        --out_dir    data/dataset/h3.6m/prepared \
        --subjects   S9 S11 \
        --camera     54138969 \
        --stride     5

    # Restrict to a few actions while validating the pipeline
    python scripts/prepare_h36m.py --h36m_root ... --out_dir ... \
        --actions directions_1 walking_1 --max_frames_per_action 200

Requires ``cdflib`` to read the .cdf pose files:

    pip install cdflib
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

# Repo root on the path so `src.` imports resolve when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.h36m_skeleton import N_JOINTS

H36M_CAMERAS = ["54138969", "55011271", "58860488", "60457274"]

MANIFEST_VERSION = 1


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------
def canonical_action(raw: str) -> str:
    """
    Normalise an H3.6M action name to the repo's canonical form.

    The official release names files ``Directions 1.cdf`` / ``TakingPhoto
    1.mp4``, while the loader, the split protocol and the existing
    exponential-map files all key on ``directions_1``. Normalising here keeps
    ground truth, frames and manifest entries on one vocabulary.

        "Directions 1"      -> "directions_1"
        "WalkingDog 1"      -> "walkingdog_1"
        "Photo 1"           -> "takingphoto_1"
        "SittingDown 2"     -> "sittingdown_2"
    """
    name = raw.strip()
    name = re.sub(r"\.(cdf|mp4)$", "", name, flags=re.IGNORECASE)
    name = name.replace("_", " ")
    name = re.sub(r"\s+", " ", name).strip().lower()
    name = name.replace(" ", "_")

    # The release is inconsistent about a handful of action names.
    aliases = {
        "photo":            "takingphoto",
        "takingphoto":      "takingphoto",
        "walkdog":          "walkingdog",
        "walkingdog":       "walkingdog",
        "walktogether":     "walkingtogether",
        "walkingtogether":  "walkingtogether",
    }
    parts = name.split("_")
    parts[0] = aliases.get(parts[0], parts[0])
    return "_".join(parts)


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------
def read_d3_positions(cdf_path: Path) -> np.ndarray:
    """
    Read one D3_Positions .cdf file into (n_frames, 32, 3) millimetre positions.

    Raises
    ------
    ImportError
        If cdflib is not installed.
    ValueError
        If the file does not contain a recognisable (T, 96) pose array.
    """
    try:
        import cdflib
    except ImportError as exc:
        raise ImportError(
            "cdflib is required to read Human3.6M D3_Positions .cdf files.\n"
            "    pip install cdflib"
        ) from exc

    cdf = cdflib.CDF(str(cdf_path))
    info = cdf.cdf_info()
    var_names = list(getattr(info, "zVariables", []) or []) + \
                list(getattr(info, "rVariables", []) or [])
    if not var_names:
        raise ValueError(f"{cdf_path.name}: no variables found in CDF")

    # The pose variable is named "Pose" in the official release; fall back to
    # the first variable that has the right trailing dimension.
    name = "Pose" if "Pose" in var_names else var_names[0]
    data = np.asarray(cdf.varget(name), dtype=np.float64)
    data = np.squeeze(data)

    if data.ndim != 2 or data.shape[1] != N_JOINTS * 3:
        raise ValueError(
            f"{cdf_path.name}: expected (T, {N_JOINTS * 3}) pose array, "
            f"got {data.shape}"
        )
    return data.reshape(data.shape[0], N_JOINTS, 3)


def count_video_frames(video_path: Path) -> int:
    """
    Frame count of a video, preferring the container metadata and falling back
    to a decode pass when the metadata is absent or implausible.
    """
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video {video_path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if n > 0:
        return n

    cap = cv2.VideoCapture(str(video_path))
    n = 0
    while cap.grab():
        n += 1
    cap.release()
    return n


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------
@dataclass
class Sample:
    """One paired (video frame, ground-truth pose) sample."""
    subject:      str
    action:       str
    camera:       str
    gt_row:       int      # row index into positions/<subject>/<action>.txt
    video_frame:  int      # 0-based frame index in the source video
    frame_file:   str      # path relative to <out_dir>


def write_positions_txt(out_path: Path, positions: np.ndarray) -> None:
    """
    Write (n_frames, 32, 3) positions as 96-value comma-separated rows.

    This is the same width h36m_loader dispatches to the D3_Positions branch,
    so the prepared ground truth flows through the existing loader unchanged.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    flat = positions.reshape(positions.shape[0], -1)
    np.savetxt(out_path, flat, delimiter=",", fmt="%.6f")


def extract_frames(
    video_path: Path,
    out_dir:    Path,
    indices:    list[int],
    quality:    int = 95,
) -> list[int]:
    """
    Decode `video_path` once and write the requested frame indices as JPEGs.

    Sequential decoding with a membership test beats per-frame seeking: H3.6M
    videos are long-GOP H.264, where random seeks are both slow and prone to
    landing on the wrong frame.

    Returns
    -------
    list[int]
        The indices actually written, in ascending order.
    """
    import cv2

    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = set(indices)
    written: list[int] = []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video {video_path}")

    idx = 0
    last = max(wanted) if wanted else -1
    while idx <= last:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in wanted:
            path = out_dir / f"frame_{idx:06d}.jpg"
            cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            written.append(idx)
        idx += 1
    cap.release()
    return written


def prepare_action(
    subject:     str,
    action:      str,
    cdf_path:    Path,
    video_paths: dict[str, Path],
    out_dir:     Path,
    stride:      int,
    max_frames:  int | None,
) -> list[Sample]:
    """
    Prepare one (subject, action): write ground-truth positions, extract the
    selected video frames for each requested camera, and return the manifest
    samples pairing them.
    """
    positions = read_d3_positions(cdf_path)
    n_pose = positions.shape[0]

    write_positions_txt(out_dir / "positions" / subject / f"{action}.txt", positions)

    samples: list[Sample] = []
    for camera, video_path in sorted(video_paths.items()):
        n_video = count_video_frames(video_path)

        # Pose and video are both 50 Hz and frame-synchronous, but the tails can
        # differ by a few frames; clip to the shorter stream rather than
        # extrapolating.
        n = min(n_pose, n_video)
        if n <= 0:
            print(f"    [!] {subject}/{action}/{camera}: no overlapping frames "
                  f"(pose {n_pose}, video {n_video}) — skipped")
            continue
        if abs(n_pose - n_video) > 10:
            print(f"    [!] {subject}/{action}/{camera}: pose/video length "
                  f"mismatch ({n_pose} vs {n_video}); using first {n}")

        indices = list(range(0, n, stride))
        if max_frames is not None:
            indices = indices[:max_frames]

        frame_dir = out_dir / "frames" / subject / action / camera
        written = extract_frames(video_path, frame_dir, indices)

        for i in written:
            rel = (Path("frames") / subject / action / camera /
                   f"frame_{i:06d}.jpg").as_posix()
            samples.append(Sample(
                subject=subject, action=action, camera=camera,
                gt_row=i, video_frame=i, frame_file=rel,
            ))

        missing = len(indices) - len(written)
        note = f" ({missing} not decoded)" if missing else ""
        print(f"    {subject}/{action}/{camera}: {len(written)} frames"
              f" of {n} @ stride {stride}{note}")

    return samples


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
def discover(
    h36m_root: Path,
    subjects:  list[str],
    cameras:   list[str],
    actions:   list[str] | None,
) -> dict[tuple[str, str], tuple[Path, dict[str, Path]]]:
    """
    Locate (cdf, {camera: video}) pairs under the raw release.

    Returns
    -------
    dict keyed by (subject, canonical_action).
    """
    found: dict[tuple[str, str], tuple[Path, dict[str, Path]]] = {}

    for subject in subjects:
        pose_dir  = h36m_root / subject / "MyPoseFeatures" / "D3_Positions"
        video_dir = h36m_root / subject / "Videos"

        if not pose_dir.is_dir():
            print(f"[!] {subject}: no D3_Positions directory at {pose_dir} — skipped")
            continue
        if not video_dir.is_dir():
            print(f"[!] {subject}: no Videos directory at {video_dir} — skipped")
            continue

        for cdf_path in sorted(pose_dir.glob("*.cdf")):
            action = canonical_action(cdf_path.stem)
            if actions and action not in actions:
                continue

            videos: dict[str, Path] = {}
            for cam in cameras:
                matches = [
                    p for p in video_dir.glob(f"*.{cam}.mp4")
                    if canonical_action(p.stem.rsplit(".", 1)[0]) == action
                ]
                if matches:
                    videos[cam] = matches[0]

            if not videos:
                print(f"[!] {subject}/{action}: no video for cameras "
                      f"{cameras} — skipped")
                continue

            found[(subject, action)] = (cdf_path, videos)

    return found


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Prepare Human3.6M video + 3D pose data for live evaluation."
    )
    ap.add_argument("--h36m_root", required=True,
                    help="Root of the extracted official release (contains S1/, S5/, ...)")
    ap.add_argument("--out_dir", required=True,
                    help="Destination for prepared positions, frames and manifest")
    ap.add_argument("--subjects", nargs="+",
                    default=["S9", "S11"],
                    help="Subjects to prepare (default: the S9/S11 test split)")
    ap.add_argument("--actions", nargs="+", default=None,
                    help="Canonical action names to restrict to (e.g. directions_1)")
    ap.add_argument("--camera", nargs="+", default=["54138969"],
                    choices=H36M_CAMERAS,
                    help="Camera IDs to extract (default: 54138969)")
    ap.add_argument("--stride", type=int, default=5,
                    help="Keep every Nth frame; 50 Hz source, so 5 gives 10 Hz "
                         "(default: 5)")
    ap.add_argument("--max_frames_per_action", type=int, default=None,
                    help="Cap kept frames per action per camera")
    args = ap.parse_args()

    if args.stride < 1:
        print("[x] --stride must be >= 1")
        return 2

    h36m_root = Path(args.h36m_root)
    out_dir   = Path(args.out_dir)

    if not h36m_root.is_dir():
        print(f"[x] --h36m_root does not exist: {h36m_root}")
        return 2

    actions = [canonical_action(a) for a in args.actions] if args.actions else None

    print(f"[>] Scanning {h36m_root}")
    found = discover(h36m_root, args.subjects, args.camera, actions)
    if not found:
        print("[x] Nothing to prepare. Check --h36m_root layout against "
              "docs/DATA_H36M.md, and confirm the Videos and D3_Positions "
              "archives are both extracted.")
        return 1

    print(f"[>] {len(found)} (subject, action) pairs to prepare\n")

    samples: list[Sample] = []
    for (subject, action), (cdf_path, videos) in sorted(found.items()):
        print(f"  {subject}/{action}")
        try:
            samples.extend(prepare_action(
                subject, action, cdf_path, videos, out_dir,
                args.stride, args.max_frames_per_action,
            ))
        except Exception as exc:
            print(f"    [x] failed: {exc}")

    if not samples:
        print("\n[x] No samples produced.")
        return 1

    manifest = {
        "version":     MANIFEST_VERSION,
        "source":      "Human3.6M D3_Positions + Videos",
        "h36m_root":   str(h36m_root),
        "subjects":    sorted({s.subject for s in samples}),
        "actions":     sorted({s.action for s in samples}),
        "cameras":     sorted({s.camera for s in samples}),
        "stride":      args.stride,
        "source_fps":  50.0,
        "sample_fps":  50.0 / args.stride,
        "n_samples":   len(samples),
        "positions_dir": "positions",
        "frames_dir":    "frames",
        "samples":     [asdict(s) for s in samples],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\n[OK] {len(samples):,} paired samples across "
          f"{len(manifest['subjects'])} subjects, "
          f"{len(manifest['actions'])} actions, "
          f"{len(manifest['cameras'])} camera(s)")
    print(f"[OK] Manifest: {manifest_path}")
    print(f"\nNext:\n"
          f"    python scripts/evaluate_h36m.py \\\n"
          f"        --manifest {manifest_path} \\\n"
          f"        --mode live --frameworks mediapipe movenet_lightning posenet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
