# Obtaining Human3.6M Video for MonoArm Evaluation

MonoArm's framework comparison needs **pixels paired with synchronized 3D ground
truth**. This document covers how to get that data, what the repository already
has (and why it is not sufficient), and how to turn a completed download into a
publication-grade evaluation run.

---

## 1. What is already in this repository, and why it is not enough

`PoseTrack/data/dataset/h3.6m/dataset/` contains the widely-mirrored
**exponential-map** release (`h3.6m.zip`), the one used throughout the human
motion-prediction literature.

| | Exponential-map release (have) | Official release (need) |
|---|---|---|
| Row width | 99 values | 96 values (`D3_Positions`) |
| Contents | 3 root-translation values + 32 joints × 3 axis-angle rotation vectors | 32 joints × 3 Cartesian coordinates (mm) |
| Video | **None** | 4 synchronized cameras, 50 fps |
| Camera calibration | None | Intrinsics + extrinsics per camera |

Two consequences:

1. **You cannot run a pose estimator on it.** There are no images. This blocks
   the framework comparison, the worst-error frame visualization, and any real
   accuracy measurement.
2. **The values are rotations, not positions.** `src/evaluation/h36m_skeleton.py`
   now converts them via forward kinematics, so the exponential-map data does
   yield correct ground-truth *angles* — useful for checking the angle solver,
   but still image-free.

The exponential-map release also contains long dead segments. `S1/walking_1.txt`,
for example, holds roughly 1600 consecutive frames (~32 s) frozen at 144° elbow
flexion with a standard deviation of 1.5°. Segments like these must be filtered
out of any ground-truth set built from it.

---

## 2. Requesting access to the official release

Human3.6M is distributed by the Institute of Mathematics of the Romanian Academy
and is **registration-gated**. There is no anonymous download, and the licence
restricts use to non-commercial academic research. Redistribution is prohibited,
which is why this repository ships no video and no `D3_Positions` files.

**Steps:**

1. Go to <http://vision.imar.ro/human3.6m/> and open the **Download** page.
2. Create an account. Use your institutional address — requests from free
   webmail domains are frequently rejected. For this project, use your IIT
   Bombay address.
3. Complete the request form. It asks for your affiliation, supervisor, and a
   short statement of intended use. State the research purpose plainly:
   monocular arm joint-angle estimation benchmarking, non-commercial, academic.
4. Accept the licence terms.
5. Wait for approval. Turnaround is operator-dependent and has historically
   ranged from a few days to several weeks. **Start this now** — it is the long
   pole in the schedule, and everything downstream is blocked on it.

If approval stalls beyond about two weeks, contact the maintainers listed on the
site directly, and ask your supervisor to send a short confirming note from an
institutional address. That is usually what unsticks a pending request.

> The licence forbids redistribution, so do not source the video from third-party
> mirrors, torrents, or re-uploads. Beyond the licence problem, mirrors are
> frequently re-encoded or subsampled, which breaks the 1:1 frame-to-pose
> correspondence the evaluation depends on.

### While you wait

Nothing in this repository needs to sit idle. Work that does not depend on the
video:

- Ground-truth angle validation against the exponential-map data (works now, via
  forward kinematics).
- The CMU Panoptic Studio path (`scripts/fetch_panoptic_sample.sh`,
  `scripts/evaluate_panoptic.py`) — not registration-gated, and it does provide
  synchronized video plus 3D keypoints.
- Own-capture video for the real-video validation requirement.

---

## 3. What to download once approved

Per subject, from the **Download → D3 Positions** and **Download → Videos**
sections:

| Archive | Contents | Approx. size per subject |
|---|---|---|
| `Videos.tgz` | 4 cameras × ~30 actions, MP4, 50 fps | 10–15 GB |
| `Poses_D3_Positions.tgz` | `.cdf` world-frame 3D joint positions | ~30 MB |

**Minimum useful download:** subjects **S9 and S11** (the standard test split)
for one camera. That alone supports the headline accuracy table.

**Full protocol download:** S1, S5, S6, S7 (tuning), S8 (selection), S9, S11
(test) — roughly 80–100 GB. Only necessary if you intend to tune filter
parameters on the training split rather than fixing them a priori.

Camera IDs: `54138969`, `55011271`, `58860488`, `60457274`. Camera `54138969` is
a reasonable default — it is the most frontal of the four for most actions, which
suits an arm-angle task.

---

## 4. Expected layout after extraction

Extract so that the tree looks like this:

```
data/dataset/h3.6m/raw/
    S9/
        Videos/
            Directions 1.54138969.mp4
            Directions 1.55011271.mp4
            ...
        MyPoseFeatures/
            D3_Positions/
                Directions 1.cdf
                Discussion 1.cdf
                ...
    S11/
        ...
```

This is the archives' native layout, so a plain extraction into
`data/dataset/h3.6m/raw/` should produce it directly.

---

## 5. Preparation

```bash
pip install cdflib          # reads the .cdf pose files

cd python
python scripts/prepare_h36m.py \
    --h36m_root data/dataset/h3.6m/raw \
    --out_dir   data/dataset/h3.6m/prepared \
    --subjects  S9 S11 \
    --camera    54138969 \
    --stride    5
```

`--stride 5` keeps every 5th frame, turning the 50 Hz source into 10 Hz. For
static joint-angle accuracy this loses nothing and cuts inference time fivefold.
Use `--stride 1` when you need the full frame rate for a temporal or smoothness
analysis.

Output:

```
data/dataset/h3.6m/prepared/
    positions/<subject>/<action>.txt              96-value rows (mm)
    frames/<subject>/<action>/<camera>/frame_%06d.jpg
    manifest.json                                 the frame ↔ GT pairing
```

### Why the manifest exists

Frame-to-ground-truth correspondence **cannot** be recovered by sorting file
paths. Ground-truth rows get dropped when a skeleton is degenerate, subjects and
actions iterate in loader order rather than lexicographic order, and video and
pose streams can differ by a few frames at the tail. Any of those shifts the
pairing, and a shifted pairing yields error numbers that look plausible and mean
nothing.

The manifest pins `(subject, action, camera, gt_row, frame_file)` per sample.
`evaluate_h36m.py --mode live` refuses to run without one, and asserts that the
ground-truth array length matches the frame-list length before evaluating.

---

## 6. Evaluation

```bash
python scripts/evaluate_h36m.py \
    --manifest data/dataset/h3.6m/prepared/manifest.json \
    --mode live \
    --frameworks mediapipe movenet_lightning posenet
```

This produces the Excel workbook, frame-level CSV, statistical tests, filter
ablation, LaTeX table, and figures described in the script's docstring — all
tagged `publication_grade=true`.

### Split protocol

Implemented in `src/evaluation/protocol.py` and applied automatically:

| Split | Subjects | Use |
|---|---|---|
| train | S1, S5, S6, S7 | Filter parameter tuning only (Kalman Q/R, window sizes) |
| val | S8 | Filter/model selection |
| test | S9, S11 | Final reported metrics; touched once |

`--protocol loso` runs Leave-One-Subject-Out instead, reporting per-fold metrics
with mean ± std across folds. `assert_no_leakage()` fails loudly if a tuning
subject appears in the test set.

---

## 7. Do not use `--mode synthetic` for results

`--mode synthetic` and `scripts/build_h36m_dataset.py` both fabricate framework
predictions as *ground truth + calibrated Gaussian noise*, with the noise levels
taken from published error rates. They are pipeline smoke tests. Every output is
tagged `publication_grade=false`.

No number from either path may appear in the paper as an experimental result.
