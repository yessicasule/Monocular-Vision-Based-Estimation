# MonoArm — Monocular Vision-Based Estimation of Human Arm Joint Angles for Real-Time Digital Avatar Control

A monocular-camera pipeline that estimates anatomically consistent shoulder
and elbow joint angles from 2D/pseudo-3D body keypoints and drives a rigged
humanoid avatar in Unity in real time — a low-cost, non-contact motion
reference intended as a future calibration/validation signal for soft
wearable arm exoskeletons.

Full mathematical derivations, the statistical evaluation protocol, and
every reproducible result reported for this project are in
[`docs/paper/monoarm_paper.tex`](docs/paper/monoarm_paper.tex). That paper
is explicit about what was and was not measured in this environment — see
its Data Availability / Limitations sections before citing any number from
this repository.

## Repository layout

```
python/         Vision + processing pipeline (pose estimation, angle
                 solver, temporal filters, calibration, UDP streaming,
                 evaluation/benchmarking harness)
  src/pose/          MediaPipe / MoveNet / PoseNet runners behind a
                     common PoseEstimator interface
  src/processing/    coordinate_frame.py, angle_solver.py, angle_filter.py,
                     calibration.py, angle_logger.py
  src/streaming/     udp_streamer.py (Unity), exoskeleton_streamer.py
                     (future wearable-exoskeleton reference channel)
  src/evaluation/    metrics.py, statistics.py, alignment.py, protocol.py,
                     h36m_loader.py, panoptic_loader.py, ablation.py,
                     occlusion_test.py, eval_plots.py, report_export.py
  scripts/           data_generator.py (Task 1), run_demo.py (live
                     pipeline), compare_filters.py, compare_frameworks.py,
                     benchmark_latency.py, evaluate_h36m.py,
                     evaluate_panoptic.py
  webapp/            browser dashboard (FastAPI + WebSocket): live
                     skeleton, per-frame derivation, latency, filter
                     comparison, calibration, logging, UDP packet inspector
  tests/             78 self-contained unit tests (no camera, no GPU,
                     no external dataset required) + 46 dashboard tests

unity/UnityMedia/    Unity project: UdpAngleReceiver.cs, ArmAngleController.cs
                     (AvatarMuscleController), calibration/smoothing scripts,
                     an X Bot Humanoid-rigged avatar, and a MonoArm > Build
                     Scene editor utility

docs/paper/          IEEE-format paper with full derivations and every
                     reproducible result, plus its generated figures
```

## Quick start

### 1 — Python environment
```bash
cd python
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # or a lighter subset — see below
```
`requirements.txt` includes `mediapipe`, `tensorflow`, and
`ai-edge-litert` for the three live pose frameworks. If you only want to
run the self-contained tests and synthetic benchmarks (no camera), a much
lighter install suffices:
```bash
pip install numpy scipy pandas matplotlib pyyaml openpyxl
```

### 2 — Run the self-contained test suite (no camera / dataset needed)
```bash
python -m unittest tests.test_pose_estimators tests.test_evaluation -v
```
All 78 tests should pass; they check the angle-solver math against known
reference-pose geometry, torso-frame orthonormality, filter behavior, the
MoveNet/PoseNet→MediaPipe keypoint remapping, and the metrics/statistics
implementations — entirely with synthetic, exactly-known inputs.

### 3 — Reproduce the paper's filter and occlusion-robustness benchmarks
```bash
python -m scripts.compare_filters
python -m src.evaluation.occlusion_test --synthetic --n_frames 3000
```

### 3b — End-to-end verification on a real photograph
```bash
pip install mediapipe opencv-python-headless
python -m scripts.verify_real_frame
```
Runs the real MediaPipe pose landmarker (not synthetic input) end to end
— image decode → real inference → torso frame → bilateral joint angles →
Kalman filter — on a real sample photo, and writes an annotated overlay
to `outputs/`. See `docs/paper/monoarm_paper.tex` Section "Results:
End-to-End Verification on a Real Photograph" for the reproduced numeric
output and a reproducibility note: some `mediapipe` releases need system
`libegl1`/`libgles2` even for CPU-only inference
(`apt-get install -y libegl1 libgles2` if you hit
`OSError: libEGL.so.1` / `libGLESv2.so.2: cannot open shared object file`).

### 4 — Test the Unity side with no camera (Task 1 → Task 2/3/4)
```bash
python -m scripts.data_generator --mode sinusoidal --hz 30
```
Then in Unity: open `unity/UnityMedia/`, open `Assets/HumanoidScene1.unity`,
run **MonoArm → Build Scene** to auto-wire the receiver/controller, and
press Play. The avatar's arm should track the generator's motion.

### 5 — Live pipeline (camera required)
```bash
python -m scripts.run_demo --filter kalman
```

### 5b — Web dashboard (browser webcam, server camera, or recorded video)
```bash
pip install -r requirements.txt -r requirements-dev.txt
python scripts/run_web.py          # then open http://127.0.0.1:8000
python -m pytest tests/ -q         # full suite incl. the dashboard tests
```
Streams the same `B,` packets to Unity on UDP 9000 while showing how every
number was derived. See [`docs/web_pipeline.md`](docs/web_pipeline.md).

### 6 — Dataset-backed evaluation (requires Human3.6M or CMU Panoptic Studio access)
```bash
python -m scripts.evaluate_h36m --mode live --frame_dir <extracted frames> ...
python -m scripts.evaluate_panoptic --sequence_dir <extracted sequence> ...
```
Neither dataset could be reached from the environment this project was
built in (see the paper's Section "Dataset Access and Reproducing
Real-Data Validation"); both scripts are complete and ready to run
wherever that access exists.

## Communication protocol

UDP, ≥20–30 Hz, newline-terminated packets:
```
S,<shoulder_flexion>,<shoulder_abduction>,<shoulder_rotation>,<elbow_flexion>\n
B,<r_flex>,<r_abd>,<r_rot>,<r_elbow>,<l_flex>,<l_abd>,<l_rot>,<l_elbow>\n   (bilateral)
```
All values in degrees. See `python/src/streaming/udp_streamer.py` and
`unity/UnityMedia/Assets/Scripts/UdpAngleReceiver.cs`.
