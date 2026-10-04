# MonoArm / PoseTrack
### Monocular vision-based estimation of human arm joint angles for real-time digital avatar control

Estimates shoulder and elbow joint angles for both arms from a single RGB
camera, filters them, and streams them to a Unity humanoid avatar over UDP in
real time. A browser dashboard shows the whole derivation — keypoints, torso
reference frame, segment vectors, angles, filtering, calibration — next to the
exact packets the avatar is receiving.

---

## Quick start

```bash
cd PoseTrack
pip install -r requirements.txt
python scripts/run_web.py            # then open http://127.0.0.1:8000
```

Allow camera access when the browser asks. The dashboard starts estimating
angles immediately and, unless started with `--no-udp`, streams them to
`127.0.0.1:9000` for Unity at the same time.

For the Unity avatar: open `Unity/UnityMedia` in the Unity Editor, use
**MonoArm → Build Scene**, save, and press Play. See
[`docs/web_pipeline.md`](docs/web_pipeline.md) for the full guide.

### Other entry points

| Command | What it does |
|---|---|
| `python scripts/run_web.py` | **web dashboard + Unity stream** (primary interface) |
| `python scripts/run_demo.py` | OpenCV desktop window version of the live pipeline |
| `python scripts/mock_streamer.py --mode sinusoidal` | drives Unity with generated angles to test the avatar rig without a camera |
| `python scripts/benchmark_latency.py --frames 300` | per-stage latency profile for this machine |
| `python scripts/compare_frameworks.py` | MediaPipe / MoveNet / PoseNet comparison |
| `python scripts/compare_filters.py` | offline filter comparison on recorded angles |
| `python scripts/plot_angles.py --csv <session.csv>` | time-series figure from a logged session |

---

## System overview

```
Webcam ─▶ 2D pose network ─▶ torso reference frame ─▶ two-link joint angles
                                                            │
                                     Kalman / moving average / Savitzky–Golay
                                                            │
                                                     calibration mapping
                                                            │
                            ┌───────────────────────────────┼──────────────────┐
                            ▼                               ▼                  ▼
                     UDP → Unity avatar            CSV session log      web dashboard
```

The three filters run on every frame so raw and filtered signals can be compared
live; only the selected filter is transmitted and logged.

---

## Repository layout

```
PoseTrack/
  webapp/                 web dashboard: FastAPI back-end + browser front-end
    server.py               REST control plane, WebSocket data plane
    pipeline.py             per-frame orchestration and instrumentation
    metrics.py              rolling latency / throughput / stability statistics
    explain.py              description of every reported quantity (served to the UI)
    sources.py              browser / server-camera / recorded-video frame sources
    static/                 index.html, app.js, style.css  (no external assets)
  src/
    pose/                   MediaPipe, MoveNet and PoseNet runners behind one interface
    processing/             torso frame, angle solver, filters, calibration, logging
    streaming/              fixed-rate UDP sender to Unity
    evaluation/             metrics, statistics, plots, occlusion and ablation studies
    models/                 experimental fusion / GAN research code
  scripts/                  entry points and benchmarks
  tests/                    unit and API tests
  docs/                     technical report, web pipeline guide, deliverables index
Unity/UnityMedia/           Unity project: receiver, avatar controllers, scene builder
```

---

## Joint angle conventions

| Channel | Positive direction | Neutral | Notes |
|---|---|---|---|
| Shoulder flexion / extension | forward | 0° arm at side | sagittal-plane rotation of the upper arm |
| Shoulder abduction / adduction | away from the midline | 0° arm at side | mirrored per side so + means "away" for both arms |
| Shoulder internal / external rotation | internal | 0° forearm in the sagittal plane | estimated from the forearm; **unobservable below 25° of elbow flexion** and flagged unreliable there |
| Elbow flexion | forearm toward the upper arm | 0° arm straight | pure dot product between the two segment directions; the most robust channel |

Angles are computed in an orthonormal torso reference frame built from the
shoulder and hip keypoints, so they do not change when the subject turns
relative to the camera. Full derivations are in
[`src/processing/angle_solver.py`](src/processing/angle_solver.py) and
[`docs/technical_report.md`](docs/technical_report.md).

---

## Filters

All three run on every frame; the selected one is what reaches Unity and the log.

| Filter | Class | Parameters | Character |
|---|---|---|---|
| Kalman, 2-state | `KalmanFilter2State` | process noise 0.01, measurement noise 1.5 | tracks angle *and* angular velocity, so it smooths with less lag — the default |
| Moving average | `MovingAverageFilter` | window 7 | cheapest baseline; lags by about half the window |
| Savitzky–Golay | `SavitzkyGolayFilter` | window 11, order 3 | preserves motion peaks a moving average would flatten |

The dashboard shows the rolling standard deviation of each channel for the raw
signal and all three filters side by side, so their effect is measured rather
than asserted.

---

## Communication protocol

```
S,<shoulder_flexion>,<shoulder_abduction>,<shoulder_rotation>,<elbow_flexion>\n
B,<r_flex>,<r_abd>,<r_rot>,<r_elbow>,<l_flex>,<l_abd>,<l_rot>,<l_elbow>\n
```

UDP, UTF-8 text, degrees to two decimal places, newline terminated, sent from a
fixed-rate thread (30 Hz by default) to port 9000. `S,` is the single-arm format
from the project specification; `B,` carries both arms with the right arm first.
An arm that is not tracked in a frame keeps its previous values, so the avatar
holds its last known pose instead of snapping to zero.

`src/streaming/exoskeleton_streamer.py` additionally emits per-joint JSON on
port 9001 with an `APPLY` / `HOLD` action derived from tracking confidence — the
interface intended for future wearable-exoskeleton calibration.

---

## Unity setup

1. Open `Unity/UnityMedia` in the Unity Editor.
2. Menu → **MonoArm → Build Scene** (and **MonoArm → Diagnose Scene** if
   something does not animate).
3. Save the scene and press **Play**.
4. Start the Python side: `python scripts/run_web.py`, or
   `python scripts/mock_streamer.py --mode sinusoidal` to exercise the rig
   without a camera.

Two interchangeable arm controllers are provided; put one on an avatar, not both:

| Script | Mechanism | Trade-off |
|---|---|---|
| `ArmBoneController.cs` | `Transform.localRotation = Quaternion.Euler(...)` on the upper-arm and forearm bones, relative to the rig's rest pose, with frame-rate-independent exponential interpolation | direct and inspectable; the axis assignment per degree of freedom depends on the rig and is exposed in the inspector |
| `ArmAngleController.cs` (`AvatarMuscleController`) | Unity humanoid muscle system via `HumanPoseHandler` | avatar-agnostic and respects the Avatar's joint limits; the degrees-to-muscle mapping is less direct |

---

## Calibration

Four reference poses per arm: `arm_down`, `arm_forward`, `arm_side`,
`elbow_bent`. Each capture averages the last 15 filtered frames. `arm_down` sets
the per-channel offset; the others set the gain.

A reference pose less than 20° from neutral cannot support a gain estimate and
that axis is left uncalibrated with a message saying which pose to repeat; a
fitted gain outside 0.25–4.0 is clamped and reported. Parameters are stored as
JSON and reloaded on the next run.

Run it from the dashboard's calibration panel, or press **C** in
`scripts/run_demo.py`.

---

## Data logging

One CSV row per frame: both arms' four angles, a per-side `tracked` flag, the
rotation-reliability flag, the active filter and whether calibration was
applied. An untracked side is written as empty cells with `tracked = 0`, so
occlusion is distinguishable from a genuine zero.

Sessions can be listed, downloaded, summarised and plotted from the dashboard,
or processed offline with `scripts/plot_angles.py` and
`scripts/compare_filters.py`.

---

## Performance requirements and how they are checked

| Requirement | Target | Measured by |
|---|---|---|
| End-to-end latency | < 100 ms | per-stage `perf_counter` timing shown live; `scripts/benchmark_latency.py` for a full profile |
| Frame rate | ≥ 20 fps | arrival timestamps of completed frames |
| Static-pose stability | ±3–5° after filtering | rolling σ per channel, raw and filtered, in the dashboard's filter comparison |
| Continuous operation | ≥ 10 minutes | session uptime and frame counters in `/api/status` |

Figures depend on the machine; pose inference dominates the budget. Measure on
your own hardware rather than quoting these.

---

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -q
```

`tests/test_webapp.py` covers the instrumentation, the calibration guards, the
HTTP surface and the WebSocket data plane, running the pose network against a
real photograph from the repository. Tests that need an unavailable dependency
skip rather than being stubbed out. `tests/test_refactored.py` additionally
requires PyTorch for the experimental fusion models.

---

## Research and experimental components

`src/models/` and parts of `src/evaluation/` contain research code beyond the
core specification: a fusion network for 2D-to-3D lifting, a GAN temporal
refiner, occlusion robustness benchmarks and an ablation harness. These are
**not** required by the live pipeline and are not exercised by the web
dashboard. No trained checkpoints are stored in the repository — the model
scripts expect weights produced by a separate training run and will say so if
they are missing.

`src/processing/gesture_recognizer.py` classifies five arm gestures
(`REST`, `RAISE_ARM`, `REACH_FORWARD`, `ELBOW_FLEX`, `WAVE`) from the filtered
right-arm angles; the current gesture is reported on every dashboard frame.

---

## Documentation

| Document | Contents |
|---|---|
| [`docs/web_pipeline.md`](docs/web_pipeline.md) | full front-end/back-end pipeline, API reference, measurement notes |
| [`docs/technical_report.md`](docs/technical_report.md) | mathematics, design decisions, evaluation methodology |
| [`docs/deliverables_index.md`](docs/deliverables_index.md) | deliverable-to-file map |
| [`REFERENCES.md`](REFERENCES.md) | bibliography with implementation notes |
