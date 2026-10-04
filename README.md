# Monocular Vision-Based Estimation of Human Arm Joint Angles

Real-time estimation of shoulder and elbow joint angles for both arms from a
single RGB camera, streamed to a Unity humanoid avatar and presented through a
web dashboard that shows how every number was derived.

Built as a low-cost, marker-free reference layer for rehabilitation robotics —
in particular as a calibration and validation signal for soft wearable arm
exoskeletons.

---

## Run it

```bash
cd PoseTrack
pip install -r requirements.txt
python scripts/run_web.py
```

Open <http://127.0.0.1:8000> and allow camera access. Angles start flowing to
Unity on UDP port 9000 at the same time.

For the avatar: open `Unity/UnityMedia` in the Unity Editor, run
**MonoArm → Build Scene**, save, and press Play.

---

## What is here

| Directory | Contents |
|---|---|
| [`PoseTrack/`](PoseTrack/) | the Python system: pose estimation, kinematics, filtering, calibration, UDP streaming, logging, evaluation, and the web dashboard |
| [`Unity/UnityMedia/`](Unity/UnityMedia/) | Unity project: UDP receiver, two interchangeable arm controllers, debug HUD, scene builder |
| [`docs/`](docs/) | project specification, implementation plan, and the IEEE-format paper draft |

Start with [`PoseTrack/README.md`](PoseTrack/README.md) for the system, and
[`PoseTrack/docs/web_pipeline.md`](PoseTrack/docs/web_pipeline.md) for the
front-end/back-end pipeline and API.

---

## How it works

```
single RGB camera
       │
       ▼
2D pose network  (MediaPipe / MoveNet / PoseNet, selectable)
       │  33 keypoints + per-keypoint confidence
       ▼
torso reference frame  built from shoulder and hip keypoints, orthonormalised
       │
       ▼
two-link arm model  →  shoulder flexion, abduction, rotation + elbow flexion, per arm
       │
       ▼
temporal filtering  Kalman (2-state) · moving average · Savitzky–Golay
       │             all three evaluated every frame; one selected for output
       ▼
calibration  per-degree-of-freedom offset and gain from four reference poses
       │
       ├──▶ UDP to Unity     "B,r_flex,r_abd,r_rot,r_elbow,l_flex,…\n"  at 30 Hz
       ├──▶ CSV session log  one row per frame, with tracking and reliability flags
       └──▶ web dashboard    skeleton overlay, live traces, per-frame derivation,
                             latency breakdown, filter comparison, packet inspector
```

The dashboard and the Unity stream are driven by the same frames, so what is
displayed is exactly what the avatar receives.

---

## Design commitments

**Nothing is fabricated.** An arm that cannot be solved reports no angles rather
than zeros, and the traces break instead of drawing through the gap. Shoulder
rotation is withheld when the elbow is too straight for it to be observable from
one camera. Every performance figure shown is measured on the running system,
per stage, and the measurement method is stated alongside it.

**Every number is explained.** The dashboard renders the derivation for the
current frame — keypoints and their confidences, the torso reference frame, the
segment vectors in that frame, the formula for each angle, and the raw →
filtered → calibrated → transmitted chain ending in the literal UDP packet. The
explanatory text is served from the running code, so it cannot drift from the
implementation.

---

## Status

The live pipeline, web dashboard, Unity integration, calibration, logging and
evaluation tooling are implemented and tested (`pip install -r requirements-dev.txt`
then `python -m pytest tests/ -q` in `PoseTrack/`). Demonstration recordings must be captured on hardware with a
camera; `scripts/record_demo.py` and `scripts/run_capture_session.py` produce
them.

`PoseTrack/src/models/` contains experimental research code (2D-to-3D fusion, a
GAN temporal refiner) that is not part of the live pipeline and ships without
trained weights.
