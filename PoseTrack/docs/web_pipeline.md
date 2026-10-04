# The MonoArm web pipeline

This document describes the browser front-end and Python back-end that make the
monocular arm-tracking system usable, inspectable and streamable end to end, and
explains what every number on the dashboard is measured from.

---

## 1. What it is

A single command starts a web service that

1. takes real camera frames — from the browser's webcam, a camera attached to
   the server, or a recorded video file;
2. estimates 33 body keypoints with a 2D pose network;
3. solves four joint angles per arm from a two-link kinematic model;
4. filters them with three temporal filters *in parallel*;
5. applies a per-user calibration;
6. transmits the result to the Unity avatar over UDP at a fixed rate;
7. logs it to CSV; and
8. returns the whole derivation to the browser, which draws the skeleton, the
   live angle traces, the latency breakdown and the exact packets on the wire.

Steps 6 and 8 happen together. Streaming to Unity is not a separate mode: the
dashboard shows precisely what the avatar is receiving, frame by frame.

```
                      ┌──────────────────────── browser ────────────────────────┐
                      │  getUserMedia → canvas → JPEG ──┐                        │
                      │  skeleton overlay ◀─────────────┤  WebSocket /ws/stream  │
                      │  angle cards, traces, trace panel, latency, packets      │
                      └───────────────────────────────┬─────────────────────────┘
                                                      │
   ┌──────────────────────────────────────────────────▼────────────────────────┐
   │ FastAPI  (webapp/server.py)                                               │
   │   decode ─▶ pose network ─▶ torso frame ─▶ joint angles                   │
   │                                   │                                       │
   │                                   ├─▶ Kalman ─┐                           │
   │                                   ├─▶ moving average  ├─ all evaluated    │
   │                                   └─▶ Savitzky–Golay ─┘  every frame      │
   │                                   │                                       │
   │                            selected filter                                │
   │                                   │                                       │
   │                            calibration mapping                            │
   │                                   ├──────────────▶ CSV session log        │
   │                                   └──────────────▶ UdpAngleSender ──┐     │
   └───────────────────────────────────────────────────────────────────┼─────┘
                                                                        │
                                          UDP :9000  "B,r_flex,…,l_elbow\n"
                                                                        │
                                   ┌────────────────────────────────────▼─────┐
                                   │ Unity — UdpAngleReceiver                  │
                                   │   → ArmBoneController (localRotation), or │
                                   │   → AvatarMuscleController (muscle space) │
                                   └───────────────────────────────────────────┘
```

---

## 2. Running it

```bash
cd PoseTrack
pip install -r requirements.txt
python scripts/run_web.py
```

Open <http://127.0.0.1:8000> and allow camera access.

| Option | Effect |
|---|---|
| `--host 0.0.0.0` | reachable from other devices on the network |
| `--port 8000` | HTTP port |
| `--udp-host` / `--udp-port` | where the Unity avatar is listening (default `127.0.0.1:9000`) |
| `--hz 30` | UDP transmission rate |
| `--filter kalman\|ma\|sg` | filter selected at startup |
| `--no-udp` | run the dashboard without transmitting |

**Camera permissions.** Browsers only grant `getUserMedia` on `localhost` or
over HTTPS. Running the server on the same machine as the browser needs nothing
extra. To use the dashboard from another device, either terminate TLS in front
of the service or select a *server camera* source so the frames never leave the
host.

---

## 3. Frame sources

| Source | Where frames come from | When to use it |
|---|---|---|
| **Browser webcam** | `getUserMedia` in the page, pushed over the WebSocket as JPEG | the normal case: laptop camera, browser and server on one machine |
| **Server camera** | OpenCV `VideoCapture` on the host running Python | the camera is attached to a machine you reach remotely |
| **Recorded video** | a file in `data/raw_videos/` replayed at its own frame rate | reproducible comparisons — identical frames, so a filter or framework change is the only variable |

Server-side sources push results to every connected dashboard and expose an
annotated MJPEG preview at `/api/preview.mjpg`.

---

## 4. What the dashboard shows

### Joint angles
Four channels per arm. The large figure is the value transmitted to Unity
(filtered and calibrated); the small grey figure beside it is the raw geometric
estimate, so the effect of filtering and calibration is visible at a glance.

Shoulder rotation is greyed out whenever elbow flexion is below 25°. In that
posture the forearm is nearly collinear with the upper arm and axial rotation is
not observable from a single camera — no value is claimed rather than a
meaningless one being displayed.

An arm that fails to produce a solution is marked *not tracked* and reports no
angles. It is never reported as zero, and the traces break rather than drawing a
line through the gap.

### Angle traces
Ten seconds of history per channel: the raw estimate as a thin grey line, the
active filter as a thick coloured one. Ticking *all filters* overlays Kalman,
moving average and Savitzky–Golay simultaneously — the live version of the
offline filter comparison.

### How this frame was computed
The derivation for the current frame, in five steps: the keypoints with their
confidences, the orthonormal torso frame, the segment vectors expressed in that
frame, the formula and result for each angle, and the raw → filtered →
transmitted values. The last column of the final table is byte-for-byte what
appears in the UDP packet shown in the Unity panel.

### Pipeline latency
Every stage is timed with `perf_counter()` around the actual call:
frame decode, pose inference, kinematics, filtering, calibration, transmit and
log. The bars show the current frame; the table shows mean, p95 and max over a
rolling window; and the browser round trip is measured on the page's own clock
by stamping each frame at send time and matching the reply.

### Filter comparison
Rolling standard deviation of every channel for the raw signal and for each of
the three filters, colour-coded against the specification's ±3–5° static-pose
band, alongside the input keypoint jitter in pixels. Hold a pose still and the
table shows directly how much noise each filter removes.

### Unity link
Destination, rate, packet and error counters, the literal text of the most
recent packet and a short history. Host, port and rate can be changed while
running.

---

## 5. Calibration

Four reference poses are captured for one arm at a time: `arm_down`,
`arm_forward`, `arm_side`, `elbow_bent`. Each capture averages the last 15
filtered frames (about half a second), so a single noisy frame cannot define the
mapping. `arm_down` fixes the per-channel offset; the other three fix the gain
that maps the measured span onto the anatomical value.

A reference pose that differs from neutral by less than 20° cannot support a
gain estimate — the ratio would be dominated by noise. That axis is left
uncalibrated and the dashboard says which pose to repeat. A gain outside
0.25–4.0 is clamped and reported for the same reason. Without these guards a
hurried calibration silently produces enormous scale factors and sends nonsense
to the avatar.

Parameters are saved to `outputs/web/calibration_<side>.json` and reloaded on
the next start.

---

## 6. Session logging

*Start log* writes one CSV row per frame to `outputs/web/logs/`, with both arms'
angles, the per-side tracked flag, the rotation-reliability flag, the filter in
use and whether calibration was applied. A side that was not tracked is written
as empty cells with `tracked = 0`, so downstream analysis can tell occlusion
apart from a genuine zero reading.

Recorded sessions can be listed, downloaded, summarised (per-channel mean, σ,
range, tracked fraction, achieved rate) and plotted as a four-panel time series,
all computed from the CSV itself.

---

## 7. HTTP and WebSocket API

Control is REST so every action is scriptable with `curl`; the per-frame data
path is a WebSocket.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/health` | liveness |
| `GET` | `/api/status` | configuration, rolling metrics, UDP and calibration state |
| `GET` | `/api/explain` | machine-readable description of every reported quantity |
| `POST` | `/api/filter` | `{"type": "kalman"\|"ma"\|"sg"}` |
| `POST` | `/api/framework` | `{"name": "mediapipe"\|"movenet_lightning"\|"movenet_thunder"\|"posenet"}` |
| `POST` | `/api/udp` | `{"enabled": bool, "host": str, "port": int, "hz": float}` |
| `POST` | `/api/reset` | reset filter state and rolling statistics |
| `POST` | `/api/mirror` | mirror the incoming frame |
| `GET`/`POST` | `/api/calibration[/begin\|capture\|cancel\|clear]` | calibration wizard |
| `POST` | `/api/logging/start\|stop` | CSV session logging |
| `GET` | `/api/sessions` | list recorded sessions |
| `GET` | `/api/sessions/{name}` | download a session CSV |
| `GET` | `/api/sessions/{name}/summary` | statistics computed from that CSV |
| `GET` | `/api/sessions/{name}/plot.png?side=right` | time-series figure |
| `GET` | `/api/sources` | cameras and videos visible to the server |
| `POST` | `/api/source` | select browser / camera / file |
| `GET` | `/api/preview.mjpg` | annotated preview of a server-side source |
| `WS` | `/ws/stream` | JPEG frames up, one JSON result per frame down |

`/api/explain` is generated from `webapp/explain.py`, which is also what the
dashboard's explanation panel renders — the documentation and the running code
cannot drift apart.

### WebSocket messages

Client → server: a binary message is one JPEG frame. Text messages are
`{"type": "rtt", "ms": <measured round trip>}`, `{"type": "ping"}` and
`{"type": "status"}`.

Server → client: `{"type": "hello", …}` on connect, then one `{"type": "frame", …}`
per processed frame carrying landmarks, raw / filtered / calibrated angles for
both arms, every filter's output, the derivation trace, per-stage timings,
rolling metrics and the UDP wire state. A frame that cannot be decoded returns
`{"type": "error", "message": …}` instead of being silently dropped.

The client keeps exactly one frame in flight. The server processes serially, so
sending faster would queue latency rather than raise the achieved rate, and it
would make the round-trip measurement meaningless.

---

## 8. Unity side

`UdpAngleReceiver.cs` binds the UDP port on a background thread and hands parsed
angles to the main thread. `MonoArmManager` forwards them to whichever arm
controller is in the scene:

* **`ArmBoneController`** writes `Transform.localRotation = Quaternion.Euler(...)`
  on the upper-arm and forearm bones, relative to the rig's authored rest pose,
  interpolated with an exponential factor `1 − exp(−dt/τ)` so the motion is
  frame-rate independent. Which local axis carries flexion, abduction and axial
  rotation depends on how the model was rigged, so each degree of freedom is
  routed to a named axis with a sign, editable in the inspector. The defaults
  match the bundled X Bot rig.
* **`AvatarMuscleController`** writes through Unity's humanoid muscle system
  instead. It is avatar-agnostic and respects the Avatar's configured joint
  limits, at the cost of a less direct mapping.

Use one or the other on a given avatar — both would write the same bones each
frame.

### Protocol

```
S,<shoulder_flexion>,<shoulder_abduction>,<shoulder_rotation>,<elbow_flexion>\n
B,<r_flex>,<r_abd>,<r_rot>,<r_elbow>,<l_flex>,<l_abd>,<l_rot>,<l_elbow>\n
```

Degrees, two decimal places, UTF-8, newline terminated, sent from a fixed-rate
thread at 30 Hz by default. Nothing is transmitted until the first tracked
frame, so the avatar is never driven by a pose that was not measured. After
that, an arm that is not tracked in a frame keeps its previous values in the
packet, so the avatar holds its last known pose instead of snapping to zero;
the dashboard marks the limb as untracked while that is happening. A failed
send (network down, host unreachable) is counted and reported in the Unity
panel, and streaming continues.

Joint angles are computed from MediaPipe's metric 3D world landmarks (or, for
2D-only networks, keypoints scaled to pixels) — never from raw normalised
image coordinates, whose x and y units differ by the image aspect ratio.

---

## 9. Measurement notes

Everything the dashboard reports is measured, and the measurement is stated:

* **Frame rate** comes from the arrival timestamps of completed frames, not from
  the configured rate. A gap longer than one second is treated as the stream
  having stopped and starts a new measurement window, so a paused camera does
  not drag the reported rate toward zero.
* **Latency** is the sum of the per-stage timings for that frame. The
  browser round trip is measured separately on the page's own clock, because
  the browser and server clocks cannot be compared directly.
* **Static-pose stability** is the standard deviation of each channel over the
  rolling window — meaningful only while the subject is actually holding still,
  which is why the raw signal is shown next to it for reference.
* **Keypoint jitter** is the standard deviation of each keypoint's pixel
  position over the same window: the input noise the filters must reject.
* **Detection rate** is the fraction of frames in which the pose network found a
  person, counted over the whole session.

Achieved figures depend on the machine. On the container used during
development — no GPU, software JPEG encoding — the server pipeline ran at a p95
of about 48 ms end to end with MediaPipe at 640×360, and the browser round trip
measured a mean of 47 ms, inside the 100 ms budget. Pose inference dominates;
run `scripts/benchmark_latency.py` to profile a specific machine.
