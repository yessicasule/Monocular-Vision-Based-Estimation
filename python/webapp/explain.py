"""
explain.py — Machine-readable description of every quantity the system reports
==============================================================================

The dashboard must not show unexplained numbers. Every angle, every filter,
every pipeline stage and every field of the UDP packet is described once here
and served to the front-end at ``/api/explain``, so the on-screen explanation
and the implementation cannot drift apart.

Sources for the conventions below:
    Wu et al. (2005), ISB recommendation on definitions of joint coordinate
        systems, J. Biomechanics 38(5):981-992.
    Biryukova et al. (2000), Kinematics of the human arm reconstructed from
        spatial tracking system recordings, J. Biomechanics 33(8):985-995.
    Koritnik, Bajd & Munih, A simple kinematic model of a human body for
        virtual environments.
"""

from __future__ import annotations

from src.processing.angle_solver import ROTATION_RELIABLE_THRESHOLD


ANGLE_DOCS = [
    {
        "key": "shoulder_flexion",
        "label": "Shoulder flexion / extension",
        "unit": "deg",
        "sign": "+ forward (flexion), − backward (extension)",
        "neutral": "0° with the arm hanging at the side",
        "typical_range": [-60, 180],
        "formula": "θ = atan2(v_z, −v_y)",
        "description": (
            "Rotation of the upper arm in the sagittal plane. v is the unit "
            "shoulder→elbow vector expressed in the torso frame, so the value "
            "is independent of how the subject is turned relative to the camera."
        ),
        "reads_from": "shoulder and elbow keypoints + torso reference frame",
    },
    {
        "key": "shoulder_abduction",
        "label": "Shoulder abduction / adduction",
        "unit": "deg",
        "sign": "+ away from the midline (abduction), − across the body (adduction)",
        "neutral": "0° with the arm hanging at the side",
        "typical_range": [-45, 90],
        "formula": "θ = arcsin(∓v_x)   (sign mirrored per side)",
        "description": (
            "Elevation of the upper arm away from the trunk in the frontal "
            "plane. The lateral component of the upper-arm vector is mirrored "
            "for the left limb so that positive means 'away from the midline' "
            "on both sides."
        ),
        "reads_from": "shoulder and elbow keypoints + torso reference frame",
    },
    {
        "key": "shoulder_rotation",
        "label": "Shoulder internal / external rotation",
        "unit": "deg",
        "sign": "+ internal rotation, − external rotation",
        "neutral": "0° with the forearm in the sagittal plane",
        "typical_range": [-90, 90],
        "formula": "θ = atan2(f⊥·e₁, f⊥·e₂),  f⊥ = f − (f·u)u",
        "description": (
            "Axial rotation about the long axis of the humerus, estimated from "
            "the forearm used as a pointer. The rotation is geometrically "
            f"unobservable when the elbow is nearly straight, so it is flagged "
            f"unreliable below {ROTATION_RELIABLE_THRESHOLD:.0f}° of elbow "
            "flexion and the dashboard greys the value out rather than "
            "presenting a number that carries no information."
        ),
        "reads_from": "shoulder, elbow and wrist keypoints",
    },
    {
        "key": "elbow_flexion",
        "label": "Elbow flexion",
        "unit": "deg",
        "sign": "+ bending the forearm toward the upper arm",
        "neutral": "0° with the arm fully straight",
        "typical_range": [0, 150],
        "formula": "θ = arccos(û_upper · û_forearm)",
        "description": (
            "Angle between the upper-arm and forearm segment directions. It is "
            "a pure dot product between two vectors, so it needs no reference "
            "frame and is the most robust of the four channels — which is why "
            "it is the one used to check static-pose stability."
        ),
        "reads_from": "shoulder, elbow and wrist keypoints",
    },
]


PIPELINE_STAGES = [
    {
        "key": "decode_ms",
        "label": "Frame decode",
        "description": (
            "JPEG decode of the frame pushed by the browser, or the OpenCV "
            "read when the server owns the camera."
        ),
    },
    {
        "key": "pose_ms",
        "label": "Pose inference",
        "description": (
            "Forward pass of the selected 2D pose network, producing 33 "
            "normalised keypoints with per-keypoint visibility. Usually the "
            "dominant cost of the whole pipeline."
        ),
    },
    {
        "key": "angles_ms",
        "label": "Kinematics",
        "description": (
            "Torso reference frame construction (Gram–Schmidt orthonormalised "
            "from the shoulder and hip keypoints) followed by the two-link "
            "joint-angle solution for both arms. Geometry uses the network's "
            "metric 3D world landmarks when available (MediaPipe), otherwise "
            "keypoints scaled to pixels — never raw normalised coordinates, "
            "whose x and y units differ by the image aspect ratio."
        ),
    },
    {
        "key": "filter_ms",
        "label": "Temporal filtering",
        "description": (
            "All three filter families are evaluated every frame so the "
            "dashboard can compare them live; only the selected one is "
            "transmitted and logged."
        ),
    },
    {
        "key": "calib_ms",
        "label": "Calibration mapping",
        "description": (
            "Per-degree-of-freedom offset and scale fitted from the operator's "
            "reference poses, mapping measured angles onto the avatar's joint "
            "ranges."
        ),
    },
    {
        "key": "stream_ms",
        "label": "Transmit + log",
        "description": (
            "Hand-off of the angles to the fixed-rate UDP sender and, when "
            "logging is on, the CSV row write."
        ),
    },
]


FILTER_DOCS = [
    {
        "key": "kalman",
        "label": "Kalman (2-state)",
        "description": (
            "Constant-velocity model tracking both the joint angle and its "
            "angular velocity. Because it predicts ahead, it removes noise "
            "with less lag than a smoother of comparable strength, which is "
            "what makes it the default for live avatar control."
        ),
        "parameters": "process noise 0.01, measurement noise 1.5, dt = 1 / stream rate",
        "tradeoff": "Lowest latency for a given amount of smoothing; needs tuning.",
    },
    {
        "key": "ma",
        "label": "Moving average",
        "description": (
            "Unweighted mean of the last N samples. The simplest possible "
            "baseline, included so the benefit of the other two is measurable "
            "rather than asserted."
        ),
        "parameters": "window 7 samples",
        "tradeoff": "Cheapest; blunts fast motion and lags by about half the window.",
    },
    {
        "key": "sg",
        "label": "Savitzky–Golay",
        "description": (
            "Least-squares polynomial fit over a sliding window. It preserves "
            "the shape of motion peaks that a moving average would flatten."
        ),
        "parameters": "window 11 samples, polynomial order 3",
        "tradeoff": "Best peak preservation; noticeable lag on abrupt reversals.",
    },
]


PROTOCOL_DOC = {
    "transport": "UDP datagrams, one line per pose, newline terminated",
    "default_port": 9000,
    "rate": "fixed-rate sender thread, default 30 Hz",
    "encoding": "UTF-8 text, degrees, two decimal places",
    "packets": [
        {
            "prefix": "S",
            "format": "S,<shoulder_flexion>,<shoulder_abduction>,<shoulder_rotation>,<elbow_flexion>",
            "description": "Single-arm (right) pose — the format in the project specification.",
        },
        {
            "prefix": "B",
            "format": ("B,<r_flexion>,<r_abduction>,<r_rotation>,<r_elbow>,"
                       "<l_flexion>,<l_abduction>,<l_rotation>,<l_elbow>"),
            "description": "Bilateral pose, right arm first — used when both arms are tracked.",
        },
    ],
    "hold_behaviour": (
        "Nothing is transmitted until the first tracked frame, so the avatar is "
        "never driven by a pose that was not measured. After that, an arm that "
        "is not tracked in a frame keeps its previous values in the packet, so "
        "the avatar holds its last known pose instead of snapping to zero. The "
        "dashboard marks the limb as untracked while this is happening."
    ),
    "unity_receiver": "unity/UnityMedia/Assets/Scripts/UdpAngleReceiver.cs",
}


DATA_FLOW = [
    "Camera frame (browser webcam, server camera, or recorded video)",
    "2D pose network → 33 normalised keypoints + visibility",
    "Isotropic 3D coordinates: metric world landmarks, or keypoints scaled to pixels",
    "Torso reference frame from shoulder and hip keypoints",
    "Segment vectors expressed in the torso frame",
    "Four joint angles per arm (two-link kinematic model)",
    "Temporal filter (Kalman / moving average / Savitzky–Golay)",
    "Calibration offset and scale per degree of freedom",
    "UDP packet to Unity + CSV row in the session log",
]


QUALITY_FLAGS = [
    {
        "key": "detected",
        "description": "A person was found in the frame by the pose network.",
    },
    {
        "key": "tracked",
        "description": (
            "This particular arm produced a valid angle solution. False means "
            "the limb's keypoints were missing or degenerate — the reported "
            "angles for that side are withheld rather than guessed."
        ),
    },
    {
        "key": "rotation_reliable",
        "description": (
            f"Elbow flexion is at least {ROTATION_RELIABLE_THRESHOLD:.0f}°, the "
            "condition under which shoulder rotation is observable from the "
            "forearm."
        ),
    },
    {
        "key": "visibility",
        "description": (
            "Per-keypoint confidence from the pose network in [0, 1]. Values "
            "below about 0.5 usually mean the joint is occluded and the angles "
            "derived from it should be treated with caution."
        ),
    },
]


REQUIREMENTS = [
    {"key": "latency", "target": "end-to-end below 100 ms",
     "measured_by": "per-stage perf_counter timing, p95 shown live"},
    {"key": "frame_rate", "target": "at least 20 FPS",
     "measured_by": "arrival timestamps of completed frames"},
    {"key": "static_stability", "target": "±3–5° angle standard deviation while still",
     "measured_by": "rolling standard deviation per channel, raw and filtered"},
    {"key": "endurance", "target": "10 minutes of continuous operation",
     "measured_by": "session uptime and frame counter"},
]


def explain_payload() -> dict:
    """Full explanation bundle served to the dashboard."""
    return {
        "angles": ANGLE_DOCS,
        "stages": PIPELINE_STAGES,
        "filters": FILTER_DOCS,
        "protocol": PROTOCOL_DOC,
        "data_flow": DATA_FLOW,
        "quality_flags": QUALITY_FLAGS,
        "requirements": REQUIREMENTS,
        "rotation_threshold_deg": ROTATION_RELIABLE_THRESHOLD,
    }
