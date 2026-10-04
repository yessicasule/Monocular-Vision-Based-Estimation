"""
test_webapp.py — Tests for the web dashboard back-end
======================================================

Covers the instrumentation, the calibration robustness guard, the explanation
payload, and the HTTP/WebSocket surface. Tests that need a pose network run
against a real photograph from the repository rather than a synthetic image,
and are skipped (not faked) when the network or the frame is unavailable.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.processing.angle_solver import ArmAngles                     # noqa: E402
from src.processing.calibration import (                              # noqa: E402
    MIN_REFERENCE_SPAN_DEG,
    POSE_ARM_DOWN,
    POSE_ARM_FORWARD,
    MAX_GAIN,
    MIN_GAIN,
    CalibrationManager,
)
from webapp.explain import explain_payload                            # noqa: E402
from webapp.metrics import (                                          # noqa: E402
    LatencyTracker,
    StabilityTracker,
    StageTimings,
    ThroughputTracker,
    _percentile,
    _std,
)
from webapp.pipeline import (                                         # noqa: E402
    FILTER_TYPES,
    LivePipeline,
    PipelineConfig,
    _mean_angles,
    angles_to_dict,
    bilateral_to_dict,
)

SAMPLE_FRAME = ROOT.parent / "docs" / "paper" / "figures" / "panoptic_171204_pose1_sample_frame50.jpg"


def _angles(flex=0.0, abd=0.0, rot=0.0, elb=0.0, side="right") -> ArmAngles:
    return ArmAngles(flex, abd, rot, elb, rotation_reliable=elb >= 25.0, side=side)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

class TestStatistics:
    def test_std_matches_numpy(self):
        vals = [3.0, 1.5, -2.0, 7.25, 0.0]
        assert _std(vals) == pytest.approx(float(np.std(vals)))

    def test_std_of_short_sequence_is_zero(self):
        assert _std([]) == 0.0
        assert _std([42.0]) == 0.0

    def test_percentile_interpolates(self):
        s = [0.0, 10.0, 20.0, 30.0]
        assert _percentile(s, 0.0) == 0.0
        assert _percentile(s, 1.0) == 30.0
        assert _percentile(s, 0.5) == pytest.approx(15.0)


class TestStageTimings:
    def test_total_is_the_sum_of_stages(self):
        t = StageTimings(decode_ms=1.0, pose_ms=20.0, angles_ms=0.5,
                         filter_ms=0.25, calib_ms=0.05, stream_ms=0.2)
        t.finalise()
        assert t.total_ms == pytest.approx(22.0)
        assert t.as_dict()["total_ms"] == pytest.approx(22.0)


class TestLatencyTracker:
    def test_budget_compliance_counts_only_conforming_frames(self):
        tr = LatencyTracker(window=10)
        for total in (10.0, 20.0, 500.0, 30.0):
            t = StageTimings(pose_ms=total)
            t.finalise()
            tr.add(t)
        assert tr.budget_compliance(100.0) == pytest.approx(0.75)

    def test_empty_tracker_reports_zeros(self):
        tr = LatencyTracker()
        assert tr.summary()["total_ms"]["n"] == 0
        assert tr.budget_compliance() == 0.0


class TestThroughputTracker:
    def test_idle_gap_restarts_the_rate_window(self, monkeypatch):
        clock = {"t": 0.0}
        monkeypatch.setattr("webapp.metrics.time.perf_counter", lambda: clock["t"])
        tr = ThroughputTracker(window=90, gap_s=1.0)
        for _ in range(10):                    # a 20 Hz burst
            tr.add(True)
            clock["t"] += 0.05
        assert tr.fps == pytest.approx(20.0, rel=1e-6)

        clock["t"] += 30.0                     # stream paused
        for _ in range(5):                     # a 50 Hz burst
            tr.add(True)
            clock["t"] += 0.02
        # The long idle gap must not drag the reported rate down.
        assert tr.fps == pytest.approx(50.0, rel=1e-6)
        assert tr.frames == 15                 # lifetime counter keeps counting

    def test_detection_rate(self):
        tr = ThroughputTracker()
        for detected in (True, True, False, True):
            tr.add(detected)
        assert tr.detection_rate == pytest.approx(0.75)


class TestStabilityTracker:
    def test_reports_per_channel_standard_deviation(self):
        tr = StabilityTracker(window=50)
        for v in (10.0, 12.0, 14.0):
            tr.add("raw", "right", _angles(flex=v))
        out = tr.summary()["raw:right"]
        assert out["shoulder_flexion"] == pytest.approx(_std([10.0, 12.0, 14.0]), abs=1e-3)
        assert out["elbow_flexion"] == 0.0

    def test_none_angles_are_ignored(self):
        tr = StabilityTracker()
        tr.add("raw", "right", None)
        assert tr.summary() == {}


# ---------------------------------------------------------------------------
# serialisation helpers
# ---------------------------------------------------------------------------

class TestSerialisation:
    def test_untracked_limb_serialises_as_null_not_zero(self):
        assert angles_to_dict(None) is None
        assert bilateral_to_dict(None) == {"right": None, "left": None}

    def test_angles_round_trip_through_json(self):
        d = angles_to_dict(_angles(12.345, -3.21, 44.0, 90.0))
        assert json.loads(json.dumps(d))["shoulder_flexion"] == pytest.approx(12.35)
        assert d["rotation_reliable"] is True

    def test_mean_angles_averages_every_channel(self):
        mean = _mean_angles([_angles(0, 0, 0, 20), _angles(10, 20, 30, 40)])
        assert mean.shoulder_flexion == pytest.approx(5.0)
        assert mean.shoulder_abduction == pytest.approx(10.0)
        assert mean.elbow_flexion == pytest.approx(30.0)
        # reliability is conjunctive: one unreliable sample taints the mean
        assert mean.rotation_reliable is False


# ---------------------------------------------------------------------------
# calibration robustness
# ---------------------------------------------------------------------------

class TestCalibrationGuards:
    def _run(self, forward_flexion: float) -> CalibrationManager:
        mgr = CalibrationManager()
        mgr.begin_static()
        assert mgr.next_pose() == POSE_ARM_DOWN
        mgr.capture_pose(_angles(flex=0.0))
        assert mgr.next_pose() == POSE_ARM_FORWARD
        mgr.capture_pose(_angles(flex=forward_flexion))
        mgr.finalise()
        return mgr

    def test_well_separated_reference_poses_fit_a_scale(self):
        mgr = self._run(forward_flexion=80.0)
        assert mgr.data.flexion.scale == pytest.approx(90.0 / 80.0)
        assert not any("flexion" in w for w in mgr.warnings)

    def test_degenerate_reference_pose_is_refused_not_scaled(self):
        # A reference pose barely different from neutral would otherwise yield
        # an enormous gain and send nonsense to the avatar.
        mgr = self._run(forward_flexion=MIN_REFERENCE_SPAN_DEG / 4)
        assert mgr.data.flexion.scale == 1.0
        assert any("uncalibrated" in w for w in mgr.warnings)

    def test_implausible_gain_is_clamped_and_reported(self):
        mgr = self._run(forward_flexion=35.0)     # would need a 2.6x gain
        assert MIN_GAIN <= mgr.data.flexion.scale <= MAX_GAIN
        assert mgr.data.flexion.scale == pytest.approx(MAX_GAIN)
        assert any("clamped" in w for w in mgr.warnings)

    def test_calibration_is_identity_before_it_is_run(self):
        mgr = CalibrationManager()
        a = _angles(31.0, -4.0, 5.0, 88.0)
        out = mgr.apply(a)
        assert out.shoulder_flexion == a.shoulder_flexion
        assert out.elbow_flexion == a.elbow_flexion


# ---------------------------------------------------------------------------
# explanation payload
# ---------------------------------------------------------------------------

class TestExplainPayload:
    def test_every_reported_channel_is_documented(self):
        payload = explain_payload()
        documented = {a["key"] for a in payload["angles"]}
        assert documented == {
            "shoulder_flexion", "shoulder_abduction",
            "shoulder_rotation", "elbow_flexion",
        }

    def test_stage_keys_match_the_measured_timings(self):
        payload = explain_payload()
        measured = set(StageTimings().as_dict())
        for stage in payload["stages"]:
            assert stage["key"] in measured

    def test_every_filter_family_is_documented(self):
        keys = {f["key"] for f in explain_payload()["filters"]}
        assert keys == set(FILTER_TYPES)

    def test_payload_is_json_serialisable(self):
        json.dumps(explain_payload())


# ---------------------------------------------------------------------------
# live pipeline (needs a pose network and a real frame)
# ---------------------------------------------------------------------------

def _load_sample_frame():
    cv2 = pytest.importorskip("cv2")
    if not SAMPLE_FRAME.exists():
        pytest.skip(f"sample frame not present at {SAMPLE_FRAME}")
    frame = cv2.imread(str(SAMPLE_FRAME))
    if frame is None:
        pytest.skip("sample frame could not be decoded")
    return cv2.resize(frame, (640, 360))


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    pytest.importorskip("mediapipe")
    cfg = PipelineConfig(udp_enabled=False)
    pipe = LivePipeline(cfg, tmp_path_factory.mktemp("web"))
    try:
        pipe.ensure_estimator()
    except Exception as exc:                       # environment without GL, etc.
        pytest.skip(f"pose estimator unavailable: {exc}")
    yield pipe
    pipe.close()


class TestLivePipeline:
    def test_real_frame_yields_angles_for_both_arms(self, pipeline):
        result = pipeline.process_frame(_load_sample_frame())
        assert result.detected
        assert result.raw["right"] is not None
        assert result.raw["left"] is not None
        assert 0.0 <= result.raw["right"]["elbow_flexion"] <= 180.0

    def test_all_filter_families_run_every_frame(self, pipeline):
        result = pipeline.process_frame(_load_sample_frame())
        assert set(result.filter_bank) == set(FILTER_TYPES)
        for ft in FILTER_TYPES:
            assert result.filter_bank[ft]["right"] is not None

    def test_timings_are_measured_and_consistent(self, pipeline):
        t = pipeline.process_frame(_load_sample_frame()).timings
        assert t["pose_ms"] > 0.0
        assert t["total_ms"] == pytest.approx(
            sum(t[k] for k in t if k != "total_ms"), abs=0.01
        )

    def test_trace_exposes_the_intermediate_geometry(self, pipeline):
        trace = pipeline.process_frame(_load_sample_frame()).trace
        assert set(trace["torso_frame"]) == {
            "origin", "x_axis_lateral", "y_axis_superior", "z_axis_anterior",
        }
        right = trace["sides"]["right"]
        assert right["landmark_indices"] == {"shoulder": 12, "elbow": 14, "wrist": 16}
        assert len(right["upper_arm_torso"]) == 3
        # the torso axes must be unit vectors, or the decomposition is invalid
        for axis in ("x_axis_lateral", "y_axis_superior", "z_axis_anterior"):
            assert np.linalg.norm(trace["torso_frame"][axis]) == pytest.approx(1.0, abs=1e-3)

    def test_blank_frame_reports_no_detection_without_inventing_angles(self, pipeline):
        blank = np.zeros((360, 640, 3), dtype=np.uint8)
        result = pipeline.process_frame(blank)
        assert result.detected is False
        assert result.raw == {"right": None, "left": None}
        assert result.calibrated == {"right": None, "left": None}
        assert result.trace == {}

    def test_result_is_json_serialisable(self, pipeline):
        json.dumps(pipeline.process_frame(_load_sample_frame()).as_dict())

    def test_filter_selection_is_validated(self, pipeline):
        pipeline.set_filter("ma")
        assert pipeline.config.filter_type == "ma"
        with pytest.raises(ValueError):
            pipeline.set_filter("not-a-filter")
        pipeline.set_filter("kalman")

    def test_capture_without_a_tracked_arm_is_refused(self, pipeline):
        pipeline.calibration_begin("right")
        pipeline._recent["right"].clear()
        with pytest.raises(RuntimeError, match="not.*tracked|step into frame"):
            pipeline.calibration_capture()
        pipeline.calibration_cancel()


# ---------------------------------------------------------------------------
# HTTP surface
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    pytest.importorskip("fastapi")
    starlette_testclient = pytest.importorskip("starlette.testclient")
    from webapp import server as server_module

    server_module.state.pipeline.set_udp(enabled=False)
    with starlette_testclient.TestClient(server_module.app) as c:
        yield c


class TestHttpApi:
    def test_health(self, client):
        assert client.get("/api/health").json()["ok"] is True

    def test_dashboard_page_is_served(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "MonoArm" in r.text

    def test_status_reports_configuration_and_metrics(self, client):
        st = client.get("/api/status").json()
        assert "metrics" in st and "udp" in st and "calibration" in st
        assert st["filters_available"] == list(FILTER_TYPES)

    def test_filter_endpoint_accepts_known_and_rejects_unknown(self, client):
        assert client.post("/api/filter", json={"type": "sg"}).json()["filter_type"] == "sg"
        assert client.post("/api/filter", json={"type": "nope"}).status_code == 400
        client.post("/api/filter", json={"type": "kalman"})

    def test_unknown_framework_is_rejected(self, client):
        assert client.post("/api/framework", json={"name": "nope"}).status_code == 400

    def test_session_paths_cannot_escape_the_log_directory(self, client):
        for name in ("../../../etc/passwd", "..%2Fsecret.csv", "nonexistent.csv"):
            assert client.get(f"/api/sessions/{name}").status_code in (404, 400)

    def test_explain_endpoint_matches_the_module(self, client):
        assert client.get("/api/explain").json() == json.loads(
            json.dumps(explain_payload())
        )

    def test_figures_are_listed_from_disk(self, client):
        data = client.get("/api/figures").json()
        keys = {g["key"] for g in data["groups"]}
        assert keys == {"outputs", "paper"}
        for group in data["groups"]:
            for fig in group["figures"]:
                assert fig["id"].startswith(group["key"] + "/")
                assert fig["size_bytes"] > 0

    def test_figure_paths_cannot_escape_their_root(self, client):
        for bad in ("paper/../../../etc/passwd", "outputs/../../README.md",
                    "nope/x.png", "paper/does-not-exist.png"):
            assert client.get(f"/api/figures/{bad}").status_code == 404

    def test_a_listed_figure_can_be_fetched(self, client):
        data = client.get("/api/figures").json()
        figs = [f for g in data["groups"] for f in g["figures"]]
        if not figs:
            pytest.skip("no figures on disk in this checkout")
        r = client.get(f"/api/figures/{figs[0]['id']}")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("image/")

    def test_udp_can_be_disabled_and_re_enabled(self, client):
        off = client.post("/api/udp", json={"enabled": False}).json()
        assert off["enabled"] is False
        on = client.post("/api/udp", json={"enabled": True, "port": 9123}).json()
        assert on["enabled"] is True and on["port"] == 9123
        client.post("/api/udp", json={"enabled": False})


class TestWebSocketDataPlane:
    """The data plane: a real JPEG in, a complete frame result out."""

    def _jpeg(self):
        cv2 = pytest.importorskip("cv2")
        frame = _load_sample_frame()
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        assert ok
        return buf.tobytes()

    def test_frame_in_result_out(self, client):
        pytest.importorskip("mediapipe")
        jpeg = self._jpeg()
        with client.websocket_connect("/ws/stream") as ws:
            hello = ws.receive_json()
            assert hello["type"] == "hello"

            ws.send_bytes(jpeg)
            msg = ws.receive_json()
            assert msg["type"] == "frame"
            assert msg["source"] == "browser"
            assert msg["detected"] is True
            assert msg["landmarks"], "overlay landmarks must be forwarded"
            assert set(msg["filters"]) == set(FILTER_TYPES)
            assert msg["timings"]["pose_ms"] > 0
            assert "metrics" in msg and "status" in msg

    def test_undecodable_payload_is_reported_not_crashed(self, client):
        with client.websocket_connect("/ws/stream") as ws:
            ws.receive_json()
            ws.send_bytes(b"this is not a jpeg")
            msg = ws.receive_json()
            assert msg["type"] == "error"
            assert "decoded" in msg["message"]

    def test_ping_is_answered(self, client):
        with client.websocket_connect("/ws/stream") as ws:
            ws.receive_json()
            ws.send_json({"type": "ping"})
            assert ws.receive_json()["type"] == "pong"

    def test_round_trip_samples_are_recorded(self, client):
        from webapp import server as server_module
        with client.websocket_connect("/ws/stream") as ws:
            ws.receive_json()
            ws.send_json({"type": "rtt", "ms": 42.0})
            ws.send_json({"type": "ping"})
            ws.receive_json()
        assert server_module.state.rtt_summary()["n"] >= 1


# ---------------------------------------------------------------------------
# regressions
# ---------------------------------------------------------------------------

class TestAngleSpaceInPipeline:
    def test_angles_are_computed_in_metric_world_space(self, pipeline):
        trace = pipeline.process_frame(_load_sample_frame()).trace
        assert trace["coordinate_space"]["id"] == "world"
        assert trace["coordinate_space"]["unit"] == "m"
        # An adult upper arm is roughly 0.25-0.35 m; normalised image
        # coordinates would give a frame-size-dependent fraction instead.
        assert 0.15 < trace["sides"]["right"]["segment_lengths"]["upper_arm"] < 0.45


class TestUdpSender:
    @staticmethod
    def _receiver():
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("127.0.0.1", 0))
        sock.settimeout(1.0)
        return sock

    def test_nothing_is_sent_before_the_first_measurement(self):
        import time
        from src.processing.angle_solver import BilateralArmAngles
        from src.streaming.udp_streamer import UdpAngleSender

        rx = self._receiver()
        tx = UdpAngleSender(port=rx.getsockname()[1], hz=100, bilateral=True, verbose=False)
        tx.start()
        try:
            time.sleep(0.15)
            assert tx.packets_sent == 0
            assert tx.wire_state()["awaiting_data"] is True
            tx.update_bilateral(BilateralArmAngles(right=_angles(flex=12.5), left=None))
            data, _ = rx.recvfrom(256)
            assert data.decode().startswith("B,12.50,")
        finally:
            tx.stop()
            rx.close()

    def test_a_send_error_does_not_stop_the_stream(self):
        import time
        from src.streaming.udp_streamer import UdpAngleSender

        class FlakySocket:
            calls = 0

            def sendto(self, msg, addr):
                FlakySocket.calls += 1
                if FlakySocket.calls == 1:
                    raise OSError("network is unreachable")
                return len(msg)

            def close(self):
                pass

        tx = UdpAngleSender(hz=200, verbose=False)
        tx._sock.close()
        tx._sock = FlakySocket()
        tx.update(_angles(flex=5.0))
        tx.start()
        try:
            time.sleep(0.2)
            assert tx.running
            assert tx.send_errors == 1
            assert tx.packets_sent > 1
            assert "unreachable" in tx.wire_state()["last_error"]
        finally:
            tx.stop()


class TestCalibrationPersistence:
    def test_saved_calibration_is_reloaded_on_restart(self, tmp_path):
        mgr = TestCalibrationGuards()._run(forward_flexion=80.0)
        mgr.save(tmp_path / "calibration_right.json")

        pipe = LivePipeline(PipelineConfig(udp_enabled=False), tmp_path)
        try:
            assert pipe.calibration_state()["calibrated"]["right"] is True
            assert pipe._calib["right"].data.flexion.scale == pytest.approx(90.0 / 80.0)
        finally:
            pipe.close()


class TestTensorflowGuard:
    def test_unloadable_tensorflow_is_hidden_not_fatal(self, monkeypatch):
        import builtins
        from src.pose import tf_guard

        monkeypatch.setattr(tf_guard, "_probed", False)
        monkeypatch.setattr(tf_guard, "_load_error", None)
        saved = {m: sys.modules.pop(m) for m in list(sys.modules)
                 if m == "tensorflow" or m.startswith("tensorflow.")}
        real_import = builtins.__import__

        def blocked(name, *args, **kwargs):
            if name == "tensorflow" or name.startswith("tensorflow."):
                raise ImportError("DLL load failed: blocked by Application Control policy")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", blocked)
        try:
            with pytest.warns(RuntimeWarning, match="failed to load"):
                err = tf_guard.hide_unloadable_tensorflow()
            assert "Application Control" in err
            monkeypatch.setattr(builtins, "__import__", real_import)
            with pytest.raises(ModuleNotFoundError):
                import tensorflow  # noqa: F401
        finally:
            sys.modules.pop("tensorflow", None)
            sys.modules.update(saved)
