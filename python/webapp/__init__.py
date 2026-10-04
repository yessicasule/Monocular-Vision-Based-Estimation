"""
webapp — Browser front-end and HTTP/WebSocket back-end for the MonoArm
monocular arm-angle pipeline.

The package wraps the existing processing modules (pose estimation, angle
solver, filter bank, calibration, UDP streaming, CSV logging) in a live
service so that the whole chain

    camera frame → landmarks → joint angles → filtering → calibration
                 → UDP packet to Unity → CSV log

can be driven, inspected and explained from a web page, while the very same
angles are transmitted to the Unity avatar over UDP.

Modules
-------
pipeline : frame-level orchestration and per-stage instrumentation
metrics  : rolling FPS / latency / jitter statistics
server   : FastAPI application (REST control plane + WebSocket data plane)
"""

__all__ = ["pipeline", "metrics", "server"]
