"""
sources.py — Server-side frame sources
=======================================

The dashboard supports three ways of feeding real frames into the pipeline:

``browser``
    The page captures the operator's webcam with ``getUserMedia`` and pushes
    JPEG frames up the WebSocket. This is the path used when the browser and
    the camera are on the same machine and the server may be elsewhere.

``camera``
    The server opens a local camera with OpenCV. Used when the Python process
    runs on the machine the webcam is attached to.

``file``
    The server replays a recorded video file frame by frame. Replaying a
    recording makes an evaluation run exactly reproducible: the same frames
    produce the same angles, so filter settings and frameworks can be
    compared on identical input.

All three deliver genuine camera imagery — there is no synthetic frame
generator here, and a source that cannot be opened reports the failure
rather than substituting placeholder content.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import cv2
import numpy as np


class SourceError(RuntimeError):
    """Raised when a camera or video file cannot be opened or read."""


class ServerFrameSource:
    """
    Background reader that pulls frames from a local camera or video file
    and hands each one to a callback at (at most) the requested rate.

    The callback runs on the reader thread; it is expected to do the pipeline
    work and is therefore the rate limiter for a camera source. For a file
    source the reader paces itself to the file's own frame rate so replay
    speed matches the recording, unless ``as_fast_as_possible`` is set.
    """

    def __init__(
        self,
        mode: str,
        camera_index: int = 0,
        path: str | Path | None = None,
        width: int = 640,
        height: int = 480,
        loop: bool = True,
        as_fast_as_possible: bool = False,
    ) -> None:
        if mode not in ("camera", "file"):
            raise ValueError("mode must be 'camera' or 'file'")
        self.mode = mode
        self.camera_index = int(camera_index)
        self.path = Path(path) if path else None
        self.width = width
        self.height = height
        self.loop = loop
        self.as_fast_as_possible = as_fast_as_possible

        self._cap: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._error: str | None = None
        self._frames_read = 0
        self._source_fps = 0.0
        self._total_frames = 0

    # ------------------------------------------------------------------

    def open(self) -> None:
        if self.mode == "camera":
            cap = cv2.VideoCapture(self.camera_index)
            if not cap.isOpened():
                raise SourceError(
                    f"Camera index {self.camera_index} could not be opened. "
                    "Check that a camera is connected and not in use by another "
                    "application, or use the browser camera source instead."
                )
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            if self.path is None or not self.path.exists():
                raise SourceError(f"Video file not found: {self.path}")
            cap = cv2.VideoCapture(str(self.path))
            if not cap.isOpened():
                raise SourceError(f"Video file could not be decoded: {self.path}")
            self._total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        self._source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self._cap = cap

    def start(self, on_frame) -> None:
        """Open the source and begin delivering frames to ``on_frame``."""
        if self._running:
            return
        self.open()
        self._running = True
        self._error = None
        self._thread = threading.Thread(
            target=self._loop, args=(on_frame,), daemon=True, name="FrameSource"
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    # ------------------------------------------------------------------

    def _loop(self, on_frame) -> None:
        interval = 0.0
        if self.mode == "file" and not self.as_fast_as_possible and self._source_fps > 0:
            interval = 1.0 / self._source_fps
        next_t = time.perf_counter()

        while self._running and self._cap is not None:
            t0 = time.perf_counter()
            ok, frame = self._cap.read()
            read_ms = (time.perf_counter() - t0) * 1000.0

            if not ok:
                if self.mode == "file" and self.loop:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                self._error = "End of stream." if self.mode == "file" else \
                              "Camera stopped delivering frames."
                self._running = False
                break

            self._frames_read += 1
            try:
                on_frame(frame, read_ms)
            except Exception as exc:                    # keep the reader alive
                self._error = f"{type(exc).__name__}: {exc}"

            if interval:
                next_t += interval
                sleep_s = next_t - time.perf_counter()
                if sleep_s > 0:
                    time.sleep(sleep_s)
                else:
                    next_t = time.perf_counter()

    # ------------------------------------------------------------------

    def state(self) -> dict:
        return {
            "mode": self.mode,
            "camera_index": self.camera_index,
            "path": str(self.path) if self.path else None,
            "running": self._running,
            "error": self._error,
            "frames_read": self._frames_read,
            "source_fps": round(self._source_fps, 2),
            "total_frames": self._total_frames,
            "loop": self.loop,
        }


def probe_cameras(max_index: int = 4) -> list[dict]:
    """
    Report which local camera indices can actually be opened.

    Each entry is a real probe result, so an empty list means the machine
    running the server has no usable camera — which is the expected outcome
    on a headless host, where the browser source should be used instead.
    """
    found = []
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx)
        try:
            if cap.isOpened():
                ok, frame = cap.read()
                if ok and frame is not None:
                    found.append({
                        "index": idx,
                        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                        "fps": round(float(cap.get(cv2.CAP_PROP_FPS) or 0.0), 2),
                    })
        finally:
            cap.release()
    return found


def encode_jpeg(frame_bgr: np.ndarray, quality: int = 80) -> bytes:
    """Encode a BGR frame as JPEG bytes for the MJPEG preview stream."""
    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise SourceError("JPEG encoding failed.")
    return buf.tobytes()
