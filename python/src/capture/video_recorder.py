import cv2
import time
from pathlib import Path

class VideoRecorder:
    def __init__(self, output_path: str, fps: float = 30.0, resolution: tuple = (640, 480)):
        self.output_path = output_path
        self.fps = fps
        self.resolution = resolution
        self.writer = None

    def start(self):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        self.writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, self.resolution)
        print(f"Started recording video to {self.output_path}")

    def write_frame(self, frame):
        if self.writer is not None:
            if frame.shape[:2] != (self.resolution[1], self.resolution[0]):
                frame = cv2.resize(frame, self.resolution)
            self.writer.write(frame)

    def stop(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            print(f"Stopped recording. Saved to {self.output_path}")
