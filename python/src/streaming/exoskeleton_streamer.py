import sys, json, socket, threading, time, csv
from datetime import datetime, timezone
from pathlib import Path

JOINT_NAMES = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow_flexion"]


class ExoskeletonStreamer:
    """
    Broadcasts per-joint calibration packets over UDP (port 9001).
    Sends APPLY when confidence >= threshold, HOLD (last safe value) otherwise.
    Also logs to CSV for post-session analysis.
    """

    def __init__(self, host="127.0.0.1", port=9001, hz=30.0,
                 confidence_threshold=0.85, log_csv="outputs/exoskeleton_log.csv",
                 source="DeepFusionPose-GAN"):
        self._host      = host
        self._port      = port
        self._interval  = 1.0 / hz
        self._threshold = confidence_threshold
        self._source    = source
        self._sock      = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._lock      = threading.Lock()
        self._running   = False
        self._thread    = None
        self._angles_mean = [0.0] * 4
        self._angles_std  = [1.0] * 4
        self._frame       = 0
        self._prev_safe   = [0.0] * 4

        self._log_path = Path(log_csv)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._csv_file = open(self._log_path, "w", newline="", encoding="utf-8")
        self._csv_writer = csv.DictWriter(
            self._csv_file,
            fieldnames=["timestamp","frame","joint","angle","confidence","uncertainty","action","source"])
        self._csv_writer.writeheader()

    def update(self, angles_mean, angles_std, frame_idx):
        with self._lock:
            self._angles_mean = list(angles_mean)
            self._angles_std  = list(angles_std)
            self._frame       = frame_idx

    def start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        self._sock.close()
        self._csv_file.close()

    def __enter__(self): self.start(); return self
    def __exit__(self, *_): self.stop()

    def _loop(self):
        target = time.perf_counter()
        while self._running:
            with self._lock:
                means = self._angles_mean[:]
                stds  = self._angles_std[:]
                frame = self._frame

            ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
            for i, joint in enumerate(JOINT_NAMES):
                angle      = means[i]
                uncertainty = stds[i]
                confidence  = round(1.0 / (1.0 + uncertainty), 4)
                if confidence >= self._threshold:
                    action = "APPLY"
                    self._prev_safe[i] = angle
                else:
                    action = "HOLD"
                    angle  = self._prev_safe[i]

                packet = {"frame": frame, "timestamp": ts, "joint": joint,
                          "angle": round(angle, 3), "confidence": confidence,
                          "uncertainty": round(uncertainty, 4),
                          "source": self._source, "action": action}
                try:
                    self._sock.sendto((json.dumps(packet) + "\n").encode(),
                                      (self._host, self._port))
                except OSError:
                    self._running = False
                    return
                self._csv_writer.writerow({**packet})

            self._csv_file.flush()
            target += self._interval
            sleep = target - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            else:
                target = time.perf_counter()
