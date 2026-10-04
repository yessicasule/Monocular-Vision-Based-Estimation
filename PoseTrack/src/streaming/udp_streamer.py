"""
udp_streamer.py — Thread-safe UDP Angle Transmission
=====================================================

Transmits arm joint angles over UDP at a fixed rate to a Unity receiver.

Packet Formats
--------------
Single-arm (legacy, right arm):
    S,<shoulder_flex>,<shoulder_abd>,<shoulder_rot>,<elbow_flex>\\n

Bilateral (both arms, right side first):
    B,<r_flex>,<r_abd>,<r_rot>,<r_elbow>,<l_flex>,<l_abd>,<l_rot>,<l_elbow>\\n

All values are in degrees, formatted to two decimal places.
Each packet is terminated with a newline character as a frame delimiter.

Example packets:
    S,45.23,-12.10,8.75,90.00\\n
    B,45.23,-12.10,8.75,90.00,12.00,3.40,-1.20,15.55\\n

The 'S' prefix identifies a single-arm pose packet; 'B' a bilateral packet.
In bilateral mode an untracked side repeats its last known values.
The fixed-rate loop uses perf_counter timing to avoid clock drift,
sending at exactly the configured Hz without accumulating latency.

Threading Model
---------------
- A background daemon thread runs the fixed-rate send loop.
- The main thread updates angles via update_angles(), protected by a lock.
- On each send iteration, angles are snapshot-copied under the lock
  so the send call itself (which may block briefly on the OS) is lock-free.
"""

from __future__ import annotations

import socket
import threading
import time
from collections import deque

from ..processing.angle_solver import ArmAngles, BilateralArmAngles


class UdpAngleSender:
    """
    Thread-safe, fixed-rate UDP sender for arm joint angles.

    Parameters
    ----------
    host : str
        Destination IP address (default: localhost).
    port : int
        Destination UDP port (default: 9000, must match Unity receiver).
    hz : float
        Target transmission rate in Hz (default: 30).
    verbose : bool
        Print a periodic TX status line to the terminal confirming that
        packets are being sent (default: True).
    log_interval_s : float
        Seconds between TX status lines when verbose (default: 1.0).
    """

    def __init__(
        self,
        host: str  = "127.0.0.1",
        port: int  = 9000,
        hz:   float = 30.0,
        bilateral: bool = False,
        verbose: bool = True,
        log_interval_s: float = 1.0,
    ) -> None:
        self._addr     = (host, port)
        self._interval = 1.0 / max(hz, 1.0)
        self._sock     = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._lock     = threading.Lock()
        self._running  = False
        self._thread: threading.Thread | None = None
        self._bilateral = bilateral

        # TX visibility
        self._verbose        = verbose
        self._log_interval   = max(log_interval_s, 0.1)
        self._packets_sent   = 0
        self._send_errors    = 0
        self._last_log_t     = 0.0
        self._last_log_count = 0

        # Wire inspection: the exact text of the most recently transmitted
        # packet, its send time, and a short rolling history. Consumed by the
        # web dashboard's packet inspector so the operator can see the bytes
        # Unity actually receives.
        self._last_packet    = ""
        self._last_packet_t  = 0.0
        self._packet_history = deque(maxlen=32)
        self._last_error     = ""

        # Nothing is transmitted until a measurement exists: zeros sent before
        # the first tracked frame would drive the avatar with a pose nobody
        # measured.
        self._has_data = False

        # Current angle values (degrees): right arm
        self._flex: float = 0.0
        self._abd:  float = 0.0
        self._rot:  float = 0.0
        self._elb:  float = 0.0
        # Left arm (bilateral mode only)
        self._l_flex: float = 0.0
        self._l_abd:  float = 0.0
        self._l_rot:  float = 0.0
        self._l_elb:  float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background sending thread."""
        if self._running:
            return
        self._running = True
        self._last_log_t = time.perf_counter()
        self._thread  = threading.Thread(target=self._loop, daemon=True, name="UdpSender")
        self._thread.start()

    def stop(self) -> None:
        """Stop the background thread and close the socket."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        try:
            self._sock.close()
        except OSError:
            pass

    def update(self, angles: ArmAngles) -> None:
        """
        Update the angles to be sent on the next packet.

        This is the primary way to push new angle data into the sender.
        Thread-safe.

        Parameters
        ----------
        angles : ArmAngles
            Latest computed (and filtered) arm joint angles.
        """
        with self._lock:
            self._flex = angles.shoulder_flexion
            self._abd  = angles.shoulder_abduction
            self._rot  = angles.shoulder_rotation
            self._elb  = angles.elbow_flexion
            self._has_data = True

    def update_bilateral(self, bilateral: BilateralArmAngles) -> None:
        """
        Update both arms' angles for the next bilateral packet.

        An untracked side (None) keeps its previous values so the avatar
        holds its last pose rather than snapping to zero. Thread-safe.
        """
        with self._lock:
            if bilateral.right is not None:
                self._flex = bilateral.right.shoulder_flexion
                self._abd  = bilateral.right.shoulder_abduction
                self._rot  = bilateral.right.shoulder_rotation
                self._elb  = bilateral.right.elbow_flexion
            if bilateral.left is not None:
                self._l_flex = bilateral.left.shoulder_flexion
                self._l_abd  = bilateral.left.shoulder_abduction
                self._l_rot  = bilateral.left.shoulder_rotation
                self._l_elb  = bilateral.left.elbow_flexion
            if bilateral.right is not None or bilateral.left is not None:
                self._has_data = True

    @property
    def packets_sent(self) -> int:
        """Total packets transmitted without an OS-level send error."""
        return self._packets_sent

    @property
    def send_errors(self) -> int:
        """Number of packets that failed at the OS send call."""
        return self._send_errors

    @property
    def destination(self) -> tuple[str, int]:
        """(host, port) the packets are addressed to."""
        return self._addr

    @property
    def target_hz(self) -> float:
        """Configured transmission rate in Hz."""
        return 1.0 / self._interval

    @property
    def running(self) -> bool:
        """True while the background send loop is active."""
        return self._running

    def wire_state(self) -> dict:
        """
        Snapshot of the transmit side for UI/telemetry.

        Returns the destination, packet counters and the literal text of the
        most recent packets, so a consumer can verify byte-for-byte what the
        Unity receiver is being sent.
        """
        with self._lock:
            history = list(self._packet_history)
        return {
            "host":          self._addr[0],
            "port":          self._addr[1],
            "running":       self._running,
            "target_hz":     round(self.target_hz, 2),
            "packets_sent":  self._packets_sent,
            "send_errors":   self._send_errors,
            "bilateral":     self._bilateral,
            "awaiting_data": not self._has_data,
            "last_error":    self._last_error,
            "last_packet":   self._last_packet,
            "last_packet_t": self._last_packet_t,
            "history":       [{"t": t, "packet": p} for t, p in history[-8:]],
        }

    def send_now(self, angles: ArmAngles) -> None:
        """
        Send a single packet immediately (bypasses the background thread).
        Useful for one-off sends in the data generator.
        """
        msg = self._format(
            angles.shoulder_flexion,
            angles.shoulder_abduction,
            angles.shoulder_rotation,
            angles.elbow_flexion,
        )
        try:
            self._sock.sendto(msg, self._addr)
        except OSError:
            pass

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        """Fixed-rate send loop — runs on the background thread."""
        target = time.perf_counter()
        last_error_print = 0.0
        while self._running:
            with self._lock:
                has_data = self._has_data
                if self._bilateral:
                    msg = self._format_bilateral(
                        self._flex,   self._abd,   self._rot,   self._elb,
                        self._l_flex, self._l_abd, self._l_rot, self._l_elb,
                    )
                else:
                    msg = self._format(self._flex, self._abd, self._rot, self._elb)
            if has_data:
                try:
                    self._sock.sendto(msg, self._addr)
                    self._packets_sent += 1
                    text = msg.decode("utf-8").strip()
                    self._last_packet   = text
                    self._last_packet_t = time.time()
                    self._packet_history.append((self._last_packet_t, text))
                except OSError as e:
                    # Usually transient (network down, host unreachable):
                    # count it, report at most once a second, keep streaming.
                    self._send_errors += 1
                    self._last_error = f"{type(e).__name__}: {e}"
                    now = time.perf_counter()
                    if now - last_error_print >= 1.0:
                        print(f"[UDP TX ERROR] send to {self._addr[0]}:{self._addr[1]} "
                              f"failed: {e} (errors so far: {self._send_errors})")
                        last_error_print = now

            if self._verbose and has_data:
                now = time.perf_counter()
                elapsed = now - self._last_log_t
                if elapsed >= self._log_interval:
                    rate = (self._packets_sent - self._last_log_count) / elapsed
                    print(f"[UDP TX OK] -> {self._addr[0]}:{self._addr[1]}  |  "
                          f"{rate:5.1f} pkt/s  |  total {self._packets_sent}  |  "
                          f"{msg.decode('utf-8').strip()}")
                    self._last_log_t     = now
                    self._last_log_count = self._packets_sent

            target += self._interval
            sleep_s = target - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                # Running behind schedule — reset target to avoid spiral
                target = time.perf_counter()

    @staticmethod
    def _format(flex: float, abd: float, rot: float, elb: float) -> bytes:
        """
        Format angles into the packet string and encode as UTF-8 bytes.

        Packet format: S,<flex>,<abd>,<rot>,<elb>\\n
        """
        return f"S,{flex:.2f},{abd:.2f},{rot:.2f},{elb:.2f}\n".encode("utf-8")

    @staticmethod
    def _format_bilateral(
        r_flex: float, r_abd: float, r_rot: float, r_elb: float,
        l_flex: float, l_abd: float, l_rot: float, l_elb: float,
    ) -> bytes:
        """
        Format both arms' angles into a bilateral packet.

        Packet format: B,<r_flex>,<r_abd>,<r_rot>,<r_elb>,<l_flex>,<l_abd>,<l_rot>,<l_elb>\\n
        """
        return (
            f"B,{r_flex:.2f},{r_abd:.2f},{r_rot:.2f},{r_elb:.2f},"
            f"{l_flex:.2f},{l_abd:.2f},{l_rot:.2f},{l_elb:.2f}\n"
        ).encode("utf-8")

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
