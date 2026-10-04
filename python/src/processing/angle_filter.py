"""
angle_filter.py — Temporal Filtering for Joint Angle Signals
=============================================================

Implements three temporal filters for reducing keypoint-noise-induced
jitter in joint angle time series while preserving motion responsiveness.

Filters Implemented
-------------------
1. MovingAverageFilter
   Simple windowed mean. O(1) update with a circular deque.
   Introduces a phase delay of (window_size − 1) / 2 frames.

2. SavitzkyGolayFilter
   Fits a polynomial of degree p to a window of n samples and evaluates
   it at the most recent sample. Preserves higher-order signal features
   (peak positions, curvature) better than a plain moving average.
   Requires scipy. Falls back to raw value while buffer fills.

3. KalmanFilter2State  (recommended default)
   Two-state discrete Kalman filter modelling:
       x = [angle, angular_velocity]^T
   State transition (constant-velocity model):
       x_{k|k-1} = F x_{k-1}   where F = [[1, dt], [0, 1]]
   Measurement model (angle observed directly):
       z_k = H x_k              where H = [1, 0]
   Noise covariances (tunable):
       Q — process noise covariance (models unexpected accelerations)
       R — measurement noise variance (keypoint detection noise)
   Optimal Kalman gain:
       K = P_{k|k-1} H^T (H P_{k|k-1} H^T + R)^{-1}
   Update:
       x_k = x_{k|k-1} + K(z_k − H x_{k|k-1})
       P_k = (I − KH) P_{k|k-1}

   The 2-state model (angle + velocity) adapts better to motion than the
   1-state scalar Kalman used in the original codebase: it anticipates
   continuing motion rather than treating each measurement independently,
   reducing lag during fast movements.

AngleFilterBank
---------------
Wraps four instances of the chosen filter (one per DOF) and exposes a
single update() method taking/returning an ArmAngles dataclass.

References:
    Savitzky, A. & Golay, M.J.E. (1964). Smoothing and differentiation of
    data by simplified least squares procedures. Anal. Chem. 36(8):1627-1639.
    Kalman, R.E. (1960). A new approach to linear filtering. Trans. ASME
    J. Basic Eng. 82(1):35-45.
    Zarchan & Musoff (2009). Fundamentals of Kalman Filtering. AIAA.
"""

from __future__ import annotations

import numpy as np
from collections import deque

try:
    from scipy.signal import savgol_filter as _scipy_savgol
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False

from .angle_solver import ArmAngles, BilateralArmAngles


# ---------------------------------------------------------------------------
# Individual filter classes
# ---------------------------------------------------------------------------

class MovingAverageFilter:
    """
    Causal moving-average (boxcar) filter.

    y[n] = (1 / W) * sum_{k=0}^{W-1} x[n-k]

    Parameters
    ----------
    window_size : int
        Number of samples in the averaging window (W).  Larger values
        give smoother output at the cost of more lag.
    """

    def __init__(self, window_size: int = 7) -> None:
        self.window_size = window_size
        self._buf: deque[float] = deque(maxlen=window_size)

    def update(self, value: float) -> float:
        self._buf.append(value)
        return float(np.mean(self._buf))

    def reset(self) -> None:
        self._buf.clear()


class SavitzkyGolayFilter:
    """
    Causal Savitzky-Golay filter.

    Fits a polynomial of degree `polyorder` to the last `window_length`
    samples and evaluates it at the rightmost (most recent) point.

    Unlike a moving average, SG preserves peaks and curvature because the
    polynomial fit weights central samples more than edge samples.

    Falls back to the raw value until the buffer contains `window_length`
    samples, and falls back permanently if scipy is unavailable.

    Parameters
    ----------
    window_length : int
        Length of the filter window (must be odd). If even, incremented.
    polyorder : int
        Degree of the fitting polynomial. Must be < window_length.
    """

    def __init__(self, window_length: int = 11, polyorder: int = 3) -> None:
        if window_length % 2 == 0:
            window_length += 1
        if polyorder >= window_length:
            polyorder = window_length - 1
        self.window_length = window_length
        self.polyorder     = polyorder
        self._buf: deque[float] = deque(maxlen=window_length)

    def update(self, value: float) -> float:
        self._buf.append(value)
        if not _SCIPY_OK or len(self._buf) < self.window_length:
            return value
        filtered = _scipy_savgol(list(self._buf), self.window_length, self.polyorder)
        return float(filtered[-1])

    def reset(self) -> None:
        self._buf.clear()


class KalmanFilter2State:
    """
    Two-state discrete Kalman filter for a single angle channel.

    State vector:  x = [angle, angular_velocity]^T   (degrees, degrees/s)
    Dynamics:      x_{k} = F x_{k-1} + w_k
    Measurement:   z_k   = H x_k + v_k

    State transition matrix F (constant-velocity model):
        F = [[1, dt],
             [0,  1 ]]

    Measurement matrix H (we observe angle only):
        H = [[1, 0]]

    Process noise covariance Q:
        Models unexpected angular accelerations.
        Q = q * [[dt^4/4, dt^3/2],
                  [dt^3/2, dt^2  ]]
        where q = process_noise (spectral density).

    Measurement noise covariance R:
        R = [[measurement_noise]]
        (scalar; units: degrees^2)

    Parameters
    ----------
    process_noise : float
        Process noise spectral density q.  Higher → faster tracking,
        more noise passes through.  Lower → smoother but more lag.
    measurement_noise : float
        Variance of the angle measurement in degrees^2.  Set to the
        approximate variance of raw keypoint-derived angles during a
        static hold.
    dt : float
        Expected time step in seconds (1 / stream_hz).
    """

    def __init__(
        self,
        process_noise:     float = 0.01,
        measurement_noise: float = 1.5,
        dt:                float = 1.0 / 30.0,
    ) -> None:
        self._dt = dt
        self._q  = process_noise
        self._r  = measurement_noise

        # State: [angle, angular_velocity]
        self._x = np.zeros(2, dtype=np.float64)

        # State covariance (initialised with high uncertainty)
        self._P = np.eye(2, dtype=np.float64) * 1000.0

        # State transition
        self._F = np.array([[1.0, dt], [0.0, 1.0]], dtype=np.float64)

        # Measurement matrix
        self._H = np.array([[1.0, 0.0]], dtype=np.float64)

        # Measurement noise covariance (scalar wrapped in 1×1 matrix)
        self._R = np.array([[measurement_noise]], dtype=np.float64)

        self._init = False

    def _build_Q(self) -> np.ndarray:
        """
        Process noise covariance matrix (discretised continuous white noise).
        Q = q * [[dt^4/4, dt^3/2],
                  [dt^3/2, dt^2  ]]
        """
        dt = self._dt
        q  = self._q
        return q * np.array([
            [dt**4 / 4.0, dt**3 / 2.0],
            [dt**3 / 2.0, dt**2       ],
        ], dtype=np.float64)

    def update(self, value: float) -> float:
        """
        Kalman predict-update cycle.

        Predict
        -------
        x_{k|k-1} = F x_{k-1|k-1}
        P_{k|k-1} = F P_{k-1|k-1} F^T + Q

        Update
        ------
        Innovation:   y = z - H x_{k|k-1}
        Innovation covariance: S = H P_{k|k-1} H^T + R
        Kalman gain:  K = P_{k|k-1} H^T S^{-1}
        State update: x_{k|k} = x_{k|k-1} + K y
        Cov update:   P_{k|k} = (I - KH) P_{k|k-1}
        """
        if not self._init:
            self._x[0] = value
            self._x[1] = 0.0
            self._init = True
            return value

        Q = self._build_Q()

        # --- Predict ---
        x_pred = self._F @ self._x
        P_pred = self._F @ self._P @ self._F.T + Q

        # --- Update ---
        z   = np.array([[value]])
        y   = z - self._H @ x_pred.reshape(-1, 1)      # innovation
        S   = self._H @ P_pred @ self._H.T + self._R    # innovation cov
        K   = P_pred @ self._H.T @ np.linalg.inv(S)    # Kalman gain

        self._x = (x_pred.reshape(-1, 1) + K @ y).flatten()
        self._P = (np.eye(2) - K @ self._H) @ P_pred

        return float(self._x[0])

    def reset(self) -> None:
        self._x    = np.zeros(2, dtype=np.float64)
        self._P    = np.eye(2, dtype=np.float64) * 1000.0
        self._init = False


# ---------------------------------------------------------------------------
# Filter bank — one filter per DOF
# ---------------------------------------------------------------------------

class AngleFilterBank:
    """
    Per-DOF filter bank that wraps four filter instances.

    Applies the chosen filter independently to each of the four joint
    angle channels: shoulder_flexion, shoulder_abduction,
    shoulder_rotation, elbow_flexion.

    Parameters
    ----------
    filter_type : str
        One of "kalman" (default), "ma", "sg".
    stream_hz : float
        Expected update rate in Hz (used to set Kalman dt).
    **kwargs
        Forwarded to the filter constructor.
    """

    _TYPES = {"kalman", "ma", "sg"}

    def __init__(
        self,
        filter_type: str = "kalman",
        stream_hz:   float = 30.0,
        **kwargs,
    ) -> None:
        if filter_type not in self._TYPES:
            raise ValueError(f"filter_type must be one of {self._TYPES}")
        self.filter_type = filter_type
        self._filters = {
            name: self._make(filter_type, stream_hz, **kwargs)
            for name in ("flexion", "abduction", "rotation", "elbow")
        }

    @staticmethod
    def _make(filter_type: str, hz: float, **kw):
        if filter_type == "ma":
            return MovingAverageFilter(kw.get("window_size", 7))
        if filter_type == "sg":
            return SavitzkyGolayFilter(
                kw.get("window_length", 11),
                kw.get("polyorder", 3),
            )
        # default: kalman
        return KalmanFilter2State(
            process_noise=kw.get("process_noise", 0.01),
            measurement_noise=kw.get("measurement_noise", 1.5),
            dt=1.0 / hz,
        )

    def update(self, angles: ArmAngles) -> ArmAngles:
        """
        Filter all four angle channels and return a new ArmAngles.

        Parameters
        ----------
        angles : ArmAngles
            Raw (unfiltered) joint angles from angle_solver.compute_arm_angles().

        Returns
        -------
        ArmAngles
            Filtered joint angles. rotation_reliable is preserved from input.
        """
        return ArmAngles(
            shoulder_flexion   = self._filters["flexion"].update(angles.shoulder_flexion),
            shoulder_abduction = self._filters["abduction"].update(angles.shoulder_abduction),
            shoulder_rotation  = self._filters["rotation"].update(angles.shoulder_rotation),
            elbow_flexion      = self._filters["elbow"].update(angles.elbow_flexion),
            rotation_reliable  = angles.rotation_reliable,
            side               = angles.side,
        )

    def reset(self) -> None:
        """Reset all filters (e.g., after a tracking loss)."""
        for f in self._filters.values():
            f.reset()


class BilateralFilterBank:
    """
    Filter bank for bilateral (left + right) arm tracking.

    Wraps two independent AngleFilterBank instances — one per side — so
    each of the eight DOF channels has its own filter state.

    A side whose angles are None for a frame (arm occluded) keeps its
    filter state frozen and returns None for that side; filtering resumes
    from the previous state when the arm reappears.

    Parameters
    ----------
    filter_type : str
        One of "kalman" (default), "ma", "sg".  Applied to both sides.
    stream_hz : float
        Expected update rate in Hz.
    **kwargs
        Forwarded to the per-side AngleFilterBank constructors.
    """

    def __init__(
        self,
        filter_type: str = "kalman",
        stream_hz:   float = 30.0,
        **kwargs,
    ) -> None:
        self.filter_type = filter_type
        self._banks = {
            "right": AngleFilterBank(filter_type, stream_hz, **kwargs),
            "left":  AngleFilterBank(filter_type, stream_hz, **kwargs),
        }

    def update(self, bilateral: BilateralArmAngles) -> BilateralArmAngles:
        """Filter both sides and return a new BilateralArmAngles."""
        return BilateralArmAngles(
            right=self._banks["right"].update(bilateral.right)
                  if bilateral.right is not None else None,
            left =self._banks["left"].update(bilateral.left)
                  if bilateral.left is not None else None,
        )

    def reset(self, side: str | None = None) -> None:
        """Reset one side's filters, or both when side is None."""
        if side is None:
            for bank in self._banks.values():
                bank.reset()
        else:
            self._banks[side].reset()
