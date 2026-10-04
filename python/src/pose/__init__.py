"""
src/pose/__init__.py
Convenience factory for loading pose estimators by name.
"""

from .base import PoseEstimator, Landmark, N_LANDMARKS, angle_landmarks, to_pixel_space
from .mediapipe_runner import MediaPipeRunner


def load_estimator(name: str, **kwargs) -> PoseEstimator:
    """
    Load a pose estimator by framework name.

    Parameters
    ----------
    name : str
        One of: "mediapipe", "movenet_lightning", "movenet_thunder", "posenet"
    **kwargs
        Forwarded to the runner's __init__().

    Returns
    -------
    PoseEstimator
    """
    name_lower = name.lower()

    if name_lower == "mediapipe":
        return MediaPipeRunner(**kwargs)

    if name_lower in ("movenet_lightning", "movenet_thunder"):
        from .movenet_runner import MoveNetRunner
        variant = "thunder" if "thunder" in name_lower else "lightning"
        return MoveNetRunner(variant=variant, **kwargs)

    if name_lower == "posenet":
        from .posenet_runner import PoseNetRunner
        return PoseNetRunner(**kwargs)

    raise ValueError(
        f"Unknown framework '{name}'. "
        "Choose from: mediapipe, movenet_lightning, movenet_thunder, posenet"
    )
