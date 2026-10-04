"""
tf_guard.py — Keep an unloadable TensorFlow from taking MediaPipe down with it
==============================================================================

MediaPipe's Tasks package imports TensorFlow purely for API-doc decorators
(``mediapipe/tasks/python/core/optional_dependencies.py``) and only tolerates
``ModuleNotFoundError``. When TensorFlow is *installed but cannot load* — a
DLL blocked by Windows Smart App Control / WDAC, a CPU without AVX, a
half-upgraded wheel — the import raises ``ImportError`` (or worse) instead, and
``import mediapipe`` fails with an unrelated-looking
``NameError: name 'audio_classifier' is not defined``.

:func:`hide_unloadable_tensorflow` probes the import once. If it fails for any
reason other than TensorFlow being absent, the half-imported modules are
purged and ``sys.modules["tensorflow"]`` is set to ``None`` so every later
``import tensorflow`` raises ``ModuleNotFoundError`` — the case MediaPipe and
the PoseNet runner already handle. MoveNet, which genuinely needs TensorFlow,
reports the original load error via :func:`tensorflow_load_error`.
"""

from __future__ import annotations

import sys
import warnings

_load_error: str | None = None
_probed = False


def hide_unloadable_tensorflow() -> str | None:
    """Probe TensorFlow once; hide it if installed but unloadable.

    Returns the load error text when TensorFlow was hidden, else ``None``.
    """
    global _probed, _load_error
    if _probed:
        return _load_error
    _probed = True

    if "tensorflow" in sys.modules:          # already imported (or already hidden)
        return None
    try:
        import tensorflow  # noqa: F401
    except ModuleNotFoundError:
        return None
    except Exception as exc:                 # ImportError from a blocked DLL, etc.
        for name in [m for m in sys.modules if m == "tensorflow" or m.startswith("tensorflow.")]:
            del sys.modules[name]
        sys.modules["tensorflow"] = None     # makes `import tensorflow` raise ModuleNotFoundError
        _load_error = f"{type(exc).__name__}: {exc}"
        warnings.warn(
            "TensorFlow is installed but failed to load and has been disabled for "
            f"this process ({_load_error}). MediaPipe and PoseNet are unaffected; "
            "MoveNet is unavailable.",
            RuntimeWarning,
            stacklevel=2,
        )
    return _load_error


def tensorflow_load_error() -> str | None:
    """The reason TensorFlow was hidden, or ``None`` if it was not."""
    return _load_error
