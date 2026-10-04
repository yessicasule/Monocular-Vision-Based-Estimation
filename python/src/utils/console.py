"""Console encoding helpers."""

from __future__ import annotations

import sys


def enable_utf8_stdout() -> None:
    """Force UTF-8 on stdout/stderr.

    The reports print degree signs, arrows and box-drawing characters. On
    Windows the default console encoding is cp1252, which raises
    UnicodeEncodeError on the first such character.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass
