"""Make script output safe on the Windows console.

Windows terminals often default to the cp1252 code page, which has no rupee
sign. Printing a formatted price then raises UnicodeEncodeError and kills the
training run - after the model has been fitted but before anything is saved.
Reconfiguring stdout to UTF-8 keeps the rupee symbol everywhere and costs
nothing on macOS/Linux, which are UTF-8 already.
"""

from __future__ import annotations

import sys


def enable_utf8_stdout() -> None:
    """Switch stdout/stderr to UTF-8, replacing anything unencodable."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                # Redirected to something that cannot be reconfigured; the
                # errors="replace" fallback below is not available, but losing
                # a currency glyph must never abort a training run.
                pass
