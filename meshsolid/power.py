"""Keep the PC awake while an engine job runs (1.0.2a).

A conversion or Metrology job can run for hours; if Windows goes to sleep mid-job the run stalls and the
wall-clock time budget (meshsolid.budget / watchdog use time.time(), so time asleep still counts) can run
out.  On Windows the job asks for ES_CONTINUOUS | ES_SYSTEM_REQUIRED (system sleep blocked; the display may
still turn off) and resets to ES_CONTINUOUS when the job ends, also on error.  No-op on other systems.
"""
from __future__ import annotations

import contextlib
import sys

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def _set(flags):
    """Call SetThreadExecutionState; returns the previous state (0 on failure) or None off Windows."""
    if not sys.platform.startswith("win"):
        return None
    try:
        import ctypes
        f = ctypes.windll.kernel32.SetThreadExecutionState
        f.restype = ctypes.c_uint
        f.argtypes = [ctypes.c_uint]
        return int(f(ctypes.c_uint(flags)))
    except Exception:  # noqa: BLE001  - never let power management break a job
        return 0


@contextlib.contextmanager
def keep_awake():
    """Block system sleep for the duration of the with-block (Windows); always released in finally."""
    prev = _set(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    try:
        yield prev
    finally:
        if prev is not None:
            _set(ES_CONTINUOUS)
