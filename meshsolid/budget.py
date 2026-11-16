"""Run-wide wall-clock budget shared by the stages (set by pipeline.run from Params.time_budget_s).

Stages that loop over OpenCASCADE calls (local trim, set-back corner fill, cell decomposition) take
`stage_budget(cap)`: their own cap, shortened to what the run has left minus a reserve for the deviation
check and the report, so they stop with a reason instead of the watchdog killing the run."""
from __future__ import annotations

import os
import time

DEADLINE = None


def set_deadline(t):
    global DEADLINE
    DEADLINE = t


def time_left():
    return float("inf") if DEADLINE is None else DEADLINE - time.time()


def stage_budget(cap_s, reserve_frac=0.2, min_s=5.0):
    """Seconds this stage may use: min(cap, left - reserve) with reserve = max(30 s, reserve_frac x left)."""
    left = time_left()
    if left == float("inf"):
        return float(cap_s)
    return max(min_s, min(float(cap_s), left - max(30.0, reserve_frac * left)))


def env_cap(name, default):
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return float(default)
