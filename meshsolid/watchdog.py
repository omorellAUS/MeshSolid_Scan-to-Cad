"""Engine watchdog: a separate process that enforces a wall-clock budget on an engine run.

A timer thread or signal handler inside the engine cannot fire while OpenCASCADE runs a long C++ call
(the Python interpreter does not get control back), so the budget is kept by a child process: when it
expires and the run has not finished, the watchdog marks the checkpoint report (written at every stage)
as status "timeout" with the stage it was in, then kills the engine.  A finished run (any status other
than the checkpoint's "incomplete") is never touched.  Works on Windows and POSIX.

    python -m meshsolid.watchdog <engine pid> <budget s> <report path>
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time


def _alive(pid):
    if os.name == "nt":
        import ctypes
        k = ctypes.windll.kernel32
        h = k.OpenProcess(0x1000, False, pid)          # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        k.GetExitCodeProcess(h, ctypes.byref(code)); k.CloseHandle(h)
        return code.value == 259                       # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:                                               # a zombie (exited, not yet reaped) is not running
        with open(f"/proc/{pid}/stat") as f:
            return f.read().split(")")[-1].split()[0] != "Z"
    except OSError:
        return True


def _kill(pid):
    if os.name == "nt":
        subprocess.call(["taskkill", "/F", "/T", "/PID", str(pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        import signal
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass


def mark_timeout(report_path, budget):
    """Rewrite a checkpoint report as a timeout.  Returns True when it did."""
    try:
        with open(report_path, encoding="utf-8") as f:
            rep = json.load(f)
    except (OSError, ValueError):
        rep = {"tool": "meshsolid", "stages": {}}
    if rep.get("status") not in (None, "incomplete"):
        return False                                   # the run finished and wrote its own report
    stage = rep.get("last_stage")
    rep["status"] = "timeout"
    rep["end_reason"] = "timed_out"
    rep["failure"] = f"time budget of {budget:g} s exceeded during stage '{stage}'; the engine was stopped. No STEP written."
    rep["timeout"] = {"budget_s": budget, "stage": stage, "by": "watchdog"}
    tmp = str(report_path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2)
    os.replace(tmp, report_path)
    return True


def main(argv=None):
    pid, budget, report = argv or sys.argv[1:4]
    pid, budget = int(pid), float(budget)
    t_end = time.time() + budget
    while time.time() < t_end:
        if not _alive(pid):
            return 0
        time.sleep(min(1.0, max(0.05, t_end - time.time())))
    if not _alive(pid):
        return 0
    if mark_timeout(report, budget):      # else the run is already writing its final report
        _kill(pid)
    print(f"watchdog: budget {budget:g} s exceeded, engine {pid} stopped", file=sys.stderr, flush=True)
    return 0


def start(budget_s, report_path):
    """Start the watchdog for this process; returns the Popen (terminate it when the run ends)."""
    if not budget_s or budget_s <= 0 or not report_path:
        return None
    kw = {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)} if os.name == "nt" else {"start_new_session": True}
    return subprocess.Popen([sys.executable, "-m", "meshsolid.watchdog", str(os.getpid()), str(budget_s), str(report_path)],
                            stdin=subprocess.DEVNULL, **kw)


if __name__ == "__main__":
    sys.exit(main())
