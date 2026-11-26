"""Heartbeat file that keeps moving during a long stage.

A thread inside the engine cannot run while OpenCASCADE (or any other long C++ call)
holds the Python interpreter, which is why a heartbeat written only at stage changes
— or from a thread — stops updating on Test Net. A child process does not need the
interpreter, so it rewrites the heartbeat on its own interval for as long as the
engine process is alive.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


def heartbeat_path(report_path) -> str:
    return str(Path(report_path).with_suffix(".heartbeat.json"))


def stage_path(report_path) -> str:
    return str(Path(report_path).with_suffix(".stage"))


def _alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        k = ctypes.windll.kernel32
        h = k.OpenProcess(0x1000, False, pid)
        if not h:
            return False
        code = ctypes.c_ulong()
        k.GetExitCodeProcess(h, ctypes.byref(code))
        k.CloseHandle(h)
        return code.value == 259
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().split(")")[-1].split()[0] != "Z"
    except OSError:
        return True


def set_stage(report_path, name: str) -> None:
    """Record the stage the engine just entered. The child process copies this name
    into the heartbeat and refreshes the timestamp until the next stage."""
    if not report_path:
        return
    path = stage_path(report_path)
    tmp = path + ".tmp"
    try:
        Path(tmp).write_text(str(name), encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        pass


def start(report_path, interval: float = 5.0, t0: float | None = None):
    """Start the heartbeat child. Returns the Popen, or None when disabled."""
    if not report_path or not interval or float(interval) <= 0:
        return None
    set_stage(report_path, "starting")
    cmd = [sys.executable, "-m", "meshsolid.heartbeat",
           heartbeat_path(report_path), stage_path(report_path),
           str(float(interval)), str(os.getpid()), str(float(t0 if t0 is not None else time.time()))]
    kw = {"stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
    if os.name == "nt":
        kw["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    else:
        kw["start_new_session"] = True
    return subprocess.Popen(cmd, **kw)


def stop(proc) -> None:
    if proc is None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=2)
    except Exception:  # noqa: BLE001
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass


def main(argv=None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    if len(args) < 5:
        print("usage: python -m meshsolid.heartbeat HEARTBEAT STAGE INTERVAL PPID T0", file=sys.stderr)
        return 2
    hp, stage, interval_s, ppid_s, t0_s = args[:5]
    interval = max(0.02, float(interval_s))
    ppid = int(ppid_s)
    t0 = float(t0_s)
    while _alive(ppid):
        name = "unknown"
        try:
            name = Path(stage).read_text(encoding="utf-8").strip() or name
        except OSError:
            pass
        payload = {"stage": name, "ts": time.time(), "elapsed_s": round(time.time() - t0, 2)}
        tmp = hp + ".tmp"
        try:
            Path(tmp).write_text(json.dumps(payload), encoding="utf-8")
            os.replace(tmp, hp)
        except OSError:
            pass
        end = time.time() + interval
        while time.time() < end:
            if not _alive(ppid):
                return 0
            time.sleep(min(0.05, max(0.0, end - time.time())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
