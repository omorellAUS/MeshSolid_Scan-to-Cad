"""Run the engine (CPU) on every case in out/user/<set>/cases.json, N in parallel, each under a timeout.
Skips cases that already have a report (delete it to rerun). Usage: python tools/testset/run_all.py drone [jobs] [timeout_s]"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SET = sys.argv[1] if len(sys.argv) > 1 else "drone"
JOBS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
TIMEOUT = int(sys.argv[3]) if len(sys.argv) > 3 else 900
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import out_dir  # noqa: E402
OUT = out_dir(SET)
RUNS = OUT / "runs"
RUNS.mkdir(parents=True, exist_ok=True)
PY = str(ROOT / ".venv" / "bin" / "python")


def run(c):
    rp = RUNS / f"{c['id']}.report.json"
    if rp.exists() or c.get("skip"):
        return
    cmd = ["timeout", "-s", "TERM", "-k", "30", str(TIMEOUT), "prlimit", "--as=7000000000", PY, "-m", "meshsolid",
           str(ROOT / c["input"]), "-o", str(RUNS / f"{c['id']}.step"), "--report", str(rp),
           "--preview", str(RUNS / f"{c['id']}.preview.png"), "--device", "cpu", "--note", f"testset {SET} {c['id']}"]
    t = time.time()
    with open(RUNS / f"{c['id']}.log", "w") as log:
        rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT,
                             env=dict(os.environ, OMP_NUM_THREADS="2", MESHSOLID_DEVICE="cpu"))
    with open(RUNS / "runs.tsv", "a") as f:
        f.write(f"{c['id']}\trc={rc}\tsecs={time.time() - t:.0f}\n")
    if not rp.exists():   # crash (OOM / segfault) before any report: record it
        rp.write_text(json.dumps({"status": "crashed", "rc": rc, "failure": f"engine exited rc={rc} without a report"}))
    print(c["id"], rc, round(time.time() - t), flush=True)


if __name__ == "__main__":
    cases = json.loads((OUT / "cases.json").read_text())
    only = sys.argv[4].split(",") if len(sys.argv) > 4 else None
    if only:
        cases = [c for c in cases if any(o in c["id"] for o in only)]
    with ThreadPoolExecutor(JOBS) as ex:
        list(ex.map(run, cases))
    print("ALLDONE", flush=True)
