"""Run every verified matched drone/VTOL STL through the engine and score vs its matched STEP portion.

Writes to a NEW directory (default /workspace/meshsolid_testset/drone_pairs_all); does not touch the
baseline /workspace/meshsolid_testset/drone. Importing the source STEP is never treated as success —
only a rebuilt solid from the STL is scored.
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "testset"))

PY = str(ROOT / ".venv" / "bin" / "python")
IN = ROOT / "tests" / "user_scans" / "drone"
PAIR_MANIFEST = ROOT / "results" / "drone_stl_step_pairing" / "manifest.json"
OUT = Path(os.environ.get("MESHSOLID_PAIR_OUT", "/workspace/meshsolid_testset/drone_pairs_all"))
TIMEOUT = int(os.environ.get("MESHSOLID_PAIR_TIMEOUT", "420"))
WORKERS = int(os.environ.get("MESHSOLID_PAIR_WORKERS", "2"))


def slug(name: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def case_id(p: dict) -> str:
    return f"{p['group']}__{slug(p['stl_name'])}__userstl"


def run_one(p: dict) -> dict:
    cid = case_id(p)
    runs = OUT / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    stl = next(IN.rglob(p["stl"]))
    step_out = runs / f"{cid}.step"
    report = runs / f"{cid}.report.json"
    logp = runs / f"{cid}.log"
    preview = runs / f"{cid}.preview.png"
    for ext in (".report.json", ".step", ".preview.png", ".log"):
        q = runs / f"{cid}{ext}"
        if q.exists():
            q.unlink()
    cmd = [
        "timeout", "-s", "TERM", "-k", "30", str(TIMEOUT + 30),
        "prlimit", "--as=7000000000",
        PY, "-m", "meshsolid", str(stl),
        "-o", str(step_out), "--report", str(report), "--preview", str(preview),
        "--device", "cpu", "--note", f"drone_pairs_all {cid}",
        "--time-budget", str(TIMEOUT - 30),
    ]
    t0 = time.time()
    with logp.open("w") as log:
        rc = subprocess.call(
            cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT,
            env={**os.environ, "OMP_NUM_THREADS": "2", "MESHSOLID_DEVICE": "cpu", "PYTHONPATH": str(ROOT)},
        )
    wall = round(time.time() - t0, 1)
    if not report.exists():
        report.write_text(json.dumps({
            "status": "crashed", "rc": rc,
            "failure": f"engine exited rc={rc} without a report",
            "seconds": wall,
        }))
    rep = json.loads(report.read_text())
    row = {
        "id": cid,
        "group": p["group"],
        "stl_name": p["stl_name"],
        "stl": p["stl"],
        "truth_step": p["step"],
        "truth_product": p["product"],
        "truth_solid": p["solid"],
        "pair_how": p["how"],
        "pair_mirrored": p["mirrored"],
        "pair_p2s_p95_mm": p["p2s_p95"],
        "status": rep.get("status"),
        "failure": rep.get("failure") or rep.get("reason"),
        "last_stage": rep.get("last_stage"),
        "engine_seconds": round(float(rep.get("seconds") or 0), 1),
        "wall_seconds": wall,
        "rc": rc,
        "build_method": (rep.get("build") or {}).get("method") if isinstance(rep.get("build"), dict) else None,
        "n_faces": rep.get("n_faces") or (rep.get("solid") or {}).get("n_faces"),
        "step_written": step_out.exists(),
    }
    with (OUT / "runs.tsv").open("a") as f:
        f.write(f"{cid}\trc={rc}\tsecs={wall}\tstatus={row['status']}\n")
    print(cid, row["status"], wall, flush=True)
    return row


def main():
    only = sys.argv[1].split(",") if len(sys.argv) > 1 and sys.argv[1] not in ("-", "") else None
    man = json.loads(PAIR_MANIFEST.read_text())
    cases = man["matched"]
    if only:
        cases = [c for c in cases if any(o.lower() in case_id(c) or o.lower() in c["stl_name"].lower() for o in only)]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "runs").mkdir(exist_ok=True)
    meta = {
        "baseline_dir": man["baseline_dir"],
        "pair_run_dir": str(OUT),
        "engine_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "timeout_s": TIMEOUT,
        "workers": WORKERS,
        "n_cases": len(cases),
        "note": "STL rebuilt solid vs matched STEP truth; source STEP import is not success",
        "started": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    }
    (OUT / "run_meta.json").write_text(json.dumps(meta, indent=2))
    rows = []
    with ThreadPoolExecutor(WORKERS) as ex:
        futs = [ex.submit(run_one, c) for c in cases]
        for fut in as_completed(futs):
            rows.append(fut.result())
    rows.sort(key=lambda r: r["id"])
    with (OUT / "engine_rows.json").open("w") as f:
        json.dump(rows, f, indent=2)
    print("ENGINE_DONE", len(rows), flush=True)


if __name__ == "__main__":
    main()
