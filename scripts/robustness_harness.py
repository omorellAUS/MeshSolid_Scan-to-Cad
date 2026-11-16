"""Robustness regression set: conversions that once crashed or hung must now end CLEANLY.

    python scripts/robustness_harness.py --manifest robustness_set.json [--csv results/robustness.csv]
        [--out-root DIR] [--label TEXT] [--time-budget S]

Manifest: {"cases": [{"case": "kkk_stg1", "input": "...mesh.stl", "sha256": "...", "params": {...},
                      "was": "ValueError: no crossing in impeller.analyse (1.0.1)",
                      "expect": "ok",            # optional: a positive case must convert (expect_met)
                      "determinism": true}]}     # optional: run twice, the numbers must be identical
--determinism runs every selected case twice.
Each case runs the engine CLI in its own process (python -m meshsolid) with a wall-clock budget. A case is
CLEAN when the process exits, the report exists and its status is ok or failed with a failure text (never
'incomplete' = crash/kill, never a Python traceback in the log). One CSV row per case, separate from the
accuracy set's harness CSVs; a case that errors still gets a row. The part files never leave the PC."""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

COLUMNS = ["timestamp", "git_commit", "label", "case", "input_sha12", "sha_ok", "exit_code", "clean", "status",
           "last_stage", "failure", "recognisers_failed", "impeller_blades", "verdict_target", "verdict_engine_limit",
           "rms_uncapped", "pass_pct_v1", "pct_excluded", "seconds", "was",
           "expect", "expect_met", "input_class", "rms_gate", "volume", "deterministic", "determinism_diff"]

# numbers compared by the determinism check (same input twice -> identical)
DET_KEYS = ("status", "volume", "n_faces", "rms_gate", "p95_gate", "rms_uncapped", "pass_pct_v1", "impeller_blades")


def _sha(p, n=None):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def run_case(c, out_root, budget):
    out = Path(out_root) / c["case"]
    out.mkdir(parents=True, exist_ok=True)
    step, rpt, log = out / "out.step", out / "report.json", out / "engine.log"
    c = dict(c, input=os.path.expandvars(c["input"]))
    sha = _sha(c["input"])
    row = {"case": c["case"], "input_sha12": sha[:12], "was": c.get("was", ""),
           "sha_ok": (sha.lower() == c["sha256"].lower()) if c.get("sha256") else ""}
    cmd = [sys.executable, "-u", "-m", "meshsolid", c["input"], "-o", str(step), "--report", str(rpt),
           "--time-budget", str(budget)]
    for k, v in (c.get("params") or {}).items():
        cmd += [f"--{k.replace('_', '-')}", str(v)]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    t0 = time.time()
    with open(log, "w", encoding="utf-8", errors="replace") as lf:
        try:
            p = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, timeout=budget + 600)
            row["exit_code"] = p.returncode
        except subprocess.TimeoutExpired:
            row["exit_code"] = "killed (budget + 600 s)"
    row["seconds"] = round(time.time() - t0, 1)
    rep = {}
    try:
        rep = json.loads(rpt.read_text())
    except (OSError, ValueError):
        pass
    tb = "Traceback (most recent call last)" in log.read_text(encoding="utf-8", errors="replace")
    st = rep.get("status")
    row.update(status=st or "no report", last_stage=rep.get("last_stage", ""),
               failure=(rep.get("failure") or "")[:300],
               recognisers_failed=";".join(f"{r['name']}: {r.get('error', '')[:80]}" for r in rep.get("recognisers", [])
                                           if r.get("status") == "failed"),
               impeller_blades=(rep.get("stages", {}).get("impeller_probe") or {}).get("blades"))
    v = rep.get("verdict") or {}
    row["verdict_target"] = (v.get("target") or {}).get("label", "")
    row["verdict_engine_limit"] = (v.get("engine_limit") or {}).get("label", "")
    m = v.get("metrics") or {}
    row.update(rms_uncapped=m.get("rms_uncapped"), pass_pct_v1=m.get("pass_pct_v1"), pct_excluded=m.get("pct_excluded"))
    dsum = (rep.get("deviation") or {}).get("summary") or {}
    b = (rep.get("stages") or {}).get("build") or {}
    row.update(input_class=(rep.get("input_class") or {}).get("class", ""), rms_gate=dsum.get("vertex_to_solid_rms_gate_mm"),
               p95_gate=dsum.get("vertex_to_solid_p95_gate_mm"), volume=b.get("volume"), n_faces=b.get("n_faces"))
    if c.get("expect"):
        row["expect"] = c["expect"]; row["expect_met"] = (st == c["expect"])
    row["clean"] = bool(st in ("ok", "partial", "failed", "timeout") and (st == "ok" or row["failure"]) and not tb
                        and row["exit_code"] in (0, 1, 2))
    return row


def determinism_diff(r1, r2):
    """(identical, text): DET_KEYS of two runs of the same input."""
    diffs = []
    for k in DET_KEYS:
        a, b = r1.get(k), r2.get(k)
        if a != b:
            diffs.append(f"{k}: {a} vs {b}")
    return (not diffs), "; ".join(diffs)


def main(argv=None):
    from scripts.metrology_harness import git_commit
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--csv", default=str(ROOT / "results" / "robustness.csv"))
    ap.add_argument("--out-root", default=str(ROOT / "results" / "robustness_runs"))
    ap.add_argument("--label", default="")
    ap.add_argument("--time-budget", type=float, default=3600.0)
    ap.add_argument("--cases", default="")
    ap.add_argument("--determinism", action="store_true", help="run every selected case twice and compare")
    a = ap.parse_args(argv)
    cases = json.loads(Path(a.manifest).read_text(encoding="utf-8-sig"))["cases"]
    want = set(filter(None, a.cases.split(",")))
    csv_path = Path(a.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new = not csv_path.exists()
    n_bad = 0
    for c in cases:
        if want and c["case"] not in want:
            continue
        try:
            row = run_case(c, a.out_root, a.time_budget)
            if a.determinism or c.get("determinism"):
                row2 = run_case(dict(c, case=c["case"] + "__run2"), a.out_root, a.time_budget)
                same, txt = determinism_diff(row, row2)
                row["deterministic"] = same; row["determinism_diff"] = txt[:300]
                row["seconds"] = f"{row.get('seconds')} + {row2.get('seconds')}"
        except Exception as ex:  # noqa: BLE001
            row = {"case": c["case"], "status": "harness error", "failure": f"{type(ex).__name__}: {ex}", "clean": False}
        row.update(timestamp=_dt.datetime.now().astimezone().isoformat(timespec="seconds"), git_commit=git_commit(),
                   label=a.label)
        n_bad += not row.get("clean")
        cols = COLUMNS
        if not new:
            # an existing CSV keeps its own header (older files have fewer columns)
            with open(csv_path, newline="", encoding="utf-8") as f_:
                cols = next(csv.reader(f_), None) or COLUMNS
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            if new:
                w.writeheader(); new = False
            w.writerow(row)
        print(f"{row['case']}: clean={row.get('clean')} status={row.get('status')} stage={row.get('last_stage')} "
              f"{row.get('seconds')} s | {row.get('failure', '')[:140]} | rec_failed={row.get('recognisers_failed', '')}"
              + (f" | expect {row['expect']} met={row.get('expect_met')}" if row.get("expect") else "")
              + (f" | deterministic={row.get('deterministic')} {row.get('determinism_diff', '')}" if "deterministic" in row else ""),
              flush=True)
    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main())
