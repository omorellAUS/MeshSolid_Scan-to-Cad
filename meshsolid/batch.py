"""M11 batch mode: run the engine on every mesh in a folder; score each built solid against a truth STEP
when one is paired with the mesh.

    python -m meshsolid.batch IN_DIR --out OUT_DIR [--recursive] [--timeout S] [--jobs 1] [engine options...]

Pairing: a mesh <name>.stl/.obj/.ply is paired with <name>.step/.stp in the same folder (case-insensitive
stem), the convention of the test-set harness folders.  Each mesh runs as its own engine process
(python -m meshsolid, a crash or memory blow-up stays in that run) with the given engine options.
Per mesh: OUT/<name>.step, .report.json, .engine.log and, for pairs, <name>.compare.json (built solid
vs truth STEP: rms/p95/max, zones, bands, volume; no alignment - the engine keeps the input frame) and
<name>.scan_vs_truth.json (input mesh vs truth STEP, aligned).  OUT/summary.json and OUT/summary.csv
list every mesh.  Nothing is written next to the inputs."""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

MESH_EXT = (".stl", ".obj", ".ply")
STEP_EXT = (".step", ".stp")


def find_pairs(in_dir, recursive=False):
    in_dir = Path(in_dir)
    it = in_dir.rglob("*") if recursive else in_dir.glob("*")
    files = sorted(p for p in it if p.is_file())
    steps = {}
    for p in files:
        if p.suffix.lower() in STEP_EXT:
            steps[(p.parent, p.stem.lower())] = p
    out = []
    for p in files:
        if p.suffix.lower() in MESH_EXT:
            out.append((p, steps.get((p.parent, p.stem.lower()))))
    return out


def _name(mesh, in_dir):
    from .names import safe_part_id
    rel = mesh.relative_to(in_dir).with_suffix("")
    # Display names may contain '&' (KK&K Stage 1). Every output file uses the safe id.
    return "__".join(safe_part_id(p) for p in rel.parts)


def run_one(mesh, truth, out_dir, name, engine_args, timeout, python=sys.executable):
    base = Path(out_dir) / name
    step, rep, log = f"{base}.step", f"{base}.report.json", f"{base}.engine.log"
    t = time.time()
    cmd = [python, "-m", "meshsolid", str(mesh), "-o", step, "--report", rep] + list(engine_args)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1]) + os.pathsep + env.get("PYTHONPATH", "")
    with open(log, "w") as lf:
        try:
            rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, timeout=timeout, env=env).returncode
            timed_out = False
        except subprocess.TimeoutExpired:
            rc, timed_out = None, True
    row = {"name": name, "mesh": str(mesh), "truth": str(truth) if truth else None, "exit_code": rc,
           "timed_out": timed_out, "seconds": round(time.time() - t, 1)}
    r = {}
    if os.path.exists(rep):
        try:
            r = json.load(open(rep))
        except Exception:  # noqa: BLE001
            r = {}
    row["status"] = "timeout" if timed_out else r.get("status", "no_report")
    row["failure"] = r.get("failure")
    d = (r.get("deviation") or {}).get("mesh_vertices_to_solid") or {}
    row["scan_to_solid_rms"] = d.get("rms"); row["scan_to_solid_p95"] = d.get("p95")
    if truth is not None:
        from .compare import compare
        if row["status"] == "ok" and os.path.exists(step):
            try:
                c = compare(step, str(truth), align=False)
                Path(f"{base}.compare.json").write_text(json.dumps(c, indent=1))
                row.update(solid_vs_truth_rms=c["test_to_ref"]["rms"], solid_vs_truth_p95=c["test_to_ref"]["p95"],
                           solid_vs_truth_max=c["test_to_ref"]["max"],
                           volume_rel_diff=(c["volume"] or {}).get("rel_diff"),
                           zones={k: {"rms": v["rms"], "p95": v["p95"], "n": v["n"]} for k, v in c["zones"].items()})
                if c["test_to_ref"]["p95"] > 1.0:
                    # the engine keeps the input frame; a scan in another pose than its STEP needs alignment
                    c2 = compare(step, str(truth), align=True)
                    Path(f"{base}.compare.json").write_text(json.dumps(c2, indent=1))
                    row.update(solid_vs_truth_rms=c2["test_to_ref"]["rms"], solid_vs_truth_p95=c2["test_to_ref"]["p95"],
                               solid_vs_truth_max=c2["test_to_ref"]["max"], solid_aligned_to_truth=True,
                               volume_rel_diff=(c2["volume"] or {}).get("rel_diff"),
                               zones={k: {"rms": v["rms"], "p95": v["p95"], "n": v["n"]} for k, v in c2["zones"].items()})
            except Exception as ex:  # noqa: BLE001
                row["compare_error"] = f"{type(ex).__name__}: {ex}"
        try:
            s = compare(str(mesh), str(truth), align=True)
            Path(f"{base}.scan_vs_truth.json").write_text(json.dumps(s, indent=1))
            row["scan_vs_truth_rms"] = s["test_to_ref"]["rms"]; row["scan_vs_truth_p95"] = s["test_to_ref"]["p95"]
        except Exception as ex:  # noqa: BLE001
            row["scan_compare_error"] = f"{type(ex).__name__}: {ex}"
    return row


CSV_COLS = ("name", "status", "seconds", "scan_to_solid_rms", "scan_to_solid_p95", "solid_vs_truth_rms",
            "solid_vs_truth_p95", "solid_vs_truth_max", "volume_rel_diff", "scan_vs_truth_rms", "scan_vs_truth_p95",
            "truth", "failure")


def run_batch(in_dir, out_dir, engine_args=(), recursive=False, timeout=3600, jobs=1, progress=print):
    """Convert every mesh. Each mesh is its own `python -m meshsolid` process (crash isolated).
    `jobs` > 1 runs that many engine processes in parallel (batch across cores)."""
    in_dir, out_dir = Path(in_dir).resolve(), Path(out_dir).resolve()
    if out_dir == in_dir:
        raise ValueError("output folder must differ from the input folder")
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = find_pairs(in_dir, recursive)
    jobs = max(1, int(jobs or 1))
    work = []
    for mesh, truth in pairs:
        name = _name(mesh, in_dir)
        work.append((mesh, truth, out_dir, name, list(engine_args), timeout))
    rows = []
    if jobs == 1 or len(work) <= 1:
        for i, args in enumerate(work, 1):
            mesh, truth, *_ = args
            progress(f"[{i}/{len(work)}] {args[3]}" + (" (with truth STEP)" if truth else ""))
            row = run_one(*args)
            rows.append(row)
            progress(f"    {row['status']} {row['seconds']} s" + (f", solid vs truth rms {row.get('solid_vs_truth_rms')} mm"
                                                                  if row.get("solid_vs_truth_rms") is not None else ""))
            _write_summary(out_dir, in_dir, rows, len(work))
        return rows
    # Parallel: each mesh stays out-of-process; futures only schedule the wrappers.
    from concurrent.futures import ThreadPoolExecutor, as_completed
    progress(f"batch jobs={jobs} meshes={len(work)}")
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        futs = {ex.submit(run_one, *args): args[3] for args in work}
        done = 0
        for fut in as_completed(futs):
            done += 1
            name = futs[fut]
            try:
                row = fut.result()
            except Exception as exn:  # noqa: BLE001
                row = {"name": name, "status": "exception", "failure": f"{type(exn).__name__}: {exn}",
                       "seconds": None, "truth": None, "exit_code": None, "timed_out": False}
            rows.append(row)
            progress(f"[{done}/{len(work)}] {name}: {row.get('status')} {row.get('seconds')} s")
            # keep summary stable-ish by name order when rewriting
            by = {r["name"]: r for r in rows}
            ordered = [by[a[3]] for a in work if a[3] in by]
            _write_summary(out_dir, in_dir, ordered, len(work))
    by = {r["name"]: r for r in rows}
    return [by[a[3]] for a in work if a[3] in by]


def _write_summary(out_dir, in_dir, rows, total):
    ok = [r for r in rows if r["status"] == "ok"]
    scored = [r for r in rows if r.get("solid_vs_truth_rms") is not None]
    summ = {"input": str(in_dir), "meshes": total, "done": len(rows), "ok": len(ok),
            "failed": len(rows) - len(ok), "with_truth": sum(1 for r in rows if r["truth"]), "scored": len(scored),
            "median_solid_vs_truth_rms": (sorted(r["solid_vs_truth_rms"] for r in scored)[len(scored) // 2] if scored else None),
            "rows": rows}
    (out_dir / "summary.json").write_text(json.dumps(summ, indent=1))
    with open(out_dir / "summary.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(CSV_COLS)
        for r in rows:
            w.writerow([r.get(c) for c in CSV_COLS])


def main(argv=None):
    ap = argparse.ArgumentParser(prog="meshsolid.batch", description=__doc__.split("\n\n")[0])
    ap.add_argument("in_dir"); ap.add_argument("--out", required=True)
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--timeout", type=float, default=3600, help="seconds per mesh")
    ap.add_argument("--jobs", type=int, default=int(os.environ.get("MESHSOLID_BATCH_JOBS", "1") or 1),
                    help="parallel engine processes (default 1; each mesh is still its own process)")
    a, engine_args = ap.parse_known_args(argv)
    if not Path(a.in_dir).is_dir():
        print(f"status failed: {a.in_dir} is not a folder"); return 2
    rows = run_batch(a.in_dir, a.out, engine_args, a.recursive, a.timeout, jobs=a.jobs)
    ok = sum(r["status"] == "ok" for r in rows)
    print(f"status done: {ok}/{len(rows)} ok; summary {Path(a.out) / 'summary.json'}")
    return 0 if rows else 2


if __name__ == "__main__":
    sys.exit(main())
