#!/usr/bin/env python3
"""Standalone MeshSolid scoreboard (Step 0).

Reads job folders only — it never modifies the engine. For each job it records
the report fields, the SHA-256 of the input mesh and of the STEP (if present),
and tags the row with engine, spec, solid version, run ID and commit.

    python scoreboard.py --jobs DIR [DIR ...] --parts parts.csv -o scoreboard.csv
        [--engine 1.0.2a] [--spec v2] [--solid-version 1.0.2a] [--commit SHA]
        [--append] [--recursive]

A job with no STEP and an end reason (crashed, timed_out, rejected, failed)
is a valid baseline row.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

AWST = timezone(timedelta(hours=8))

COLUMNS = [
    "part_id",
    "engine",
    "engine_version",
    "spec",
    "solid_version",
    "run_id",
    "commit",
    "install",
    "job_path",
    "status",
    "end_reason",
    "failure",
    "last_stage",
    "seconds",
    "input_mesh_sha256",
    "step_sha256",
    "has_step",
    "mesh_faces",
    "mesh_volume",
    "acceptance_passed",
    "rms_excl",
    "p95_excl",
    "notes",
    "recorded_at_awst",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(8 * 1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_parts(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def match_part(parts: list[dict], mesh_sha: str | None) -> str:
    if not mesh_sha:
        return ""
    for p in parts:
        if (p.get("input_mesh_sha256") or "").strip().lower() == mesh_sha.lower():
            return (p.get("part_id") or "").strip()
    return ""


def find_report(job: Path) -> Path | None:
    for name in ("mesh_body.report.json", "report.json"):
        p = job / name
        if p.is_file():
            return p
    cands = sorted(job.glob("*.report.json"))
    return cands[0] if cands else None


def find_mesh(job: Path, report: dict | None) -> Path | None:
    for name in ("mesh_body.stl", "mesh_body.obj", "mesh_body.ply"):
        p = job / name
        if p.is_file():
            return p
    if report:
        inp = report.get("input")
        if isinstance(inp, str) and Path(inp).is_file():
            return Path(inp)
        if isinstance(inp, dict):
            for k in ("path", "file", "name"):
                v = inp.get(k)
                if v and Path(v).is_file():
                    return Path(v)
    meshes = [p for p in job.iterdir() if p.is_file() and p.suffix.lower() in {".stl", ".obj", ".ply"}]
    return meshes[0] if len(meshes) == 1 else None


def find_step(job: Path) -> Path | None:
    for name in ("mesh_body.step", "mesh_body.stp"):
        p = job / name
        if p.is_file():
            return p
    steps = [p for p in job.iterdir() if p.is_file() and p.suffix.lower() in {".step", ".stp"}]
    return steps[0] if len(steps) == 1 else None


def derive_end_reason(report: dict | None, has_step: bool) -> str:
    if not report:
        return "crashed" if not has_step else "unknown"
    if report.get("end_reason"):
        return str(report["end_reason"])
    status = (report.get("status") or "").lower()
    failure = (report.get("failure") or "")
    fl = failure.lower()
    if status == "ok":
        return "ok"
    if status in {"timeout", "timed_out"} or "killed by signal" in fl or "time budget" in fl and "timeout" in status:
        return "timed_out"
    if status == "timeout" or "TIMEOUT" in failure:
        return "timed_out"
    if "timeout" in status or status == "timed_out":
        return "timed_out"
    if "timed out" in fl or "exceeded its" in fl and "budget" in fl and not has_step and status == "timeout":
        return "timed_out"
    if status == "timeout":
        return "timed_out"
    if "gate" in fl or "rejected" in fl or status in {"gate_rejected", "rejected"}:
        return "rejected"
    if status in {"cancelled", "canceled"}:
        return "cancelled"
    if status == "incomplete":
        return "crashed"
    if status == "failed":
        return "failed"
    if not has_step:
        return "crashed"
    return status or "unknown"


def install_label(job: Path) -> str:
    s = str(job).replace("\\", "/").lower()
    if "meshsolid_testnet" in s:
        return "testnet"
    if "meshsolid_regression" in s:
        return "regression"
    if "meshsolid" in s:
        return "stable"
    return ""


def read_job(job: Path, parts: list[dict], tags: dict) -> dict:
    rp = find_report(job)
    report = None
    if rp:
        with open(rp, encoding="utf-8") as f:
            report = json.load(f)
    mesh = find_mesh(job, report)
    step = find_step(job)
    mesh_sha = sha256(mesh) if mesh else ""
    step_sha = sha256(step) if step else ""
    faces = volume = acc = rms = p95 = notes = version = status = failure = last_stage = seconds = spec = None
    if report:
        version = report.get("version")
        status = report.get("status")
        failure = report.get("failure")
        last_stage = report.get("last_stage")
        seconds = report.get("seconds")
        notes = report.get("notes")
        if isinstance(notes, list):
            notes = " | ".join(str(n) for n in notes)
        mesh_info = report.get("mesh") or {}
        if isinstance(mesh_info, dict):
            faces = mesh_info.get("faces")
            volume = mesh_info.get("volume")
        dev = report.get("deviation") or {}
        if isinstance(dev, dict):
            accp = (dev.get("acceptance") or {}).get("passed") if isinstance(dev.get("acceptance"), dict) else None
            acc = accp
            excl = dev.get("mesh_vertices_to_solid_excl") or {}
            if isinstance(excl, dict):
                rms = excl.get("rms")
                p95 = excl.get("p95")
        spec = report.get("spec_version") or (report.get("accuracy") or {}).get("class")
    end_reason = derive_end_reason(report, bool(step))
    # If the report status is "timeout" or failure mentions timeout, prefer timed_out
    if report:
        st = (report.get("status") or "").lower()
        fl = (report.get("failure") or "").lower()
        if st in {"timeout", "timed_out"} or "killed by signal" in fl:
            end_reason = "timed_out"
    part_id = match_part(parts, mesh_sha) or tags.get("part_id") or ""
    return {
        "part_id": part_id,
        "engine": tags.get("engine") or (version or ""),
        "engine_version": version or "",
        "spec": tags.get("spec") or (spec or ""),
        "solid_version": tags.get("solid_version") or (version or ""),
        "run_id": job.name,
        "commit": tags.get("commit") or (report or {}).get("git_commit") or "",
        "install": install_label(job),
        "job_path": str(job),
        "status": status or "",
        "end_reason": end_reason,
        "failure": (failure or "")[:500],
        "last_stage": last_stage or "",
        "seconds": seconds if seconds is not None else "",
        "input_mesh_sha256": mesh_sha,
        "step_sha256": step_sha,
        "has_step": "yes" if step else "no",
        "mesh_faces": faces if faces is not None else "",
        "mesh_volume": volume if volume is not None else "",
        "acceptance_passed": acc if acc is not None else "",
        "rms_excl": rms if rms is not None else "",
        "p95_excl": p95 if p95 is not None else "",
        "notes": notes or "",
        "recorded_at_awst": datetime.now(AWST).strftime("%Y-%m-%d %H:%M:%S AWST"),
    }


def iter_jobs(roots: list[Path], recursive: bool) -> list[Path]:
    out = []
    for root in roots:
        if not root.exists():
            print(f"warning: missing {root}", file=sys.stderr)
            continue
        # A job folder contains a report and/or a mesh
        if find_report(root) or find_mesh(root, None):
            out.append(root)
            continue
        children = root.rglob("*") if recursive else root.iterdir()
        for p in children:
            if p.is_dir() and (find_report(p) or find_mesh(p, None)):
                out.append(p)
    # unique, stable order
    seen = set()
    uniq = []
    for p in out:
        key = str(p.resolve()) if p.exists() else str(p)
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    return uniq


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", nargs="+", required=True, help="job folder or parent of job folders")
    ap.add_argument("--parts", required=True, help="parts.csv (part_id + input_mesh_sha256)")
    ap.add_argument("-o", "--output", required=True, help="scoreboard CSV path")
    ap.add_argument("--engine", default="", help="engine tag written on every new row")
    ap.add_argument("--spec", default="", help="measurement spec tag (e.g. v2)")
    ap.add_argument("--solid-version", default="", help="solid / package version tag")
    ap.add_argument("--commit", default="", help="engine source commit")
    ap.add_argument("--part-id", default="", help="override part_id if the mesh is not in parts.csv")
    ap.add_argument("--append", action="store_true", help="append to an existing CSV")
    ap.add_argument("--recursive", action="store_true")
    a = ap.parse_args(argv)

    parts = load_parts(Path(a.parts))
    tags = {
        "engine": a.engine,
        "spec": a.spec,
        "solid_version": a.solid_version,
        "commit": a.commit,
        "part_id": a.part_id,
    }
    jobs = iter_jobs([Path(p) for p in a.jobs], a.recursive)
    if not jobs:
        print("no job folders found", file=sys.stderr)
        return 2
    rows = [read_job(job, parts, tags) for job in jobs]
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_header = not (a.append and out.exists() and out.stat().st_size > 0)
    mode = "a" if a.append and out.exists() else "w"
    with open(out, mode, newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if write_header:
            w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"wrote {len(rows)} row(s) -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
