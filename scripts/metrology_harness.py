"""Minimal regression harness: run Metrology with the frozen measurement spec on (scan, solid) pairs and
append one CSV row per part.

    python scripts/metrology_harness.py --manifest accuracy_set.json [--csv results/harness.csv]
        [--out-root DIR] [--device auto|cpu|cuda] [--label TEXT]
    python scripts/metrology_harness.py --pair NAME SCAN SOLID [...]

Manifest: {"parts": [{"part": "stg1", "scan": "...", "solid": "...",
                      "zones": "box(...); cyl(...)"            # optional named per-part exclusions
                      "zones_reason": "why",                   # required with zones
                      "exclude_mask_npy": "mask.npy",          # optional per-vertex bool mask
                      "exclude_mask_reason": "why",
                      "align": true,                           # spec default; false is logged as a deviation
                      "lin": 0.01}]}                           # spec default; other values logged as a deviation
Every row carries spec_version + spec_hash + git commit. Anything that departs from the spec is listed in
the 'deviations' column. A part that errors still gets a row (status=error) so failures are never silent.
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from meshsolid import metrology as M  # noqa: E402
from meshsolid.metrology_spec import SPECS, metrics, spec_hash  # noqa: E402

COLUMNS = [
    "timestamp", "spec_version", "spec_hash", "git_commit", "label", "part", "status",
    "scan", "scan_sha12", "solid", "solid_sha12",
    "n_scan", "n_inspected", "n_excluded", "pct_excluded", "excluded_by_reason",
    "cap_mm", "capped_count", "pct_capped", "tol_mm",
    "rms_uncapped", "rms_capped", "mean_signed", "p95", "p95_capped", "p99", "max_plus", "max_minus",
    "pass_pct_v1", "pass_pct_legacy", "target_met_v1",
    "align", "rot_deg", "rot_axis", "trans_x_mm", "trans_y_mm", "trans_z_mm", "icp_median_pre_mm",
    "seconds", "device", "deviations", "error",
]
# spec v2 rows carry two extra columns (alignment detail, N); a v1-format CSV is never mixed with them
COLUMNS_V2 = COLUMNS + ["n_fold", "align_detail"]


def git_commit():
    try:
        out = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True, timeout=10)
        if out.returncode == 0 and out.stdout.strip():
            dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"],
                                   capture_output=True, text=True, timeout=10).stdout.strip()
            return out.stdout.strip() + ("-dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        pass
    f = ROOT / "meshsolid" / "_build_commit.txt"
    return f.read_text().strip() if f.is_file() else "unknown"


def sha12(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()[:12]


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.6g}"
    if isinstance(v, bool):
        return "1" if v else "0"
    return "" if v is None else str(v)


def measure(part, out_dir, device="auto", spec="v1"):
    """Run one part with the frozen spec (v1 or v2). Returns (row dict, report)."""
    SPEC = SPECS[spec]
    SPEC_VERSION = SPEC["spec_version"]
    spec_lin = SPEC["distance"]["solid_tessellation"]["linear_deflection_mm"]
    align = bool(part.get("align", SPEC["alignment"]["default"].startswith("on")))
    lin = float(part.get("lin", spec_lin))
    deviations = []
    if not align:
        deviations.append("align=off")
    if abs(lin - spec_lin) > 1e-12:
        deviations.append(f"lin={lin:g}")
    zones = part.get("zones") or ""
    if zones and not part.get("zones_reason"):
        raise ValueError(f"{part['part']}: per-part zones need a zones_reason")
    mask = None
    if part.get("exclude_mask_npy"):
        if not part.get("exclude_mask_reason"):
            raise ValueError(f"{part['part']}: exclude_mask_npy needs an exclude_mask_reason")
        mask = np.load(part["exclude_mask_npy"]).astype(bool)
    cap = float(SPEC["outlier_cap"]["cap_mm"])
    tol = float(SPEC["pass"]["tol_mm"])
    t0 = time.time()
    rep = M.run(part["scan"], part["solid"], out_dir, pass_tol=tol, exclude=SPEC["exclusions"]["default_zone_tags"],
                max_dist=cap, align=align, device=device, lin=lin, do_map=False, do_zones=False,
                do_histogram=False, do_section=False, zones_text=zones, pdf=False,
                accept_pct=SPEC["pass"]["accept_pct_display"], exclude_mask=mask,
                exclude_mask_reason=part.get("exclude_mask_reason", "external vertex mask"),
                meta={"title": f"harness {SPEC_VERSION}", "part_name": part["part"]}, keep_arrays=True,
                align_method="v2" if SPEC_VERSION == "v2" else "v1", n_fold=part.get("n_fold"),
                align_min_points=part.get("align_min_points"))
    secs = time.time() - t0
    A = rep.pop("_arrays")
    named = A["excluded"] & ~A["beyond_cap"]
    m = metrics(A["dv"], named, A["beyond_cap"], tol=tol)
    by_reason = {}
    for e in rep.get("not_inspected") or []:
        if e["reason"] == "beyond outlier cap":
            continue
        key = e["label"] if e["reason"] != e["label"] else e["reason"]
        by_reason[key] = by_reason.get(key, 0) + int(e["n_new"])
    pose = M.pose_of(A["T"])
    al = rep.get("align") or {}
    if part.get("align_min_points"):
        deviations.append(f"align_min_points={part['align_min_points']}")
    av = rep.get("alignment") or {}
    if av:
        rf, dc = av.get("refinement") or {}, av.get("direct_check") or {}
        deviations_al = (f"icp_it={av.get('iterations')};conv={int(bool(av.get('converged')))};"
                         f"bound_hit={int(bool(av.get('bound_hit')))};refine={rf.get('rotation_deg')}deg/"
                         f"{rf.get('centroid_shift_mm')}mm;direct_check_ok={int(bool(dc.get('ok')))}")
    else:
        deviations_al = ""
    row = {
        "spec_version": SPEC_VERSION, "spec_hash": spec_hash(SPEC), "part": part["part"], "status": "ok",
        "align_detail": deviations_al, "n_fold": rep.get("sectors", {}) and (rep.get("sectors") or {}).get("n_fold"),
        "scan": str(part["scan"]), "solid": str(part["solid"]),
        "excluded_by_reason": ";".join(f"{k}={v}" for k, v in by_reason.items()) or "none",
        "cap_mm": cap, "tol_mm": tol, "align": align,
        "rot_deg": pose["rotation_deg"], "rot_axis": " ".join(f"{v:.5f}" for v in pose["rotation_axis"]),
        "trans_x_mm": pose["translation_mm"][0], "trans_y_mm": pose["translation_mm"][1],
        "trans_z_mm": pose["translation_mm"][2],
        "icp_median_pre_mm": al.get("icp_median_residual_mm_pre_alignment_sampled"),
        "seconds": round(secs, 2), "device": rep.get("device_resolved"),
        "deviations": ";".join(deviations) or "none", "error": "",
    }
    row.update(m)
    return row, rep


def append_rows(csv_path, rows, columns=None):
    COLUMNS = columns or globals()["COLUMNS"]
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new = not csv_path.is_file() or csv_path.stat().st_size == 0
    if not new:
        with open(csv_path, newline="", encoding="utf-8") as f:
            hdr = next(csv.reader(f), [])
        if hdr != COLUMNS:
            raise SystemExit(f"{csv_path}: header differs from this harness's columns; "
                             "start a new CSV (columns changed means a new spec/harness version)")
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(r.get(k)) for k in COLUMNS})


def run_parts(parts, csv_path, out_root, device="auto", label="", spec="v1"):
    commit = git_commit()
    ts = _dt.datetime.now().astimezone().isoformat(timespec="seconds")
    rows = []
    cols = COLUMNS if spec == "v1" else COLUMNS_V2
    for p in parts:
        base = {"timestamp": ts, "git_commit": commit, "label": label, "part": p.get("part", "?"),
                "spec_version": spec, "spec_hash": spec_hash(SPECS[spec]), "scan": p.get("scan"), "solid": p.get("solid")}
        try:
            base["scan_sha12"], base["solid_sha12"] = sha12(p["scan"]), sha12(p["solid"])
            row, _rep = measure(p, Path(out_root) / p["part"], device=device, spec=spec)
            row = base | row
        except Exception as e:  # noqa: BLE001  (a failure is a row, never silent)
            row = base | {"status": "error", "error": f"{type(e).__name__}: {e}"}
            traceback.print_exc()
        rows.append(row)
        append_rows(csv_path, [row], cols)  # append as we go: a crash on part 3 keeps parts 1-2
        print(f"{row['part']}: {row['status']} rms_uncapped {row.get('rms_uncapped')} rms_capped "
              f"{row.get('rms_capped')} p95 {row.get('p95')} p99 {row.get('p99')} pass_v1 {row.get('pass_pct_v1')} "
              f"pass_legacy {row.get('pass_pct_legacy')} excl {row.get('pct_excluded')}% capped "
              f"{row.get('capped_count')} rot {row.get('rot_deg')} {row.get('seconds')} s", flush=True)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", help="JSON manifest {'parts': [...]}")
    ap.add_argument("--pair", nargs=3, action="append", default=[], metavar=("NAME", "SCAN", "SOLID"))
    ap.add_argument("--csv", default=str(ROOT / "results" / "harness.csv"))
    ap.add_argument("--out-root", default=str(ROOT / "results" / "harness_runs"))
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--label", default="")
    ap.add_argument("--spec", choices=sorted(SPECS), default="v1",
                    help="measurement spec (v1 rows and v2 rows go to separate CSVs: v2 adds columns)")
    a = ap.parse_args(argv)
    parts = []
    if a.manifest:
        parts += json.loads(Path(a.manifest).read_text(encoding="utf-8-sig"))["parts"]
    parts += [{"part": n, "scan": s, "solid": d} for n, s, d in a.pair]
    if not parts:
        ap.error("give --manifest or --pair")
    rows = run_parts(parts, a.csv, a.out_root, device=a.device, label=a.label, spec=a.spec)
    return 0 if all(r["status"] == "ok" for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
