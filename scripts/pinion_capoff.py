"""Controlled cap-off comparison on the pinion (X2.B1872): why did the engine report rms 0.3908 mm and
Metrology 0.056 mm for 'the same solid'?

One factor is changed at a time, from Metrology's 1.0.1 settings to the engine's gate definition:
    cap (0.2 mm -> off), exclusions (backface+bore+tip zone tags -> the engine's gate exclusions),
    alignment (ICP -> none: the engine measures in the scan frame), solid (Fusion-exported body -> engine STEP),
    tessellation (lin 0.01 -> the engine's 0.002 / ang 0.05).

    python scripts/pinion_capoff.py --engine-job DIR --metro-job DIR [--out DIR] [--device auto]

Writes capoff.json + capoff.csv (spec-v1 metrics for every configuration) into --out. Read-only on the jobs.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from meshsolid import metrology as M  # noqa: E402
from meshsolid.metrology_spec import metrics  # noqa: E402


def engine_gate_mask(scan, rep):
    """Rebuild the engine's gate exclusions on Metrology's vertex order: gear tip_edge_rounding (any face
    touching it, as pipeline.py does) + internal capture caps. Hole-fill artefact faces are not rebuilt."""
    from meshsolid.gear import cyl
    from meshsolid.internal import capture_caps
    g = rep["gear"]
    a0, u = np.asarray(g["axis_point"], float), np.asarray(g["axis_dir"], float)
    u = u / np.linalg.norm(u)
    P = np.asarray(scan.vertices, float)
    r, _th, z = cyl(P, a0, u)
    za, zb = g["z_ends_axis_frame"]
    lo, hi = g["free_fit"]["mid_flank_band_r"]
    r_root = 0.5 * float(g["free_fit"]["root_diameter"])
    gear = (z >= za - 0.05) & (z <= zb + 0.05) & (r > r_root - 0.3)
    ends = gear & ((z < za + 1.5) | (z > zb - 1.5))
    tipz = gear & ((r > hi) | ends) & ~(r < lo)
    F = np.asarray(scan.faces)
    tip_f = tipz[F].any(1)
    caps, _info = capture_caps(scan, a0, u)
    exf = tip_f | np.asarray(caps, bool)
    ev = np.zeros(len(P), bool)
    ev[F[exf].ravel()] = True
    return ev, {"tip_edge_faces": int(tip_f.sum()), "capture_cap_faces": int(np.asarray(caps).sum()),
                "engine_reported": rep["deviation"].get("excluded_from_gate"), "vertices_excluded": int(ev.sum())}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine-job", required=True)
    ap.add_argument("--metro-job", required=True, help="the Metrology job whose 0.056 is being explained")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args(argv)
    ej, mj, out = Path(a.engine_job), Path(a.metro_job), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    erep = json.loads((ej / "mesh_body.report.json").read_text())
    mrep = json.loads((mj / "report.json").read_text())
    scan_p, e_step, f_step = ej / "mesh_body.stl", ej / "mesh_body.step", mj / "solid.step"
    scan = M.load_scan(scan_p)
    mask, minfo = engine_gate_mask(scan, erep)
    np.save(out / "engine_gate_mask.npy", mask)
    print("engine mask", minfo, flush=True)
    tags = mrep["settings"]["exclude_tags"]
    C = [  # name, solid, align, tags, cap, use engine mask, lin, ang
        ("R0 metrology 1.0.1 as run (Fusion body, ICP, tags, cap 0.2)", f_step, True, tags, 0.2, False, 0.01, 0.1),
        ("R1 = R0 with the cap OFF", f_step, True, tags, None, False, 0.01, 0.1),
        ("R2 = R1 with engine exclusions instead of tags", f_step, True, [], None, True, 0.01, 0.1),
        ("R3 = R2 without ICP (scan frame)", f_step, False, [], None, True, 0.01, 0.1),
        ("R4 CONTROLLED: engine STEP, no ICP, engine exclusions, cap off", e_step, False, [], None, True, 0.01, 0.1),
        ("R5 = R4 with no exclusions (engine all-vertex 0.4269)", e_step, False, [], None, False, 0.01, 0.1),
        ("R6 = R4 at the engine's tessellation (lin 0.002, ang 0.05)", e_step, False, [], None, True, 0.002, 0.05),
        ("R7 engine STEP with metrology 1.0.1 settings (ICP, tags, cap 0.2)", e_step, True, tags, 0.2, False, 0.01, 0.1),
        ("R8 = R4 but with ICP", e_step, True, [], None, True, 0.01, 0.1),
    ]
    rows = []
    for k, (name, solid, align, tg, cap, use_mask, lin, ang) in enumerate(C):
        t0 = time.time()
        rep = M.run(scan_p, solid, out / f"R{k}", pass_tol=0.05, exclude=tg, max_dist=cap, align=align,
                    device=a.device, lin=lin, ang=ang, do_map=False, do_zones=bool(tg), do_histogram=False,
                    do_section=False, pdf=False, exclude_mask=mask if use_mask else None,
                    exclude_mask_reason="engine gate exclusions (gear tip_edge_rounding + capture caps)",
                    keep_arrays=True)
        A = rep.pop("_arrays")
        named = A["excluded"] & ~A["beyond_cap"]
        # v1 metrics treat the 0.2 cap as failures; with cap=None there are none
        bc = A["beyond_cap"] if cap else (np.abs(A["dv"]) > 0.2) & ~named
        m = metrics(A["dv"], named, bc, tol=0.05)
        pose = M.pose_of(A["T"])
        reasons = {}
        for e in rep["not_inspected"]:
            if e["reason"] != "beyond outlier cap":
                reasons[e["label"]] = e["n_new"]
        row = {"config": name, "solid": Path(solid).parent.name + "/" + Path(solid).name, "align": align,
               "tags": "+".join(tg) or "-", "cap": cap, "engine_mask": use_mask, "lin": lin, "ang": ang,
               "metrology_stats_rms": (rep.get("stats") or {}).get("rms"),
               "metrology_stats_within": (rep.get("stats") or {}).get("within_pass_pct"),
               "rot_deg": pose["rotation_deg"], "trans_mm": pose["translation_mm"],
               "excluded": reasons, "seconds": round(time.time() - t0, 1)} | m
        rows.append(row)
        print(json.dumps(row), flush=True)
    (out / "capoff.json").write_text(json.dumps({"engine_mask": minfo, "engine": {
        "all": erep["deviation"]["mesh_vertices_to_solid"], "excl": erep["deviation"].get("mesh_vertices_to_solid_excl")},
        "metrology_job": {"stats_rms": mrep["stats"]["rms"], "within": mrep["stats"]["within_pass_pct"]},
        "rows": rows}, indent=2, default=str))
    keys = list(rows[0])
    with open(out / "capoff.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
