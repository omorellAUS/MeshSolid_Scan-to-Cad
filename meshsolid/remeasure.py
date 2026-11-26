"""Re-measure an existing STEP + mesh pair under spec v3 and compare it with spec v2.

    python -m meshsolid.remeasure MESH.stl SOLID.step --out DIR [--convert-report REPORT] [--align]

Every metric that moves between the two specs is listed. The solids are not rebuilt.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def flat_metrics(report):
    stats = (report or {}).get("stats") or {}
    spec = dict(stats.get("spec_metrics") or {})
    spec["spec_version"] = (report or {}).get("spec_version") or stats.get("spec_version")
    balance = stats.get("balance_cuts") or (report or {}).get("balance_cuts") or {}
    if isinstance(balance, dict):
        spec["balance_cut_count"] = balance.get("count")
        spec["balance_cut_pct"] = balance.get("pct_of_scan")
        spec["balance_cut_area_mm2"] = balance.get("area_mm2")
    volume = stats.get("volume_check") or (report or {}).get("volume_check") or {}
    if isinstance(volume, dict):
        spec["volume_error_pct"] = volume.get("error_pct")
        spec["hand_entered_mm3"] = volume.get("hand_entered_mm3")
    gate = stats.get("solid_to_scan_gate") or {}
    if isinstance(gate, dict):
        spec["unsupported_pct"] = gate.get("unsupported_pct")
        spec["support_result"] = gate.get("result")
    return spec


def compare_metrics(v2, v3):
    """Every key whose value differs. Order is the v2 keys, then keys that exist only on v3."""
    left = dict(v2 or {})
    right = dict(v3 or {})
    keys = list(left.keys())
    for key in right:
        if key not in left:
            keys.append(key)
    changes = []
    for key in keys:
        a, b = left.get(key), right.get(key)
        if a != b:
            changes.append({"metric": key, "v2": a, "v3": b})
    return {"spec_pair": ["v2", "v3"], "n_changed": len(changes), "changes": changes}


def remeasure(mesh, step, out_dir, convert_report=None, align=False):
    """Run Metrology twice (v2, then v3) on an existing pair and write the comparison."""
    from .metrology import run

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    common = dict(
        align=bool(align), align_method="v2", do_map=False, pdf=False, do_histogram=False,
        do_section=False, do_zones=False, convert_report=convert_report,
    )
    v2 = run(mesh, step, out / "v2", spec_version="v2", **common)
    v3 = run(mesh, step, out / "v3", spec_version="v3", **common)
    m2, m3 = flat_metrics(v2), flat_metrics(v3)
    diff = compare_metrics(m2, m3)
    payload = {
        "mesh": str(mesh),
        "step": str(step),
        "convert_report": convert_report,
        "v2": m2,
        "v3": m3,
        "diff": diff,
        "note": "like-for-like re-measure of an existing STEP + mesh; the solid is not rebuilt",
    }
    path = out / "v2_v3_compare.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    payload["path"] = str(path)
    return payload


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m meshsolid.remeasure", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mesh", help="scan or CAD mesh (STL/OBJ/PLY)")
    ap.add_argument("step", help="existing solid STEP")
    ap.add_argument("--out", required=True, help="directory for the v2 report, v3 report and comparison")
    ap.add_argument("--convert-report", default=None, help="MeshSolid conversion report beside the STEP")
    ap.add_argument("--align", action="store_true", help="spec v2 alignment (off by default, same as Metrology)")
    args = ap.parse_args(argv)
    payload = remeasure(args.mesh, args.step, args.out, args.convert_report, align=args.align)
    print(f"spec v3 re-measure: {payload['diff']['n_changed']} metric(s) changed vs v2")
    for row in payload["diff"]["changes"]:
        print(f"  {row['metric']}: v2={row['v2']}  v3={row['v3']}")
    print("comparison", payload["path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
