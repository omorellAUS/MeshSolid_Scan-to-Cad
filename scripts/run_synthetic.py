"""Generate noisy synthetic scans with known dimensions, run the pipeline, print truth vs recovered."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from meshsolid import synthetic as S  # noqa: E402
from meshsolid.pipeline import Params, run  # noqa: E402

out = Path(sys.argv[1] if len(sys.argv) > 1 else "out")
out.mkdir(exist_ok=True)
summary = {}
for name, fn in S.PARTS.items():
    shape, truth = fn()
    m = S.to_scan(shape, noise=0.02, seed=1)
    stl = out / f"{name}.stl"
    m.export(stl)
    rep, *_ = run(stl, out / f"{name}.step", out / f"{name}.report.json", Params(),
                  out / f"{name}.deviation.ply", out / f"{name}.preview.png")
    b = rep["stages"]["build"]
    summary[name] = {
        "truth": truth, "status": rep["status"], "brepcheck_valid": b["brepcheck_valid"],
        "volume": b["volume"], "volume_err_pct": 100 * (b["volume"] - truth["volume"]) / truth["volume"],
        "dims": [(d["name"], d["type"], round(d["value"], 4), round(d["fit_value"], 4)) for d in rep["dimensions"]],
        "deviation": rep.get("deviation", {}).get("mesh_to_solid"),
        "solid_to_mesh": rep.get("deviation", {}).get("solid_to_mesh"),
        "low_confidence_regions": [r["id"] for r in rep["regions"] if r["low_confidence"]],
    }
(out / "synthetic_summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
