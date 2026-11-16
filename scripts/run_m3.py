"""M3 ladder: noisy synthetic scans of harder prismatic parts with known truth.
Usage: python scripts/run_m3.py [out_dir] [part ...]. Writes <out>/m3_summary.json."""
import json
import sys
import resource
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from meshsolid import synthetic as S  # noqa: E402
from meshsolid.pipeline import Params, run  # noqa: E402

out = Path(sys.argv[1] if len(sys.argv) > 1 else "out/m3")
out.mkdir(parents=True, exist_ok=True)
ALL = {**S.M3_PARTS, **S.MESH_PARTS, **S.FREEFORM_PARTS}
names = sys.argv[2:] or list(ALL)
if len(names) > 1:
    # one fresh process per part: running parts back to back in one process made later segmentations
    # slower (21 s -> 308 s) and different (state left behind by earlier OCC builds), so results
    # depended on batch order
    import subprocess
    for name in names:
        subprocess.run([sys.executable, __file__, str(out), name], check=False)
    sys.exit(0)
sp = out / "m3_summary.json"
summary = json.loads(sp.read_text()) if sp.exists() else {}
for name in names:
    if name in S.MESH_PARTS:
        m, truth = S.MESH_PARTS[name]()          # mesh-native part (already a noisy scan)
    else:
        shape, truth = {**S.M3_PARTS, **S.FREEFORM_PARTS}[name]()
        m = S.to_scan(shape, noise=0.02, seed=1)
    stl = out / f"{name}.stl"
    m.export(stl)
    t0 = time.time()
    rep, *_ = run(stl, out / f"{name}.step", out / f"{name}.report.json", Params(),
                  out / f"{name}.deviation.ply", out / f"{name}.preview.png")
    b = rep["stages"]["build"]
    dev = rep.get("deviation", {})
    summary[name] = {
        "truth": truth, "status": rep["status"], "method": b.get("method"),
        "brepcheck_valid": b.get("brepcheck_valid"), "n_solids": b.get("n_solids"),
        "regions": rep["stages"]["segmentation"].get("regions"),
        "volume": b.get("volume"),
        # thread parts: compare with the intended model (thread zone as a plain cylinder), not the scan
        "volume_err_pct": (100 * (b["volume"] - truth.get("model_volume", truth["volume"])) / truth.get("model_volume", truth["volume"])) if b.get("volume") else None,
        "dims": [(d["name"], d["type"], round(d["value"], 4), round(d["fit_value"], 4)) for d in rep["dimensions"]],
        "vertices_to_solid": dev.get("mesh_vertices_to_solid"), "solid_to_mesh": dev.get("solid_to_mesh"),
        "acceptance": dev.get("acceptance"), "threads": rep.get("threads"),
        "freeform": {k: v for k, v in rep["stages"].get("freeform", {}).items() if k != "log"},
        "vertices_to_solid_excl": dev.get("mesh_vertices_to_solid_excl"), "solid_to_mesh_excl": dev.get("solid_to_mesh_excl"), "reason": b.get("reason"), "seconds": round(time.time() - t0, 1),
        "stage_seconds": {k: round(v["seconds"], 1) for k, v in rep["stages"].items()
                          if isinstance(v, dict) and isinstance(v.get("seconds"), (int, float))}
        | {"deviation": round(dev.get("seconds", 0.0), 1)},
        "peak_rss_mb_process": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
    }
    sp.write_text(json.dumps(summary, indent=1, default=str))
    s = summary[name]
    print(f"{name}: {s['status']} {s['method']} valid={s['brepcheck_valid']} vol_err={s['volume_err_pct']} "
          f"regions={s['regions']} {s['seconds']}s stages={s['stage_seconds']} rss={s['peak_rss_mb_process']}MB reason={s['reason']}", flush=True)
