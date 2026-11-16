"""Run the full pipeline on the synthetic screw rotors (out/screw/*.stl from gen_screw_parts.py) and
compare the report against the known parameters.  Writes results/m8_screw_rotors.json."""
import json, math, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, trimesh, igl
from meshsolid import synthetic_screw as S, occ_util as U
from meshsolid.pipeline import run, Params

TRUTH = {"male_4": dict(lobes=4, hand="right", lead=300.0, tip=80.0, root=48.0, length=120.0, wrap=144.0),
         "female_6": dict(lobes=6, hand="left", lead=450.0, tip=None, root=80.0, length=120.0, wrap=96.0),
         "pump_twin": dict(lobes=2, hand="right", lead=40.0, tip=44.0, root=28.0, length=100.0, wrap=900.0),
         "pump_triple": dict(lobes=2, hand="left", lead=36.0, tip=34.0, root=20.0, length=90.0, wrap=900.0),
         "variable_pitch": dict(lobes=1, hand="right", lead=[30.0, 50.0], tip=36.0, root=24.0, length=120.0, wrap=1152.0)}
out = {}
path = "results/m8_screw_rotors.json"
if os.path.exists(path):
    out = json.load(open(path))
for n in sys.argv[1:] or S.PARTS:
    t = time.time()
    rep, shape, *_ = run(f"out/screw/{n}.stl", f"out/screw/run/{n}.step", f"out/screw/run/{n}.json", Params())
    sr = rep.get("screw_rotor", {}); tr = TRUTH[n]; b = rep["stages"].get("build", {})
    row = {"status": rep["status"], "seconds": round(time.time() - t, 1), "detected": sr.get("detected"),
           "kind": sr.get("kind"), "valid": b.get("brepcheck_valid"), "solids": b.get("n_solids"), "faces": b.get("n_faces")}
    if sr.get("detected"):
        lead = sr["lead_range_mm"] if tr["lead"].__class__ is list else sr["lead_mm"]
        row["found"] = {"lobes": sr["lobes"], "hand": sr["hand"], "lead_mm": lead, "tip_d": sr["tip_diameter"],
                        "root_d": sr["root_diameter"], "length": sr["rotor_length_mm"], "wrap_deg": sr["wrap_deg"],
                        "profile_fit_rms": sr["profile_fit_rms_mm"]}
        row["truth"] = tr
        row["deviation_zones"] = sr.get("deviation_zones")
    clean = trimesh.load(f"out/screw/{n}_clean.stl")
    if shape is not None:
        T = np.load(f"out/screw/{n}_pose.npy")
        P = trimesh.transform_points(clean.vertices, T)
        tm = U.shape_to_trimesh(shape, lin=0.01, ang=0.1)
        d = np.sqrt(igl.point_mesh_squared_distance(P, np.asarray(tm.vertices, float), np.asarray(tm.faces, np.int64))[0])
        row["truth_surface_to_solid"] = {"rms": round(float(np.sqrt(np.mean(d ** 2))), 4), "p95": round(float(np.percentile(d, 95)), 4),
                                         "max": round(float(d.max()), 4)}
        row["volume"] = {"solid": round(b.get("volume", 0), 1), "truth": round(float(clean.volume), 1),
                         "rel_err": round((b.get("volume", 0) - clean.volume) / clean.volume, 6)}
    row["scan_deviation"] = {k: rep.get("deviation", {}).get(k) for k in ("mesh_to_solid", "solid_to_mesh")}
    out[n] = row
    print(n, json.dumps(row, default=float)[:900], flush=True)
    json.dump(out, open(path, "w"), indent=1, default=float)
