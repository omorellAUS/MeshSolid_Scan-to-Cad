"""Per-solid breakdown of the two STEPs and axial radius profiles of STEP A, STEP B and the scan in the
STEP frame (axis = x). Usage: python scripts/pinion_profiles.py scan.stl a.step b.step outdir"""
import json, sys
from pathlib import Path
import numpy as np
import trimesh
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from meshsolid import occ_util as U  # noqa: E402
from OCP.TopAbs import TopAbs_SOLID
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS
from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse
from OCP.collections import List_TopoDS_Shape as TopTools_ListOfShape

scan_p, a_p, b_p, out = sys.argv[1:5]; out = Path(out)
res = {}
meshes = {}
for key, p in (("A", a_p), ("B", b_p)):
    s = U.read_step(p)
    ex = TopExp_Explorer(s, TopAbs_SOLID); sol = []
    solids = []
    while ex.More():
        so = TopoDS.Solid_s(ex.Current()) if hasattr(TopoDS, "Solid_s") else TopoDS.Solid(ex.Current())
        b = Bnd_Box(); BRepBndLib.Add_s(so, b); lo, hi = b.CornerMin(), b.CornerMax()
        sol.append({"volume": round(U.volume(so), 2), "faces": U.count(so, __import__("OCP.TopAbs", fromlist=["x"]).TopAbs_FACE),
                    "x": [round(lo.X(), 2), round(hi.X(), 2)], "r_max": round(max(abs(lo.Y()), abs(hi.Y()), abs(lo.Z()), abs(hi.Z())), 3)})
        solids.append(so); ex.Next()
    # fuse to get the union volume (overlaps between bodies)
    fused = None
    try:
        args = TopTools_ListOfShape(); tools = TopTools_ListOfShape()
        args.Append(solids[0])
        for so in solids[1:]: tools.Append(so)
        f = BRepAlgoAPI_Fuse(); f.SetArguments(args); f.SetTools(tools); f.SetFuzzyValue(1e-4); f.Build()
        if f.IsDone():
            fs = f.Shape(); fused = {"volume": round(U.volume(fs), 2), "n_solids": U.count(fs, TopAbs_SOLID), "valid": bool(U.check(fs))}
    except Exception as e:  # noqa: BLE001
        fused = {"error": str(e)}
    meshes[key] = U.shape_to_trimesh(s, lin=0.02, ang=0.1)   # whole compound (profiles use outer radius)
    sol.sort(key=lambda d: d["x"][0])
    res[key] = {"sum_solid_volumes": round(sum(d["volume"] for d in sol), 2), "fused": fused,
                "solids": sol}
    print(key, json.dumps({k: v for k, v in res[key].items() if k != "solids"}), flush=True)
scan = trimesh.load(scan_p, process=True)
T = np.load(out / "T_stepA_to_scan.npy")
scan.apply_transform(np.linalg.inv(T))           # scan into STEP-A frame
meshes["scan"] = scan
TB = np.load(out / "T_stepB_to_scan.npy")
# B into A frame: B->scan then scan->A
if "B" in meshes:
    meshes["B"].apply_transform(np.linalg.inv(T) @ TB)
prof = {}
edges = np.arange(46.0, 325.0, 0.5)
for key, m in meshes.items():
    P = trimesh.sample.sample_surface(m, 1500000, seed=0)[0]
    r = np.hypot(P[:, 1], P[:, 2]); x = P[:, 0]
    i = np.digitize(x, edges)
    rmax = np.array([np.percentile(r[i == k], 99) if (i == k).sum() > 50 else np.nan for k in range(1, len(edges))])
    rmed = np.array([np.median(r[i == k]) if (i == k).sum() > 50 else np.nan for k in range(1, len(edges))])
    prof[key] = {"rmax99": rmax, "rmed": rmed}
xc = (edges[:-1] + edges[1:]) / 2
rows = []
for k in range(len(xc)):
    rows.append([round(float(xc[k]), 2)] + [None if np.isnan(prof[q]["rmax99"][k]) else round(float(2 * prof[q]["rmax99"][k]), 3) for q in ("scan", "A", "B") if q in prof])
res["profile_cols"] = ["x", "D99_scan", "D99_A", "D99_B"]
res["profile"] = rows
json.dump(res, open(out / "pinion_profiles.json", "w"), indent=1)
print("done")
