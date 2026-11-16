"""Scanned-surface metrology of a candidate impeller model with the REFERENCE model's masks and zones.

The scan points are the same vertex sample for every model (stg1_compare.py, seed 0), so the reference
(solid_body) masks - balance cuts, scan-only stub and flat cap - and its zone labels (blades / hub /
frontface / backface / bore) are applied by index to the candidate's signed deviations.  This removes the
per-model differences of the cut search and of the nearest-model-point zoning.
Usage: python scripts/stg1_metrology.py out/user/stg1 candidate [reference=solid_body] [out.json]"""
import json, sys
from pathlib import Path
import numpy as np, trimesh
from scipy.spatial import cKDTree
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from meshsolid.balance import BalanceConfig, detect, end_faces, scan_only  # noqa: E402

SUF = ""
if "--suffix" in sys.argv:                       # scan_vs_<name><suffix>.* (e.g. _r = point-to-plane refined)
    i = sys.argv.index("--suffix"); SUF = sys.argv[i + 1]; del sys.argv[i:i + 2]
d0 = Path(sys.argv[1]); cand = sys.argv[2]; ref = sys.argv[3] if len(sys.argv) > 3 else "solid_body"
outj = sys.argv[4] if len(sys.argv) > 4 else str(d0 / f"{cand}{SUF}_metrology.json")


def st(x):
    if len(x) == 0:
        return None
    a = np.abs(x)
    return {"n": int(len(x)), "rms": round(float(np.sqrt(np.mean(x ** 2))), 4), "p95": round(float(np.percentile(a, 95)), 4),
            "median": round(float(np.median(a)), 4), "max": round(float(a.max()), 4)}


def masks(name):
    m = trimesh.load(d0 / f"{name}.tess.stl", process=False)
    c = m.bounds.mean(0); m.apply_translation([-c[0], -c[1], 0.0])
    Q, fi = trimesh.sample.sample_surface(m, 4000000, seed=4); N = m.face_normals[fi]
    P = np.load(d0 / f"scan_vs_{name}{SUF}.pts.npy").astype(float); ds = np.load(d0 / f"scan_vs_{name}{SUF}.signed.npy").astype(float)
    j = cKDTree(Q).query(P)[1]; p = Q[j]; n = N[j]
    Mr = np.hypot(Q[:, 0], Q[:, 1])
    nin = -np.einsum("ij,ij->i", N, np.c_[Q[:, 0], Q[:, 1], 0 * Mr] / np.maximum(Mr, 1e-9)[:, None])
    r_bore = float(np.percentile(Mr[nin > 0.9], 5)); R_out = float(np.percentile(Mr, 99.9))
    faces = end_faces(Q, N, r_bore, R_out)
    zlo, zhi = float(Q[:, 2].min()), float(Q[:, 2].max())
    cuts, cmask = detect(P, ds, n, faces, r_bore, R_out, blades=15)
    stub, cap = scan_only(P, ds, faces, zlo, zhi, r_bore)
    r = np.hypot(p[:, 0], p[:, 1]); er = np.c_[p[:, 0], p[:, 1], 0 * r] / np.maximum(r, 1e-9)[:, None]
    et = np.c_[-er[:, 1], er[:, 0], 0 * r]
    nt = np.abs(np.einsum("ij,ij->i", n, et)); nz = np.abs(n[:, 2])
    zone = np.full(len(P), "hub", dtype=object); zone[nt > 0.5] = "blades"
    zone[(nz > 0.85) & ((p[:, 2] - faces["z_back"]) * faces["back_sign"] > -1.0)] = "backface"
    zone[(nz > 0.85) & (np.abs(p[:, 2] - faces["z_front"]) < BalanceConfig().plane_tol) & (r < faces["r_front_out"] + 1)] = "frontface"
    zone[r < r_bore + 1.0] = "bore"
    return dict(P=P, ds=ds, excl=cmask | stub | cap, zone=zone, n_cuts=len(cuts), r=np.hypot(P[:, 0], P[:, 1]), z=P[:, 2])


R = masks(ref)
C = {"ds": np.load(d0 / f"scan_vs_{cand}{SUF}.signed.npy").astype(float)}
assert len(C["ds"]) == len(R["ds"])
keep = np.isin(R["zone"], ["blades", "hub", "frontface"]) & ~R["excl"]
out = {"basis": f"scanned surfaces (blades, hub, front hub face) with {ref}'s zones and masks (balance cuts, stub, flat cap) applied to both",
       "excluded_points": int(R["excl"].sum()), "reference_cuts": R["n_cuts"]}
for name, ds in ((ref, R["ds"]), (cand, C["ds"])):
    out[name] = {"overall": st(ds[keep]), "zones": {zn: st(ds[(R["zone"] == zn) & ~R["excl"]]) for zn in ("blades", "hub", "frontface")}}
json.dump(out, open(outj, "w"), indent=1)
print(json.dumps(out, indent=1))
# where the candidate is worse in the hub zone: (r, z) bins of |ds| > 0.25
k = (R["zone"] == "hub") & ~R["excl"] & (np.abs(C["ds"]) > 0.25)
H, re, ze = np.histogram2d(R["r"][k], R["z"][k], bins=[np.arange(0, 90, 5), np.arange(0, 72, 4)])
print("candidate hub |dev|>0.25 by r (rows, 5 mm) x z (cols, 4 mm):")
for i, row in enumerate(H.astype(int)):
    if row.sum():
        print(f"  r{re[i]:4.0f} " + " ".join(f"{c:4d}" for c in row))
np.save(d0 / f"{cand}{SUF}_metrology_keep.npy", keep)
np.save(d0 / f"{ref}{SUF}_zones.npy", np.where(R["excl"], "excluded", R["zone"]).astype("U10"))
