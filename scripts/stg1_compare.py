"""Scan-to-model deviation for an impeller, overall and by zone (bore, backface, blades, hub).
Model frame: axis = z. Usage: python scripts/stg1_compare.py scan_mm.stl model.tess.stl out_prefix [--refine]"""
import json, sys
from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial import cKDTree
sys.path.insert(0, str(Path(__file__).resolve().parent))
from step_compare import align, stats  # noqa: E402

scan = trimesh.load(sys.argv[1], process=True); model = trimesh.load(sys.argv[2], process=False); pre = sys.argv[3]
# put the model axis on z through the origin (models may sit anywhere in their file's frame)
_c = model.bounds.mean(0); model.apply_translation([-_c[0], -_c[1], 0.0])
model_shift_xy = [round(float(-_c[0]), 3), round(float(-_c[1]), 3)]
S = trimesh.sample.sample_surface(scan, 120000, seed=2)[0]
Pm = trimesh.sample.sample_surface(model, 120000, seed=3)[0]
R, t, med = align(Pm, S)                     # moves the SCAN into the model frame
assert np.linalg.det(R) > 0
V = np.asarray(scan.vertices)
sel = np.random.default_rng(0).choice(len(V), min(len(V), 300000), replace=False)
Vs = V[sel] @ R.T + t
Q, fi = trimesh.sample.sample_surface(model, 4000000, seed=4); N = model.face_normals[fi]
dd, j = cKDTree(Q).query(Vs)
ds = np.einsum("ij,ij->i", Vs - Q[j], N[j]); ds = np.where(np.abs(ds) <= dd, ds, np.sign(ds) * dd)
REFINE = "--refine" in sys.argv
if REFINE:
    # point-to-plane refinement of the same rigid fit (trimmed: only points within 0.5 mm of the model,
    # so scan-only regions do not pull); the coarse ICP above is point-to-point on sparse samples
    tQ = cKDTree(Q)
    for _ in range(10):
        n_ = N[j]; ok = (np.abs(ds) < 0.5) & (dd < 0.6)
        A_ = np.c_[np.cross(Vs[ok], n_[ok]), n_[ok]]
        x = np.linalg.lstsq(A_, -ds[ok], rcond=None)[0]
        w = x[:3]; th_ = np.linalg.norm(w)
        K = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]]) / max(th_, 1e-12)
        Ri = np.eye(3) + np.sin(th_) * K + (1 - np.cos(th_)) * K @ K
        Vs = Vs @ Ri.T + x[3:]; R = Ri @ R; t = Ri @ t + x[3:]
        dd, j = tQ.query(Vs)
        ds = np.einsum("ij,ij->i", Vs - Q[j], N[j]); ds = np.where(np.abs(ds) <= dd, ds, np.sign(ds) * dd)
        if th_ < 1e-7 and np.linalg.norm(x[3:]) < 1e-5:
            break
n = N[j]; p = Q[j]
r = np.hypot(p[:, 0], p[:, 1]); er = np.c_[p[:, 0], p[:, 1], 0 * r] / np.maximum(r, 1e-9)[:, None]
et = np.c_[-er[:, 1], er[:, 0], 0 * r]
nt = np.abs(np.einsum("ij,ij->i", n, et)); nz = np.abs(n[:, 2])
# bore radius = smallest radius with substantial inward-facing radial surface in the model
Mr = np.hypot(Q[:, 0], Q[:, 1]); nin = -np.einsum("ij,ij->i", N, np.c_[Q[:, 0], Q[:, 1], 0 * Mr] / np.maximum(Mr, 1e-9)[:, None])
bore_pts = Mr[(nin > 0.9)]
r_bore = float(np.percentile(bore_pts, 5)) if len(bore_pts) else 0.0
R_out = float(np.percentile(Mr, 99.9))
# backface: the axial end whose outer annulus (r > 0.8 R) carries axial-normal area
zlo, zhi = Q[:, 2].min(), Q[:, 2].max()
rim = Mr > 0.8 * R_out
rlo, rhi = Q[rim, 2].min(), Q[rim, 2].max()          # axial extent of the outer rim
outer = rim & (np.abs(N[:, 2]) > 0.9)
z_back = rlo if np.sum(np.abs(Q[outer, 2] - rlo) < 1.5) >= np.sum(np.abs(Q[outer, 2] - rhi) < 1.5) else rhi
zone = np.full(len(Vs), "hub", dtype=object)
zone[nt > 0.5] = "blades"
back_side = np.sign(z_back - (rlo + rhi) / 2)
zone[(nz > 0.85) & ((p[:, 2] - z_back) * back_side > -1.0)] = "backface"   # back plane and anything beyond it (boss)
zone[(r < r_bore + 1.0)] = "bore"
far = np.abs(ds) > 2.0
out = {"model": Path(sys.argv[2]).name, "point_to_plane_refined": REFINE, "model_shift_xy": model_shift_xy, "align_median_mm": round(med, 4), "rotation_det": round(float(np.linalg.det(R)), 3),
       "scan_points": int(len(Vs)), "overall": stats(ds), "signed_mean": round(float(ds.mean()), 4),
       "overall_excluding_gt2mm": stats(ds[~far]), "frac_gt2mm": round(float(far.mean()), 4),
       "gt2mm_z_range_model_frame": [round(float(np.percentile(Vs[far, 2], 1)), 2), round(float(np.percentile(Vs[far, 2], 99)), 2)] if far.any() else None,
       "model_r_bore": round(r_bore, 3), "model_R_out": round(R_out, 3), "model_z_range": [round(float(zlo), 2), round(float(zhi), 2)], "z_back": round(float(z_back), 2),
       "zones": {}}
for zn in ("blades", "hub", "bore", "backface"):
    k = zone == zn
    if k.any():
        out["zones"][zn] = {"n": int(k.sum()), **stats(ds[k]), "signed_median": round(float(np.median(ds[k])), 4),
                            "excl_gt2mm": stats(ds[k & ~far]) if (k & ~far).any() else None}
np.save(pre + ".T_scan_to_model.npy", np.r_[np.c_[R, t], [[0, 0, 0, 1]]])
np.save(pre + ".signed.npy", ds.astype(np.float32)); np.save(pre + ".pts.npy", Vs.astype(np.float32))
json.dump(out, open(pre + ".json", "w"), indent=1)
print(json.dumps(out, indent=1))
