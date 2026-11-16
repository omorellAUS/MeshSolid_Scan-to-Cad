"""Stage 1: detect balance cuts on both balance planes, mask them plus scan-only geometry (stub, flat
cap), and recompute scan-vs-model deviation overall and per zone, before vs after.
Usage: python scripts/stg1_balance.py out/user/stg1 [out_json] [preview_png]"""
import json, sys
from pathlib import Path
import numpy as np, trimesh
from scipy.spatial import cKDTree
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from meshsolid.balance import BalanceConfig, detect, end_faces, scan_only, cuts_json  # noqa: E402

d0 = Path(sys.argv[1]); outj = sys.argv[2] if len(sys.argv) > 2 else str(d0 / "stg1_balance.json")
png = sys.argv[3] if len(sys.argv) > 3 else str(d0 / "stg1_balance.png")
names = sys.argv[4].split(",") if len(sys.argv) > 4 else ["solid_body", "brep_v1", "circ_pattern"]


def st(x):
    if len(x) == 0:
        return None
    a = np.abs(x)
    return {"n": int(len(x)), "rms": round(float(np.sqrt(np.mean(x ** 2))), 4), "p95": round(float(np.percentile(a, 95)), 4),
            "median": round(float(np.median(a)), 4), "max": round(float(a.max()), 4)}


res = {"config": BalanceConfig().__dict__, "models": {}}
views = {}
for name in names:
    m = trimesh.load(d0 / f"{name}.tess.stl", process=False)
    c = m.bounds.mean(0); m.apply_translation([-c[0], -c[1], 0.0])
    Q, fi = trimesh.sample.sample_surface(m, 4000000, seed=4); N = m.face_normals[fi]
    P = np.load(d0 / f"scan_vs_{name}.pts.npy").astype(float); ds = np.load(d0 / f"scan_vs_{name}.signed.npy").astype(float)
    j = cKDTree(Q).query(P)[1]; p = Q[j]; n = N[j]
    Mr = np.hypot(Q[:, 0], Q[:, 1])
    nin = -np.einsum("ij,ij->i", N, np.c_[Q[:, 0], Q[:, 1], 0 * Mr] / np.maximum(Mr, 1e-9)[:, None])
    r_bore = float(np.percentile(Mr[nin > 0.9], 5)); R_out = float(np.percentile(Mr, 99.9))
    faces = end_faces(Q, N, r_bore, R_out)
    zlo, zhi = float(Q[:, 2].min()), float(Q[:, 2].max())
    cuts, cmask = detect(P, ds, n, faces, r_bore, R_out, blades=15)
    stub, cap = scan_only(P, ds, faces, zlo, zhi, r_bore)
    excl = cmask | stub | cap
    r = np.hypot(p[:, 0], p[:, 1]); er = np.c_[p[:, 0], p[:, 1], 0 * r] / np.maximum(r, 1e-9)[:, None]
    et = np.c_[-er[:, 1], er[:, 0], 0 * r]
    nt = np.abs(np.einsum("ij,ij->i", n, et)); nz = np.abs(n[:, 2])
    zone = np.full(len(P), "hub", dtype=object); zone[nt > 0.5] = "blades"
    zone[(nz > 0.85) & ((p[:, 2] - faces["z_back"]) * faces["back_sign"] > -1.0)] = "backface"
    zone[(nz > 0.85) & (np.abs(p[:, 2] - faces["z_front"]) < BalanceConfig().plane_tol) & (r < faces["r_front_out"] + 1)] = "frontface"
    zone[r < r_bore + 1.0] = "bore"
    out = {"faces": {k: round(v, 3) for k, v in faces.items()}, "r_bore": round(r_bore, 3), "R_out": round(R_out, 3),
           "model_z_range": [round(zlo, 2), round(zhi, 2)],
           "excluded_points": {"balance_cuts": int(cmask.sum()), "stub": int(stub.sum()), "flat_cap": int(cap.sum()), "total": int(excl.sum())},
           "balance_cuts": cuts_json(cuts), "search_zones": detect.last_diag,
           "before": {"overall": st(ds), "zones": {}}, "after": {"overall": st(ds[~excl]), "zones": {}},
           "after_cuts_only": {"overall": st(ds[~cmask])}}
    # scanned surfaces only: the back face of solid_body was modelled as an extrusion from physical
    # measurements (poor scan of the back), so it says nothing about reverse-engineering accuracy
    scanned = np.isin(zone, ["blades", "hub", "frontface"]) & ~excl
    out["scanned_surfaces"] = {"overall": st(ds[scanned]),
                               "zones": {zn: st(ds[(zone == zn) & ~excl]) for zn in ("blades", "hub", "frontface")}}
    for zn in ("blades", "hub", "frontface", "backface", "bore"):
        k = zone == zn
        out["before"]["zones"][zn] = st(ds[k]); out["after"]["zones"][zn] = st(ds[k & ~excl])
    res["models"][name] = out; views[name] = (P, ds, cmask, stub | cap, cuts)
    print(name, out["faces"], json.dumps({k: out[k] for k in ("excluded_points", "balance_cuts", "search_zones")}), "\n before", out["before"]["overall"], "\n after ", out["after"]["overall"], flush=True)
json.dump(res, open(outj, "w"), indent=1)

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt  # noqa: E402
fig, ax = plt.subplots(2, len(views), figsize=(6 * len(views), 11), squeeze=False)
for i, (name, (P, ds, cm, so, cuts)) in enumerate(views.items()):
    keep = ~(cm | so); k = np.where(keep)[0]; k = k[np.argsort(np.abs(ds[k]))]
    for row, (X, Y) in enumerate(((P[:, 0], P[:, 1]), (np.hypot(P[:, 0], P[:, 1]), P[:, 2]))):
        a = ax[row, i]
        sc = a.scatter(X[k], Y[k], c=np.clip(ds[k], -0.4, 0.4), s=0.06, cmap="coolwarm", vmin=-0.4, vmax=0.4, rasterized=True)
        a.scatter(X[so], Y[so], s=0.2, c="0.6", rasterized=True, label="scan-only (stub/cap), masked")
        a.scatter(X[cm], Y[cm], s=0.4, c="lime", rasterized=True, label="balance cut, masked")
        if row == 0:
            a.set_aspect("equal")
            for c in cuts:
                t = np.radians(c.angle_deg)
                a.annotate(f"{c.plane} {c.angle_deg:.0f}°\n{c.depth_mm:.2f} mm", (c.radius_mm * np.cos(t), c.radius_mm * np.sin(t)), fontsize=7)
            a.set_title(f"scan vs {name}: top view, masked regions\n(green = balance cuts, grey = stub/cap)", fontsize=9)
            a.legend(loc="lower left", fontsize=7, markerscale=20)
        else:
            a.set_xlabel("radius (mm)"); a.set_ylabel("z, model frame (mm)"); a.set_title(f"scan vs {name}: radius-z", fontsize=9)
fig.colorbar(sc, ax=ax.ravel().tolist(), shrink=0.6)
fig.savefig(png, dpi=100); print("saved", png)
