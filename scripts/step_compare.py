"""Compare user STEP models with each other and with a scan mesh.
Usage: python scripts/step_compare.py mesh.stl a.step b.step outdir"""
import json, sys, time
from collections import Counter
from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial import cKDTree
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from meshsolid import occ_util as U  # noqa: E402
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.TopAbs import TopAbs_FACE
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS
from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib

KIND = {0: "plane", 1: "cylinder", 2: "cone", 3: "sphere", 4: "torus", 5: "bezier", 6: "bspline",
        7: "revolution", 8: "extrusion", 9: "offset", 10: "other"}


def face_types(shape):
    c = Counter()
    ex = TopExp_Explorer(shape, TopAbs_FACE)
    while ex.More():
        c[KIND.get(int(BRepAdaptor_Surface(TopoDS.Face(ex.Current())).GetType()), "other")] += 1
        ex.Next()
    return dict(c)


def info(path):
    t = time.time()
    s = U.read_step(path)
    b = Bnd_Box(); BRepBndLib.Add_s(s, b)
    lo, hi = b.CornerMin(), b.CornerMax()
    m = U.shape_to_trimesh(s, lin=0.02, ang=0.1)
    return s, m, {"file": Path(path).name, "valid": bool(U.check(s)), "n_solids": int(U.count(s, __import__("OCP.TopAbs", fromlist=["TopAbs_SOLID"]).TopAbs_SOLID)),
                  "n_faces": int(U.count(s, TopAbs_FACE)), "face_types": face_types(s),
                  "volume": round(float(U.volume(s)), 2), "area": round(float(U.area(s)), 2),
                  "bbox": [round(hi.X() - lo.X(), 3), round(hi.Y() - lo.Y(), 3), round(hi.Z() - lo.Z(), 3)],
                  "bbox_min": [round(lo.X(), 3), round(lo.Y(), 3), round(lo.Z(), 3)],
                  "tess_triangles": int(len(m.faces)), "seconds": round(time.time() - t, 1)}


def pca(P):
    c = P.mean(0)
    w, V = np.linalg.eigh(np.cov((P - c).T))
    return c, V[:, ::-1]


def align(A, B, iters=50):
    """Rigid transform taking point set B onto A: PCA candidates (sign flips) + rotation about the main
    axis (shafts are near-symmetric), then trimmed point-to-point ICP. Returns (R, t, median)."""
    tA = cKDTree(A)
    ca, Ua = pca(A); cb, Ub = pca(B)
    best = None
    for s in [(a, b, c) for a in (1, -1) for b in (1, -1) for c in (1, -1)]:
        R0 = Ua @ np.diag(s) @ Ub.T
        if np.linalg.det(R0) < 0:      # proper rotations only: a mirror would flip the helix hand
            continue
        for ang in np.radians(np.arange(0, 360, 10)):
            ax = Ua[:, 0]
            K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
            Rz = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K
            R = Rz @ R0
            t = ca - R @ cb
            d, _ = tA.query(B[::10] @ R.T + t)
            sc = float(np.median(d))
            if best is None or sc < best[0]:
                best = (sc, R, t)
    _, R, t = best
    for _ in range(iters):
        Bt = B @ R.T + t
        d, j = tA.query(Bt)
        k = d < np.percentile(d, 85)
        P, Q = Bt[k], A[j[k]]
        pc, qc = P.mean(0), Q.mean(0)
        Uu, _, Vt = np.linalg.svd((P - pc).T @ (Q - qc))
        Ri = (Uu @ Vt).T
        if np.linalg.det(Ri) < 0:
            Vt[-1] *= -1; Ri = (Uu @ Vt).T
        R = Ri @ R
        t = Ri @ (t - pc) + qc
    d, _ = tA.query(B @ R.T + t)
    assert np.linalg.det(R) > 0
    return R, t, float(np.median(d))


def stats(d):
    d = np.abs(d)
    return {"rms": round(float(np.sqrt(np.mean(d ** 2))), 4), "p95": round(float(np.percentile(d, 95)), 4),
            "max": round(float(d.max()), 4), "median": round(float(np.median(d)), 4)}


def color_ply(mesh, d, path, lim):
    import matplotlib
    cm = matplotlib.colormaps["coolwarm"]
    c = (cm(np.clip(0.5 + d / (2 * lim), 0, 1)) * 255).astype(np.uint8)
    m = mesh.copy()
    m.visual.vertex_colors = c
    m.export(path)


if __name__ == "__main__":
    mesh_p, a_p, b_p, out = sys.argv[1], sys.argv[2], sys.argv[3], Path(sys.argv[4])
    out.mkdir(parents=True, exist_ok=True)
    res = {}
    sa, ma, res["A"] = info(a_p)
    sb, mb, res["B"] = info(b_p)
    print(json.dumps(res), flush=True)
    PA = trimesh.sample.sample_surface(ma, 60000, seed=0)[0]
    PB = trimesh.sample.sample_surface(mb, 60000, seed=1)[0]
    # A vs B: as-is (same coordinate frame?) and after alignment
    dAB0 = cKDTree(PA).query(PB)[0]; dBA0 = cKDTree(PB).query(PA)[0]
    R, t, _ = align(PA, PB)
    PBt = PB @ R.T + t
    dAB = cKDTree(PA).query(PBt)[0]; dBA = cKDTree(PBt).query(PA)[0]
    ang = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
    res["A_vs_B"] = {"as_is": {"B_to_A": stats(dAB0), "A_to_B": stats(dBA0), "hausdorff": round(float(max(dAB0.max(), dBA0.max())), 4)},
                     "aligned": {"rotation_deg": round(ang, 3), "translation": np.round(t, 3).tolist(),
                                 "B_to_A": stats(dAB), "A_to_B": stats(dBA),
                                 "hausdorff": round(float(max(dAB.max(), dBA.max())), 4)},
                     "volume_diff": round(res["B"]["volume"] - res["A"]["volume"], 2),
                     "volume_diff_pct": round(100 * (res["B"]["volume"] - res["A"]["volume"]) / res["A"]["volume"], 3)}
    print(json.dumps(res["A_vs_B"]), flush=True)
    # where do A and B differ? (axial position along the shaft = bbox long axis, as-is frame)
    axis = int(np.argmax(res["A"]["bbox"]))
    far = dAB0 > 0.3
    hist, edges = np.histogram(PB[far, axis], bins=28, range=(PB[:, axis].min(), PB[:, axis].max()))
    res["A_vs_B"]["B_points_off_A_by_gt0.3mm_axial_hist"] = {"axis": "xyz"[axis], "edges": np.round(edges, 1).tolist(), "counts": hist.tolist()}
    farA = dBA0 > 0.3
    hist, _ = np.histogram(PA[farA, axis], bins=edges)
    res["A_vs_B"]["A_points_off_B_by_gt0.3mm_axial_hist"] = hist.tolist()
    # each STEP vs the scan. Distances by point-to-plane against a dense oriented sample of the model
    # (closest_point on a 1.1M-triangle tessellation does not fit in memory).
    scan = trimesh.load(mesh_p, process=True)
    S = trimesh.sample.sample_surface(scan, 80000, seed=2)[0]
    V = np.asarray(scan.vertices)
    for key, m, P in (("A", ma, PA), ("B", mb, PB)):
        R, t, _ = align(S, P)              # move the STEP onto the scan
        T = np.r_[np.c_[R, t], [[0, 0, 0, 1]]]
        mt = m.copy(); mt.apply_transform(T)
        Q, fi = trimesh.sample.sample_surface(mt, 3000000, seed=4)
        Nq = mt.face_normals[fi]
        dd, j = cKDTree(Q).query(V)
        ds = np.einsum("ij,ij->i", V - Q[j], Nq[j])           # + = scan outside model
        ds = np.where(np.abs(ds) < dd + 1e-9, ds, np.sign(ds) * dd)
        dm = cKDTree(V).query(trimesh.sample.sample_surface(mt, 80000, seed=3)[0])[0]
        res[f"{key}_vs_scan"] = {"rotation_deg": round(float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))), 3),
                                 "translation": np.round(t, 3).tolist(),
                                 "scan_to_model": stats(ds), "scan_to_model_signed_mean": round(float(ds.mean()), 4),
                                 "model_to_scan_vertices_unsigned": stats(dm)}
        color_ply(scan, ds, out / f"scan_vs_step{key}.deviation.ply", 0.2)
        np.save(out / f"T_step{key}_to_scan.npy", T)
        np.save(out / f"scan_vs_step{key}.signed.npy", ds.astype(np.float32))
        print(key, json.dumps(res[f"{key}_vs_scan"]), flush=True)
    json.dump(res, open(out / "step_compare.json", "w"), indent=1)
    print("done")
