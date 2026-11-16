"""M11: compare a test shape with a reference: scan vs scan, scan vs STEP, built STEP vs truth STEP.

    python -m meshsolid.compare TEST REF [--report out.json] [--no-align] [--mirror] [--samples N]

TEST / REF: mesh (STL/OBJ/PLY) or STEP.  Both are sampled by area; TEST is rigidly aligned onto REF
(PCA starts + trimmed point-to-point ICP, proper rotations only unless --mirror) unless --no-align.
Distances are exact point-to-triangle (libigl), signed by the reference's outward normal where the
reference is closed (positive = TEST outside REF).  Reported:
  test_to_ref / ref_to_test   rms, p95, max, mean signed, n
  hausdorff                   max of both directions
  bands                       fraction of TEST samples within 0.02 / 0.05 / 0.1 / 0.2 / 0.5 mm of REF
  zones                       by the face type of the nearest REF face when REF is a STEP (planar,
                              cylindrical, conical_toroidal, freeform: the test-set harness zones,
                              tools/testset/common.py), plus 'edges' = TEST samples within e of a REF
                              face boundary (the zone scans round), e = max(0.5, 2 x mean sample spacing)
  volume                      both volumes and the relative difference when both are closed
The zone vocabulary and STEP tessellation come from the test-set harness (tools/testset/common.py) so
compare and harness scores are read the same way."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
BANDS = (0.02, 0.05, 0.1, 0.2, 0.5)
STEP_EXT = (".step", ".stp")


def _harness():
    """tools/testset/common.py (owned by the test-set harness); imported, never copied."""
    p = ROOT / "tools" / "testset"
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
    import common  # noqa: E402
    return common


def load(path, lin=0.01):
    """(trimesh, per-face kind array or None, source kind) for a mesh or STEP file."""
    path = Path(path)
    if path.suffix.lower() in STEP_EXT:
        from . import occ_util as U
        C = _harness()
        shape = U.read_step(str(path))
        m, kinds = C.tessellate_labeled(shape, lin)
        m.merge_vertices()
        return m, kinds, "step"
    m = trimesh.load(str(path), force="mesh", process=True)
    if not isinstance(m, trimesh.Trimesh) or len(m.faces) == 0:
        raise ValueError(f"{path}: no triangles")
    return m, None, "mesh"


def _icp(A, B, mirror=False, iters=40):
    """Rigid transform taking points B onto A (trimmed point-to-point ICP from the 4 proper PCA starts;
    with mirror the mirrored starts too).  Returns (4x4, median residual, mirrored)."""
    from scipy.spatial import cKDTree
    tA = cKDTree(A)

    def pca(X):
        c = X.mean(0); _, V = np.linalg.eigh(np.cov((X - c).T)); return c, V[:, ::-1]
    ca, Ua = pca(A)
    best = None
    for M in ([np.eye(3)] + ([np.diag([-1.0, 1, 1])] if mirror else [])):
        Bm = B @ M.T
        cb, Ub = pca(Bm)
        for s in [(a, b, c) for a in (1, -1) for b in (1, -1) for c in (1, -1)]:
            R = Ua @ np.diag(s) @ Ub.T
            if np.linalg.det(R) < 0:
                continue
            t = ca - R @ cb
            for _ in range(iters):
                Bt = Bm @ R.T + t
                d, j = tA.query(Bt, workers=-1)
                k = d <= np.percentile(d, 90)
                P, Q = Bt[k], A[j[k]]
                pc, qc = P.mean(0), Q.mean(0)
                Uu, _, Vt = np.linalg.svd((P - pc).T @ (Q - qc))
                Ri = (Uu @ Vt).T
                if np.linalg.det(Ri) < 0:
                    Vt[-1] *= -1; Ri = (Uu @ Vt).T
                R = Ri @ R; t = Ri @ (t - pc) + qc
            med = float(np.median(tA.query(Bm @ R.T + t, workers=-1)[0]))
            if best is None or med < best[1]:
                T = np.eye(4); T[:3, :3] = R @ M; T[:3, 3] = t
                best = (T, med, M[0, 0] < 0)
    return best


def _dist(P, m, signed):
    import igl
    V = np.asarray(m.vertices, float); F = np.asarray(m.faces, np.int64)
    d2, fi, C = igl.point_mesh_squared_distance(np.asarray(P, float), V, F)
    d = np.sqrt(d2)
    if signed:
        s = np.einsum("ij,ij->i", P - C, m.face_normals[fi])
        d = np.where(s < 0, -d, d)
    return d, fi


def _st(d):
    a = np.abs(d)
    if len(a) == 0:
        return None
    return {"rms": round(float(np.sqrt(np.mean(a ** 2))), 5), "p95": round(float(np.percentile(a, 95)), 5),
            "max": round(float(a.max()), 5), "mean_signed": round(float(np.mean(d)), 5), "n": int(len(a))}


def _boundary_dist(P, m, kinds_face_id):
    """Distance from P to the REF face boundaries (edges between triangles of different source faces)."""
    from scipy.spatial import cKDTree
    e = m.face_adjacency_edges; fa = m.face_adjacency
    k = kinds_face_id[fa[:, 0]] != kinds_face_id[fa[:, 1]]
    k |= m.face_adjacency_angles > math.radians(20)
    if not k.any():
        return np.full(len(P), np.inf)
    a, b = m.vertices[e[k, 0]], m.vertices[e[k, 1]]
    L = np.linalg.norm(b - a, axis=1)
    n = np.maximum(1, np.ceil(L / 0.05)).astype(int)
    pts = np.vstack([a[i] + np.outer(np.linspace(0, 1, n[i] + 1), b[i] - a[i]) for i in range(len(a))])
    return cKDTree(pts).query(P, workers=-1)[0]


def compare(test, ref, align=True, mirror=False, samples=200000, seed=0):
    t0 = time.time()
    mt, kt, st = load(test) if not isinstance(test, tuple) else test
    mr, kr, sr = load(ref) if not isinstance(ref, tuple) else ref
    Pt = np.asarray(trimesh.sample.sample_surface(mt, samples, seed=seed)[0])
    Pr = np.asarray(trimesh.sample.sample_surface(mr, samples, seed=seed + 1)[0])
    out = {"test": str(test if not isinstance(test, tuple) else "<mesh>"), "ref": str(ref if not isinstance(ref, tuple) else "<mesh>"),
           "test_kind": st, "ref_kind": sr, "samples": samples}
    T = np.eye(4)
    if align:
        T, med, mir = _icp(Pr, Pt, mirror=mirror)
        Pt = Pt @ T[:3, :3].T + T[:3, 3]
        mt = mt.copy(); mt.apply_transform(T)
        out["alignment"] = {"transform": np.round(T, 6).tolist(), "median_residual_mm": round(med, 5), "mirrored": bool(mir)}
    else:
        out["alignment"] = None
    dtr, fi = _dist(Pt, mr, signed=bool(mr.is_watertight))
    drt, _ = _dist(Pr, mt, signed=bool(mt.is_watertight))
    out["test_to_ref"] = _st(dtr); out["ref_to_test"] = _st(drt)
    out["signed"] = {"test_to_ref": bool(mr.is_watertight), "ref_to_test": bool(mt.is_watertight)}
    out["hausdorff"] = round(float(max(np.abs(dtr).max(), np.abs(drt).max())), 5)
    out["bands"] = {f"<= {b} mm": round(float(np.mean(np.abs(dtr) <= b)), 4) for b in BANDS}
    zones = {}
    if kr is not None:
        C = _harness()
        zf = np.array([C.zone_of(k) for k in kr])
        for z in ("planar", "cylindrical", "conical_toroidal", "freeform"):
            s = _st(dtr[zf[fi] == z])
            if s:
                zones[z] = s
        spacing = math.sqrt(mr.area / samples)
        e = max(0.5, 2 * spacing)
        # face ids: tessellate_labeled emits faces in order; source face id = run of equal-kind blocks is not
        # enough, so boundaries also include sharp mesh edges (> 20 deg)
        fid = np.unique(kr, return_inverse=True)[1]
        bd = _boundary_dist(Pt, mr, fid)
        near = bd <= e
        if near.any():
            zones["edges"] = _st(dtr[near]) | {"within_mm": round(e, 3)}
            zones["away_from_edges"] = _st(dtr[~near])
    out["zones"] = zones
    vol = {}
    if mt.is_watertight and mr.is_watertight:
        vt, vr = abs(float(mt.volume)), abs(float(mr.volume))
        vol = {"test": round(vt, 3), "ref": round(vr, 3), "rel_diff": round((vt - vr) / vr, 6)}
    out["volume"] = vol or None
    out["seconds"] = round(time.time() - t0, 2)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="meshsolid.compare", description="Compare a mesh/STEP with a reference mesh/STEP.")
    ap.add_argument("test"); ap.add_argument("ref")
    ap.add_argument("--report", help="write the comparison JSON here")
    ap.add_argument("--no-align", action="store_true", help="the two are already in the same frame")
    ap.add_argument("--mirror", action="store_true", help="also try mirrored alignments (left/right twins)")
    ap.add_argument("--samples", type=int, default=200000)
    a = ap.parse_args(argv)
    try:
        res = compare(a.test, a.ref, align=not a.no_align, mirror=a.mirror, samples=a.samples)
        res["status"] = "ok"
    except Exception as ex:  # noqa: BLE001
        res = {"status": "failed", "error": f"{type(ex).__name__}: {ex}", "test": a.test, "ref": a.ref}
    txt = json.dumps(res, indent=1)
    if a.report:
        Path(a.report).write_text(txt)
    s = res.get("test_to_ref") or {}
    print(f"status {res['status']}" + (f": test->ref rms {s.get('rms')} p95 {s.get('p95')} max {s.get('max')} mm; "
                                         f"hausdorff {res.get('hausdorff')} mm" if res["status"] == "ok" else f": {res.get('error')}"))
    return 0 if res["status"] == "ok" else 2


if __name__ == "__main__":
    sys.exit(main())
