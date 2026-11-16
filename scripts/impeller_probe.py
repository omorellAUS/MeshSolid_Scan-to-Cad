"""Rotor/ring probe for triage: axis, outer/inner diameters, height, blade counts at several radii
(FFT of the angular density of tangentially-facing surface), for meshes or point clouds."""
import json, sys
import numpy as np
import trimesh


def load_points(p):
    g = trimesh.load(p, process=False)
    if isinstance(g, trimesh.PointCloud) or len(getattr(g, "faces", [])) == 0:
        V = np.asarray(g.vertices)
        N = None
        try:
            N = np.asarray(g.metadata["_ply_raw"]["vertex"]["data"][["nx", "ny", "nz"]].tolist())
        except Exception:  # noqa: BLE001
            pass
        return V, N
    m = trimesh.load(p, force="mesh", process=False)
    m.merge_vertices()
    cc = trimesh.graph.connected_components(m.face_adjacency, nodes=np.arange(len(m.faces)), min_len=1)
    sizes = np.array([len(c) for c in cc])
    # keep components with >= 1% of the faces (drops scan debris / tiny shells)
    keep = np.concatenate([c for c in cc if len(c) >= 0.01 * len(m.faces)])
    sub = m.submesh([keep], append=True)
    P, fi = trimesh.sample.sample_surface(sub, 300000, seed=0)
    return np.asarray(P), np.asarray(sub.face_normals[fi])


def axis_of(P, N):
    if N is not None:
        # rotor axis: direction most perpendicular to the normals of the curved (blade/hub) surfaces
        # combined with PCA; use the eigenvector of the normal covariance with smallest eigenvalue
        # unless the part is disc-like, then PCA smallest-variance direction
        pass
    c = P.mean(0)
    w, U = np.linalg.eigh(np.cov((P - c).T))
    # candidate axes: the eigenvector whose eigenvalue is most distinct from the other two
    d = [abs(w[0] - w[1]) / w.max() + abs(w[0] - w[2]) / w.max() * 0,
         min(abs(w[1] - w[0]), abs(w[1] - w[2])) / w.max(),
         abs(w[2] - w[1]) / w.max()]
    k = int(np.argmax([abs(w[1] - w[0]) / w.max(), -1, abs(w[2] - w[1]) / w.max()]))
    a = U[:, k]
    # refine centre: circle fit in the plane normal to a (algebraic, on the outer points)
    e1 = np.cross(a, [1, 0, 0] if abs(a[0]) < 0.9 else [0, 1, 0]); e1 /= np.linalg.norm(e1)
    e2 = np.cross(a, e1)
    x, y = (P - c) @ e1, (P - c) @ e2
    A = np.c_[x, y, np.ones_like(x)]
    sol, *_ = np.linalg.lstsq(A, x * x + y * y, rcond=None)
    cx, cy = sol[0] / 2, sol[1] / 2
    return a, c + cx * e1 + cy * e2, e1, e2


def probe(path):
    P, N = load_points(path)
    a, c, e1, e2 = axis_of(P, N)
    d = P - c
    h = d @ a
    x, y = d @ e1, d @ e2
    r = np.hypot(x, y)
    th = np.arctan2(y, x)
    R = float(np.percentile(r, 99.9))
    out = {"axis": np.round(a, 4).tolist(), "outer_d": round(2 * R, 3), "height": round(float(np.ptp(h)), 3),
           "inner_d_min": round(2 * float(np.percentile(r, 0.05)), 3)}
    counts = {}
    if N is not None:
        tang = np.c_[-np.sin(th), np.cos(th)]
        nt = np.abs(N @ e1 * tang[:, 0] + N @ e2 * tang[:, 1])
        for f in (0.35, 0.5, 0.65, 0.8, 0.92):
            m = (np.abs(r - f * R) < 0.02 * R) & (nt > 0.6)
            if m.sum() < 500:
                continue
            hist, _ = np.histogram(th[m], 720, (-np.pi, np.pi))
            F = np.abs(np.fft.rfft(hist - hist.mean()))
            k = int(np.argmax(F[4:41]) + 4)
            # sidewalls: two per blade face pair -> the fundamental is the blade count
            counts[f"{f:.2f}R"] = {"k": k, "strength": round(float(F[k] / (F[1:200].mean() + 1e-9)), 1)}
    out["blade_count_by_radius"] = counts
    # radial histogram of points near the axis -> bore
    out["radius_percentiles"] = {q: round(float(np.percentile(r, q)), 3) for q in (0.1, 1, 5, 50, 95, 99.9)}
    return out


if __name__ == "__main__":
    res = {}
    for p in sys.argv[1:]:
        res[p.rsplit("/", 1)[-1]] = probe(p)
        print(p, json.dumps(res[p.rsplit("/", 1)[-1]]), flush=True)
