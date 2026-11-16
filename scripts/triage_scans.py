"""Triage of user scan meshes: stats, multi-view previews, contact sheet, same-part grouping.
Usage: python scripts/triage_scans.py tests/user_scans results/user_scans"""
import json, sys, time
from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial import cKDTree

src, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
files = sorted(p for p in src.iterdir() if p.suffix.lower() in (".stl", ".ply", ".obj"))


def load(p):
    g = trimesh.load(p, process=False)
    if isinstance(g, trimesh.PointCloud) or len(getattr(g, "faces", [])) == 0:
        return g if isinstance(g, trimesh.PointCloud) else trimesh.PointCloud(g.vertices)
    m = trimesh.load(p, force="mesh", process=False)
    m.merge_vertices()          # STL soups -> shared vertices (exact duplicates only)
    return m


def noise_estimate(m, n=4000, k=24, seed=0):
    """Median RMS distance of vertices to their local PCA plane (k-NN): roughness incl. scanner noise."""
    V = np.asarray(m.vertices)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(V), min(n, len(V)), replace=False)
    tr = cKDTree(V)
    _, nn = tr.query(V[idx], k)
    G = V[nn] - V[nn].mean(1, keepdims=True)
    ev = np.linalg.eigvalsh(np.einsum("nki,nkj->nij", G, G) / k)
    return float(np.median(np.sqrt(np.maximum(ev[:, 0], 0))))


def render_views(m, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    mm = m
    if len(m.faces) > 80000:      # display copy only: quadric decimation keeps the surface closed-looking
        import pymeshlab
        ms = pymeshlab.MeshSet()
        ms.add_mesh(pymeshlab.Mesh(np.asarray(m.vertices, float), np.asarray(m.faces, np.int32)))
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum=80000, preservenormal=True)
        q = ms.current_mesh()
        mm = trimesh.Trimesh(q.vertex_matrix(), q.face_matrix(), process=False)
    T = mm.vertices[mm.faces]
    n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    views = [("iso", 25, 45), ("front", 0, -90), ("top", 90, -90), ("side", 0, 0)]
    fig = plt.figure(figsize=(16, 4.4))
    c, e = m.bounds.mean(0), float(np.max(np.ptp(m.bounds, 0))) / 2
    for i, (nm, el, az) in enumerate(views):
        ax = fig.add_subplot(1, 4, i + 1, projection="3d")
        ea, aa = np.radians(el), np.radians(az)
        vd = np.array([np.cos(ea) * np.cos(aa), np.cos(ea) * np.sin(aa), np.sin(ea)])
        light = vd + np.array([0.3, 0.3, 0.5])
        light /= np.linalg.norm(light)
        sh = 0.12 + 0.8 * np.abs(n @ light) ** 1.5
        pc = Poly3DCollection(T, facecolors=np.c_[sh * 0.75, sh * 0.8, sh * 0.9, np.ones(len(sh))], linewidths=0)
        ax.add_collection3d(pc)
        for f, cc in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), c):
            f(cc - e, cc + e)
        ax.view_init(el, az)
        ax.set_box_aspect((1, 1, 1))
        ax.set_axis_off()
        ax.set_title(nm, fontsize=9)
    fig.suptitle(title, fontsize=10)
    fig.savefig(path, dpi=80, bbox_inches="tight")
    plt.close(fig)


def render_points(V, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    P = V[np.random.default_rng(0).choice(len(V), min(40000, len(V)), replace=False)]
    views = [("iso", 25, 45), ("front", 0, -90), ("top", 90, -90), ("side", 0, 0)]
    fig = plt.figure(figsize=(16, 4.4))
    c, e = (V.min(0) + V.max(0)) / 2, float(np.max(np.ptp(V, 0))) / 2
    for i, (nm, el, az) in enumerate(views):
        ax = fig.add_subplot(1, 4, i + 1, projection="3d")
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=0.2, c=P[:, 2], cmap="viridis")
        for f, cc in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), c):
            f(cc - e, cc + e)
        ax.view_init(el, az); ax.set_box_aspect((1, 1, 1)); ax.set_axis_off(); ax.set_title(nm, fontsize=9)
    fig.suptitle(title, fontsize=10)
    fig.savefig(path, dpi=80, bbox_inches="tight")
    plt.close(fig)


rows = {}
meshes = {}
for p in files:
    t = time.time()
    m = load(p)
    meshes[p.stem] = m
    if isinstance(m, trimesh.PointCloud):
        ext = np.ptp(m.vertices, axis=0)
        V = np.asarray(m.vertices)
        tr = cKDTree(V)
        dnn, _ = tr.query(V[np.random.default_rng(0).choice(len(V), 20000, replace=False)], 2)
        r = {"file": p.name, "format": p.suffix[1:], "kind": "point cloud (no faces)", "triangles": 0,
             "vertices": int(len(V)), "bbox": [round(float(x), 3) for x in ext],
             "bbox_min": [round(float(x), 3) for x in V.min(0)],
             "units_guess": "mm" if 5 < ext.max() < 2000 else "unknown",
             "point_spacing_mm": round(float(np.median(dnn[:, 1])), 4),
             "roughness_mm": round(noise_estimate(m), 4)}
        render_points(V, out / f"{p.stem}.views.png", f"{p.name}: point cloud {len(V)} pts, bbox {r['bbox']} mm")
        r["seconds"] = round(time.time() - t, 1)
        rows[p.stem] = r
        print(p.stem, r, flush=True)
        continue
    E = m.edges_sorted
    eu, cnt = np.unique(E, axis=0, return_counts=True)
    el = np.linalg.norm(m.vertices[eu[:, 0]] - m.vertices[eu[:, 1]], axis=1)
    comps = trimesh.graph.connected_components(m.face_adjacency, nodes=np.arange(len(m.faces)), min_len=1)
    sizes = sorted((len(c) for c in comps), reverse=True)
    ext = np.ptp(m.bounds, axis=0)
    r = {
        "file": p.name, "format": p.suffix[1:], "triangles": int(len(m.faces)), "vertices": int(len(m.vertices)),
        "bbox": [round(float(x), 3) for x in ext], "bbox_min": [round(float(x), 3) for x in m.bounds[0]],
        "units_guess": "mm" if 5 < ext.max() < 2000 else ("m" if ext.max() <= 5 else "unknown"),
        "watertight": bool(m.is_watertight), "open_edges": int((cnt == 1).sum()),
        "nonmanifold_edges": int((cnt > 2).sum()), "components": len(sizes), "largest_components_faces": sizes[:5],
        "mean_edge_mm": round(float(el.mean()), 4), "median_edge_mm": round(float(np.median(el)), 4),
        "roughness_mm": round(noise_estimate(m), 4), "area_mm2": round(float(m.area), 1),
        "volume_mm3": round(float(m.volume), 1) if m.is_watertight else None,
        "has_vertex_colors": bool(getattr(m.visual, "kind", None) == "vertex"),
        "seconds": None,
    }
    render_views(m, out / f"{p.stem}.views.png", f"{p.name}: {r['triangles']} tris, bbox {r['bbox']} mm")
    r["seconds"] = round(time.time() - t, 1)
    rows[p.stem] = r
    print(p.stem, r, flush=True)

# same-part grouping: sorted-bbox similarity, then symmetric ICP-style distance after PCA alignment
def sample(m, n=6000):
    if isinstance(m, trimesh.PointCloud):
        V = np.asarray(m.vertices)
        return V[np.random.default_rng(0).choice(len(V), n, replace=False)]
    return np.asarray(trimesh.sample.sample_surface(m, n, seed=0)[0])


def pca_frames(P):
    c = P.mean(0)
    w, U = np.linalg.eigh(np.cov((P - c).T))
    return c, U[:, ::-1]


def icp_dist(A, B, iters=30):
    """Best of the 8 PCA sign flips + 4 axis permutations of B, refined by point-to-point ICP; returns
    symmetric median and p90 distance (mm) and the transform."""
    ca, Ua = pca_frames(A)
    cb, Ub = pca_frames(B)
    tA = cKDTree(A)
    best = None
    import itertools
    for perm in itertools.permutations(range(3)):
        for s in itertools.product((1, -1), repeat=3):
            R = Ua @ np.diag(s) @ Ub[:, perm].T
            if np.linalg.det(R) < 0:
                continue
            Bt = (B - cb) @ R.T + ca
            d, _ = tA.query(Bt)
            sc = float(np.median(d))
            if best is None or sc < best[0]:
                best = (sc, R, Bt)
    _, R, Bt = best
    for _ in range(iters):
        d, j = tA.query(Bt)
        keep = d < np.percentile(d, 90)
        P, Q = Bt[keep], A[j[keep]]
        pc, qc = P.mean(0), Q.mean(0)
        U, _, Vt = np.linalg.svd((P - pc).T @ (Q - qc))
        Ri = (U @ Vt).T
        if np.linalg.det(Ri) < 0:
            Vt[-1] *= -1
            Ri = (U @ Vt).T
        Bt = (Bt - pc) @ Ri.T + qc
    d1, _ = tA.query(Bt)
    d2, _ = cKDTree(Bt).query(A)
    d = np.r_[d1, d2]
    return float(np.median(d)), float(np.percentile(d, 90))


names = list(rows)
samples = {k: sample(meshes[k]) for k in names}
pairs = {}
for i, a in enumerate(names):
    for b in names[i + 1:]:
        ea, eb = np.sort(rows[a]["bbox"]), np.sort(rows[b]["bbox"])
        rel = float(np.max(np.abs(ea - eb) / np.maximum(ea, eb)))
        rec = {"bbox_rel_diff": round(rel, 3)}
        if rel < 0.25:
            med, p90 = icp_dist(samples[a], samples[b])
            rec.update({"icp_median_mm": round(med, 3), "icp_p90_mm": round(p90, 3)})
        pairs[f"{a}-{b}"] = rec
        print(a, b, rec, flush=True)
json.dump({"files": rows, "pairs": pairs}, open(out / "triage_raw.json", "w"), indent=1)

# contact sheet
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
ncol = 3
nrow = int(np.ceil(len(names) / ncol))
fig, axs = plt.subplots(nrow, ncol, figsize=(8 * ncol, 2.4 * nrow))
for ax in np.ravel(axs):
    ax.set_axis_off()
for ax, k in zip(np.ravel(axs), names):
    ax.imshow(mpimg.imread(out / f"{k}.views.png"))
fig.tight_layout()
fig.savefig(out / "contact_sheet.png", dpi=90, bbox_inches="tight")
print("done")
