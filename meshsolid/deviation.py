"""Deviation between the input mesh and the reconstructed solid."""
from __future__ import annotations

import numpy as np
import trimesh

from .occ_util import shape_to_trimesh


def _stats(d):
    a = np.abs(d)
    return {"max": float(a.max()), "mean": float(a.mean()), "rms": float(np.sqrt(np.mean(d * d))),
            "p95": float(np.percentile(a, 95)), "n": int(len(d))}


def _on_surface(target, P, chunk=2000, budget=1e6):
    """Exact closest points on `target` for points P, in chunks: trimesh's candidate arrays scale with
    (points x triangles whose AABB is near), which blew memory on long sliver triangles in one call, and
    with the point's distance (all triangles within that radius are candidates): points far from a densely
    tessellated surface (scan-only stubs on freeform parts) go in small chunks, sized from a KD-tree
    estimate of each point's candidate count."""
    from scipy.spatial import cKDTree
    from . import device as D
    if D.device() in ("cuda", "warp-cpu"):     # optional GPU path (M8b): exact float64 result on the BVH's face
        return D.closest_points(target.vertices, target.faces, P)
    try:                                       # libigl's AABB tree: exact, memory-light, independent of facet shape
        import igl
    except ImportError:
        igl = None
    if igl is not None:
        P = np.ascontiguousarray(np.asarray(P, float))
        d2, tri, cp = igl.point_mesh_squared_distance(P, np.ascontiguousarray(target.vertices, dtype=float),
                                                      np.ascontiguousarray(target.faces, dtype=np.int64))
        return np.asarray(cp), np.sqrt(np.maximum(np.asarray(d2), 0.0)), np.asarray(tri, int)
    # long facets (CAD exports) make every nearby point a candidate of every sliver whose box is near:
    # query on an exact subdivision (same surface) and map the triangle ids back
    fmap = None
    el = target.edges_unique_length
    # (edge limit also bounded below so the subdivision stays within ~1M triangles)
    lim = max(10.0 * float(np.median(el)), 1e-3 * float(np.linalg.norm(target.extents)), float(np.sqrt(4.0 * target.area / 1e6)))
    if el.max() > lim:
        # subdivide_to_size splits whole faces 4-way per level: a face whose longest edge is e becomes
        # 4^ceil(log2(e/lim)) faces.  Only subdivide when that stays under ~1.5M faces; otherwise the
        # chunked search below copes (slivers on long cylinders would multiply into tens of millions)
        emax = target.edges_unique_length[target.faces_unique_edges].max(axis=1)
        est = float(np.sum(4.0 ** np.ceil(np.log2(np.maximum(emax / lim, 1.0)))))
    if el.max() > lim and est <= 1.5e6:
        v2, f2, fmap = trimesh.remesh.subdivide_to_size(target.vertices, target.faces, lim, max_iter=12,
                                                         return_index=True)
        target = trimesh.Trimesh(v2, f2, process=False)
    pq = trimesh.proximity.ProximityQuery(target)
    P = np.asarray(P, float)
    cp, d, tri = np.empty((len(P), 3)), np.empty(len(P)), np.empty(len(P), int)
    tc = target.triangles_center
    h = max(float(np.median(target.edges_unique_length)), 1e-9)
    d0, _ = cKDTree(tc).query(P)
    cost = 1.0 + (d0 / h) ** 2 * 3.0           # ~triangles within the search sphere
    order = np.argsort(d0); P_sorted = P[order]; cs = np.cumsum(cost[order])
    res_cp, res_d, res_t = np.empty((len(P), 3)), np.empty(len(P)), np.empty(len(P), int)
    s0 = 0
    while s0 < len(P):
        base = cs[s0 - 1] if s0 else 0.0
        e = int(np.searchsorted(cs, base + budget, side="right"))
        e = min(max(e, s0 + 1), s0 + chunk, len(P))
        try:
            res_cp[s0:e], res_d[s0:e], res_t[s0:e] = pq.on_surface(P_sorted[s0:e])
        except MemoryError:
            if budget <= 1e4:
                raise
            budget /= 4
            continue
        s0 = e
    cp[order], d[order], tri[order] = res_cp, res_d, res_t
    if fmap is not None:
        tri = np.asarray(fmap)[tri]
    return cp, d, tri


def _n_open(mesh):
    import trimesh.grouping as G
    return len(G.group_rows(mesh.edges_sorted, require_count=1))


def _scan_coverage(mesh, Q, dr, trq, near=0.2):
    """Mask of solid samples Q that are covered by the scan (None for a closed scan)."""
    import trimesh.grouping as G
    from scipy.spatial import cKDTree
    bi = G.group_rows(mesh.edges_sorted, require_count=1)
    if len(bi) == 0:
        return None
    E = mesh.edges_sorted[bi]
    A, B = mesh.vertices[E[:, 0]], mesh.vertices[E[:, 1]]
    L = np.linalg.norm(B - A, axis=1)
    n = np.maximum(1, np.ceil(L / 0.05).astype(int))
    pts = np.concatenate([A[i] + np.linspace(0, 1, n[i] + 1)[:, None] * (B[i] - A[i]) for i in range(len(E))])
    cp, _, _ = _on_surface(mesh, Q)
    d_b = cKDTree(pts).query(cp)[0]
    return ~((d_b < near) & (dr > near))


FIELD_LEGEND = {
    "mesh_vertices_to_solid": "scan vertex -> solid surface distance, every scan vertex (signed, mm)",
    "mesh_vertices_to_solid_excl": "same, without the vertices of excluded zones - this is what the gate uses when present",
    "mesh_to_solid": "area-uniform samples on the scan -> solid (reference)",
    "mesh_to_solid_excl": "same, without excluded zones (reference)",
    "solid_to_mesh": "area-uniform samples on the solid -> scan (finds solid surface the scan does not support)",
    "solid_to_mesh_excl": "same, without excluded zones",
    "solid_to_mesh_scanned": "solid -> scan, only where the scan was not open (covered)",
}


def deviation(mesh: trimesh.Trimesh, shape, labels=None, n_samples=30000, seed=0, lin=0.002, exclude_faces=None, ang=0.05):
    """mesh->solid distances (per mesh vertex, per face, per region) and solid->mesh distances."""
    solid = shape_to_trimesh(shape, lin=lin, ang=ang)
    # per-vertex signed distance (positive = mesh outside solid)
    cp, dv, tri = _on_surface(solid, mesh.vertices)
    sgn = np.sign(np.einsum("ij,ij->i", mesh.vertices - cp, solid.face_normals[tri]))
    dv = dv * np.where(sgn == 0, 1, sgn)
    # area-uniform samples on mesh
    rng = np.random.default_rng(seed)
    P, fid = trimesh.sample.sample_surface(mesh, n_samples, seed=int(rng.integers(1 << 31)))
    _, ds, _ = _on_surface(solid, P)
    # reverse: samples on solid -> mesh
    Q, _ = trimesh.sample.sample_surface(solid, n_samples, seed=int(rng.integers(1 << 31)))
    _, dr, trq = _on_surface(mesh, Q)
    face_dev = np.abs(dv[mesh.faces]).max(1)
    cov = _scan_coverage(mesh, Q, dr, trq)
    out = {"mesh_vertices_to_solid": _stats(dv), "mesh_to_solid": _stats(ds), "solid_to_mesh": _stats(dr),
           "hausdorff": float(max(ds.max(), dr.max())),
           "solid_tessellation": {"triangles": int(len(solid.faces)), "linear_deflection": float(lin)}}
    if cov is not None:
        # open scan: solid surface whose nearest scan point is on the scan's open boundary was not
        # scanned (synthesised or hidden faces); it cannot be judged against the scan
        out["solid_scan_coverage"] = {"covered_frac": float(cov.mean()), "open_boundary_edges": int(_n_open(mesh))}
        if cov.sum() > 10:
            out["solid_to_mesh_scanned"] = _stats(dr[cov])
    if exclude_faces is not None and exclude_faces.any():
        # zones the solid models only by a feature spec (threads): stats without them, used for acceptance
        ev = np.zeros(len(mesh.vertices), bool)
        ev[mesh.faces[exclude_faces].ravel()] = True
        keep_v, keep_s, keep_r = ~ev, ~exclude_faces[fid], ~exclude_faces[trq]
        out["excluded_zones"] = {"faces": int(exclude_faces.sum()),
                                 "area_frac": float(mesh.area_faces[exclude_faces].sum() / mesh.area)}
        if keep_v.sum() > 10 and keep_s.sum() > 10 and keep_r.sum() > 10:
            out["mesh_vertices_to_solid_excl"] = _stats(dv[keep_v])
            out["mesh_to_solid_excl"] = _stats(ds[keep_s])
            out["solid_to_mesh_excl"] = _stats(dr[keep_r])
    out["field_legend"] = FIELD_LEGEND
    gate_key = "mesh_vertices_to_solid_excl" if "mesh_vertices_to_solid_excl" in out else "mesh_vertices_to_solid"
    out["summary"] = {
        "vertex_to_solid_rms_all_mm": round(out["mesh_vertices_to_solid"]["rms"], 5),
        "vertex_to_solid_rms_gate_mm": round(out[gate_key]["rms"], 5),
        "vertex_to_solid_p95_gate_mm": round(out[gate_key]["p95"], 5),
        "gate_field": gate_key,
        "note": ("the acceptance gate reads gate_field; *_all includes the excluded zones (threads, named "
                 "exclusions) and is reported for reference only"),
    }
    per_region = {}
    if labels is not None:
        for lab in np.unique(labels):
            sel = labels[fid] == lab
            if sel.sum() == 0:
                continue
            per_region[int(lab)] = _stats(ds[sel])
    return out, per_region, dv, face_dev, solid


def save_deviation_ply(mesh, dv, path, vmax=None):
    """Colour map PLY of signed per-vertex deviation (blue=-vmax, green=0, red=+vmax)."""
    import matplotlib
    vmax = vmax or max(float(np.percentile(np.abs(dv), 99)), 1e-6)
    cmap = matplotlib.colormaps["jet"]
    col = (cmap(np.clip((dv / vmax + 1) / 2, 0, 1)) * 255).astype(np.uint8)
    m = trimesh.Trimesh(mesh.vertices, mesh.faces, vertex_colors=col, process=False)
    m.export(path)
    return vmax
