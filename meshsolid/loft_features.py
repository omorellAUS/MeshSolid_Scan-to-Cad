"""Outer-skin loft + discrete pocket features for skins with non-loftable recesses.

WING 2 L: ~92% of loft SSE is two on-skin pockets in the root half. Skin-only rms
excluding them ~0.19. Pipeline:

  1. Loft outer skin (smooth shared-wire).
  2. Cluster high mesh→solid residuals; grow each cluster to a crease-bounded face region.
  3. Fit floor plane (+ optional walls); rim loop = region boundary on the mesh.
  4. Extrude rim along signed inward direction by measured depth → BOP Cut.
  5. Validate single solid; score full and skin-only.
"""
from __future__ import annotations

import math
import time
from typing import Sequence

import numpy as np


def cluster_high_residual(V: np.ndarray, dists: np.ndarray, thr: float | None = None,
                          link: float | None = None, extents_norm: float | None = None):
    """DBSCAN-like clusters of verts with dists >= thr. Returns list of index arrays."""
    from scipy.spatial import cKDTree
    thr = float(thr if thr is not None else max(0.35, np.percentile(dists, 75)))
    link = float(link if link is not None else max(
        3.0, 0.02 * (extents_norm or float(np.linalg.norm(V.ptp(0))))))
    hi = np.where(dists >= thr)[0]
    if len(hi) == 0:
        return [], thr, link
    tree = cKDTree(V[hi])
    visited = set()
    clusters = []
    for i0 in range(len(hi)):
        if i0 in visited:
            continue
        stack = [i0]
        visited.add(i0)
        members = []
        while stack:
            j = stack.pop()
            members.append(int(hi[j]))
            for k in tree.query_ball_point(V[hi[j]], link):
                if k not in visited:
                    visited.add(k)
                    stack.append(k)
        clusters.append(np.asarray(members, int))
    clusters.sort(key=lambda m: -(float(np.sqrt((dists[m] ** 2).mean())) * math.sqrt(len(m))))
    return clusters, thr, link


def classify_cluster(pts: np.ndarray, d_span: np.ndarray, h_lo: float, h_hi: float):
    """Crude shape guess for a residual cluster."""
    c = pts.mean(0)
    X = pts - c
    evals, evecs = np.linalg.eigh(X.T @ X / max(len(pts), 1))
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]
    extents = np.sqrt(np.maximum(evals, 0.0)) * 2
    axis = evecs[:, int(np.argmin(extents))] if extents.min() < 0.35 * extents.max() else evecs[:, 0]
    w = X - np.outer(X @ axis, axis)
    r = np.linalg.norm(w, axis=1)
    r_med = float(np.median(r)) if len(r) else 0.0
    r_std = float(np.std(r)) if len(r) else 0.0
    along = X @ axis
    len_along = float(along.max() - along.min()) if len(along) else 0.0
    span = h_hi - h_lo
    h_c = float(c @ d_span)
    shape = "blob"
    if len(pts) >= 8 and r_med > 1.0 and r_std < 0.35 * max(r_med, 1e-6) and len_along > 2.0:
        shape = "cylinder_shell"
    elif len(pts) >= 6 and extents.min() < 1.5 and extents.max() > 5 and extents[1] > 3:
        shape = "thin_slot_or_rib"
    elif len(pts) >= 6 and extents.max() < 8 and r_med < 5:
        shape = "pocket_or_boss"
    elif abs(h_c - h_lo) < 0.15 * span or abs(h_c - h_hi) < 0.15 * span:
        shape = "root_or_tip_feature"
    elif h_c < h_lo + 0.45 * span:
        shape = "root_half_feature"
    return {
        "shape": shape, "centroid": c, "h_span": h_c, "extents_pca": extents,
        "axis": axis, "r_med": r_med, "r_std": r_std, "len_along": len_along,
    }


def skin_mask_excluding(clusters: Sequence[np.ndarray], n_verts: int, top_k: int = 2,
                        dists: np.ndarray | None = None, outlier_max: float = 2.0):
    """Boolean skin mask: True = keep for skin-only score."""
    feat = set()
    for mem in list(clusters)[:top_k]:
        feat.update(int(i) for i in mem)
    if dists is not None:
        for mem in clusters:
            if float(dists[mem].max()) > outlier_max:
                feat.update(int(i) for i in mem)
    mask = np.ones(n_verts, bool)
    if feat:
        mask[list(feat)] = False
    return mask, feat


def _face_graph(mesh):
    """Undirected face adjacency list + dihedral angle (deg) per edge."""
    adj = mesh.face_adjacency
    ang = np.degrees(mesh.face_adjacency_angles)
    n = len(mesh.faces)
    g = [[] for _ in range(n)]
    for (a, b), th in zip(adj, ang):
        g[int(a)].append((int(b), float(th)))
        g[int(b)].append((int(a), float(th)))
    return g


def grow_pocket_faces(mesh, seed_verts, dists, crease_deg=35.0, max_faces=120,
                      max_radius=None):
    """Grow a face region from seed verts until a crease (dihedral) boundary.

    Tight growth: only high-residual faces, stop at modest creases, ball-limited
    around the seed centroid so one pocket does not swallow the whole skin.
    """
    V = np.asarray(mesh.vertices, float)
    F = np.asarray(mesh.faces, int)
    seed = set(int(i) for i in seed_verts)
    face_dist = dists[F].mean(1)
    seeds = []
    for fi, face in enumerate(F):
        n_hit = sum(1 for v in face if int(v) in seed)
        if n_hit >= 2 or (n_hit >= 1 and face_dist[fi] >= 0.35):
            seeds.append(fi)
    if not seeds:
        c = V[list(seed)].mean(0) if seed else V.mean(0)
        d = np.linalg.norm(mesh.triangles_center - c, axis=1)
        seeds = list(np.argsort(d)[: max(3, min(12, len(F)))])
    c = V[list(seed)].mean(0) if seed else mesh.triangles_center[seeds].mean(0)
    if max_radius is None:
        # cluster span + margin
        if seed:
            max_radius = float(np.linalg.norm(V[list(seed)] - c, axis=1).max()) + 3.0
        else:
            max_radius = 12.0
        max_radius = float(np.clip(max_radius, 6.0, 25.0))
    g = _face_graph(mesh)
    thr_keep = max(0.25, float(np.percentile(dists[list(seed)], 20)) * 0.7) if seed else 0.25
    centers = mesh.triangles_center
    region = set()
    stack = list(seeds)
    while stack and len(region) < max_faces:
        f = stack.pop()
        if f in region:
            continue
        if float(np.linalg.norm(centers[f] - c)) > max_radius:
            continue
        if face_dist[f] < thr_keep and f not in seeds:
            continue
        region.add(f)
        for nb, th in g[f]:
            if nb in region:
                continue
            if float(np.linalg.norm(centers[nb] - c)) > max_radius:
                continue
            # hard stop on sharp crease into low-residual skin
            if th >= crease_deg and face_dist[nb] < thr_keep:
                continue
            if face_dist[nb] < thr_keep:
                continue
            stack.append(nb)
    return np.asarray(sorted(region), int)


def _region_boundary_loops(mesh, face_ids):
    """Oriented boundary loops (vertex index cycles) of a face region."""
    F = np.asarray(mesh.faces, int)
    region = set(int(i) for i in face_ids)
    # count undirected edges
    from collections import defaultdict
    edge_faces = defaultdict(list)
    for fi in region:
        a, b, c = map(int, F[fi])
        for e in ((a, b), (b, c), (c, a)):
            edge_faces[tuple(sorted(e))].append(fi)
    boundary = [e for e, fs in edge_faces.items() if len(fs) == 1]
    if not boundary:
        return []
    # adjacency of boundary verts
    adj = defaultdict(list)
    for a, b in boundary:
        adj[a].append(b)
        adj[b].append(a)
    loops = []
    unused = set(boundary)
    # walk
    while unused:
        a0, b0 = next(iter(unused))
        loop = [a0]
        prev, cur = a0, b0
        unused.discard((a0, b0))
        guard = 0
        while cur != a0 and guard < 10000:
            loop.append(cur)
            guard += 1
            nxts = [x for x in adj[cur] if x != prev]
            if not nxts:
                break
            # prefer unused edge
            chosen = None
            for n in nxts:
                e = tuple(sorted((cur, n)))
                if e in unused:
                    chosen = n
                    break
            if chosen is None:
                chosen = nxts[0]
            e = tuple(sorted((cur, chosen)))
            unused.discard(e)
            prev, cur = cur, chosen
        if len(loop) >= 3:
            loops.append(np.asarray(loop, int))
    loops.sort(key=len, reverse=True)
    return loops


def fit_pocket_floor_and_walls(mesh, face_ids, floor_angle_deg=35.0, preferred_normal=None):
    """Split region into floor (dominant plane) and wall faces; fit planes.

    Floor = faces whose normals align with `preferred_normal` (loft-skin outward from
    cavity) or, if None, the region's area-weighted mean normal. Walls = the rest.
    """
    from meshsolid.fit import fit_plane
    F = np.asarray(mesh.faces, int)
    V = np.asarray(mesh.vertices, float)
    N = np.asarray(mesh.face_normals, float)
    A = np.asarray(mesh.area_faces, float)
    ids = np.asarray(face_ids, int)
    if len(ids) == 0:
        return None
    w = A[ids]
    if preferred_normal is not None:
        n_mean = np.asarray(preferred_normal, float)
        n_mean = n_mean / (np.linalg.norm(n_mean) + 1e-12)
    else:
        n_mean = (N[ids] * w[:, None]).sum(0)
        n_mean = n_mean / (np.linalg.norm(n_mean) + 1e-12)
    # floor candidates: normal ~ n_mean
    dots = N[ids] @ n_mean
    floor_mask = dots >= math.cos(math.radians(floor_angle_deg))
    if floor_mask.sum() < 3:
        # take top third by alignment
        k = max(3, len(ids) // 3)
        floor_mask = np.zeros(len(ids), bool)
        floor_mask[np.argsort(-dots)[:k]] = True
    floor_ids = ids[floor_mask]
    wall_ids = ids[~floor_mask]
    # fit floor
    P_floor = V[F[floor_ids]].reshape(-1, 3)
    N_floor = np.repeat(N[floor_ids], 3, axis=0)
    floor = fit_plane(P_floor, N_floor)
    # ensure floor normal points toward skin opening (same as n_mean)
    fn = np.asarray(floor.params["normal"], float)
    if fn @ n_mean < 0:
        floor.params["normal"] = -fn
        fn = -fn
    walls = []
    if len(wall_ids) >= 3:
        # cluster wall normals
        Wn = N[wall_ids]
        used = np.zeros(len(wall_ids), bool)
        for _ in range(6):
            if used.all():
                break
            # seed = unused face with largest area
            rem = np.where(~used)[0]
            seed = rem[np.argmax(A[wall_ids[rem]])]
            ref = Wn[seed]
            sel = (~used) & (np.abs(Wn @ ref) >= math.cos(math.radians(25)))
            if sel.sum() < 2:
                used[seed] = True
                continue
            used |= sel
            ids_w = wall_ids[sel]
            Pw = V[F[ids_w]].reshape(-1, 3)
            Nw = np.repeat(N[ids_w], 3, axis=0)
            walls.append(fit_plane(Pw, Nw))
    return {
        "floor": floor,
        "walls": walls,
        "floor_ids": floor_ids,
        "wall_ids": wall_ids,
        "n_mean": n_mean,
        "floor_normal": np.asarray(floor.params["normal"], float),
        "floor_point": np.asarray(floor.params["point"], float),
    }


def _ordered_rim_points(mesh, loop_verts, n_target=48):
    """Resample a closed vertex loop to n_target points along arc length."""
    V = np.asarray(mesh.vertices, float)
    P = V[loop_verts]
    if np.linalg.norm(P[0] - P[-1]) > 1e-9:
        P = np.r_[P, P[:1]]
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(seg)]
    if s[-1] <= 0:
        return P[:-1]
    t = np.linspace(0.0, s[-1], n_target + 1)[:-1]
    return np.c_[np.interp(t, s, P[:, 0]), np.interp(t, s, P[:, 1]), np.interp(t, s, P[:, 2])]


def build_pocket_tool(mesh, face_ids, dists, cp_on_solid, depth_pad=0.35, rim_pad=-0.10,
                      n_rim=64, multi_station=False, n_stations=5):
    """Build a solid tool that fills the lofted material inside the pocket cavity.

    Rim prefers the region's mesh boundary loop (inset by |rim_pad| when negative)
    over a convex hull — hulls over-cover and split the solid (nsol=2). Depth is
    clamped by local wall thickness along -n so the tool cannot breach the opposite
    skin. When multi_station=True, loft a rail of section polygons along the pocket
    long axis (better for elongated servo cutouts) instead of a single prism.

    `cp_on_solid`: closest points on the loft solid for each mesh vertex (same
    order as mesh.vertices) — used to estimate how much loft sits above the
    pocket (should match depth).
    """
    from OCP.gp import gp_Pnt, gp_Vec
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakePolygon, BRepBuilderAPI_MakeFace
    from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
    from OCP.BRepCheck import BRepCheck_Analyzer

    V = np.asarray(mesh.vertices, float)
    F = np.asarray(mesh.faces, int)
    reg_verts = np.unique(F[face_ids].ravel())
    P = V[reg_verts]
    # Preferred cut direction: mesh outward normals on the pocket (point into the
    # cavity toward the opening / loft skin). Do NOT average (cp-V) — directions
    # cancel on a bowl and leave a span-wise noise vector.
    pref = np.asarray(mesh.vertex_normals[reg_verts], float).mean(0)
    if np.linalg.norm(pref) < 1e-9:
        pref = np.asarray(mesh.face_normals[face_ids], float).mean(0)
    pref = pref / (np.linalg.norm(pref) + 1e-12)
    fit = fit_pocket_floor_and_walls(mesh, face_ids, preferred_normal=pref)
    if fit is None:
        return None, {"ok": False, "reason": "no fit"}
    fn = np.asarray(pref, float)  # trust loft-skin direction over noisy floor fit normal
    fp = np.asarray(fit["floor_point"], float)
    # Re-fit floor point as mean of deepest verts (most opposite to pref)
    deep = np.argsort(P @ fn)[: max(5, len(P) // 5)]
    fp = P[deep].mean(0)
    mean_side = float(np.mean((cp_on_solid[reg_verts] - V[reg_verts]) @ fn))
    if mean_side < 0:
        fn = -fn
        mean_side = -mean_side
    # Opening centre: loft closest-points of high-residual verts (the skin bridge the
    # loft lays over the cavity). Mesh outward-most verts sit on the pocket lip/floor
    # and produced tools that removed volume elsewhere without lowering seed rms.
    hi_open = reg_verts[dists[reg_verts] >= max(0.3, float(np.percentile(dists[reg_verts], 40)))]
    if len(hi_open) < 6:
        hi_open = reg_verts
    rim_c = np.asarray(cp_on_solid[hi_open], float).mean(0)
    # Fallback if cp collapsed: mesh outward-most
    if not np.isfinite(rim_c).all() or float(np.linalg.norm(rim_c - P.mean(0))) > 50:
        s = P @ fn
        open_idx = np.argsort(-s)[: max(8, len(P) // 4)]
        rim_c = P[open_idx].mean(0)
    from scipy.spatial import ConvexHull
    a = np.array([1.0, 0.0, 0.0]) if abs(fn[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(fn, a); e1 /= (np.linalg.norm(e1) + 1e-12)
    e2 = np.cross(fn, e1)
    # Prefer mesh boundary loop of the grown region (true opening rim). Fall back to
    # high-residual convex hull only if no loop. Negative rim_pad = inset (non-split).
    hi = reg_verts[dists[reg_verts] >= max(0.3, float(np.percentile(dists[reg_verts], 40)))]
    if len(hi) < 6:
        hi = reg_verts
    # Prefer loft cps for outline (bridge footprint); mesh verts as fallback.
    Ph = np.asarray(cp_on_solid[hi], float)
    rim2 = None
    rim_src = "hull_cp"
    loops = _region_boundary_loops(mesh, face_ids)
    if loops:
        # project mesh boundary into opening plane but keep loft centre
        Ploop = _ordered_rim_points(mesh, loops[0], n_target=max(16, int(n_rim)))
        UV_loop = np.c_[(Ploop - rim_c) @ e1, (Ploop - rim_c) @ e2]
        if len(UV_loop) >= 3:
            rim2 = UV_loop
            rim_src = "boundary"
    if rim2 is None:
        UV = np.c_[(Ph - rim_c) @ e1, (Ph - rim_c) @ e2]
        try:
            hull = ConvexHull(UV)
            rim2 = UV[hull.vertices]
        except Exception:
            return None, {"ok": False, "reason": "hull failed"}
    c2 = rim2.mean(0)
    mean_r = float(np.linalg.norm(rim2 - c2, axis=1).mean()) + 1e-9
    scale = 1.0 + float(rim_pad) / mean_r
    scale = float(np.clip(scale, 0.55, 1.35))
    rim2 = c2 + (rim2 - c2) * scale
    # Depth from loft-vs-mesh along fn (primary). Floor-plane depth is a sanity check only —
    # it overestimates when the "floor" fit is contaminated by walls.
    depth_mesh = float(np.percentile(np.abs((cp_on_solid[reg_verts] - V[reg_verts]) @ fn), 90))
    depth_dist = float(np.percentile(dists[reg_verts], 90))
    depth_floor = float((rim_c - fp) @ fn)
    thick_min = float("nan")
    depth = max(depth_mesh, depth_dist, 0.5)
    # If floor depth is in a plausible band, allow it to raise depth slightly
    if 0.5 < depth_floor < depth + 1.5:
        depth = max(depth, 0.85 * depth_floor)
    depth = float(min(depth, 3.5))  # hard cap — servo pockets on this wing are ~1–3 mm
    # Non-splitting: clamp depth by local wall thickness from the mesh.
    # Sample from opening plane (rim) along -fn. Prefer the *shortest* near-rim
    # thickness (opposite skin or cavity back-wall). Cap use at 8 mm so a through-
    # wing ray (~15 mm) does not silently disable the clamp.
    open_pts = np.array([rim_c + u * e1 + v * e2 for u, v in rim2[:: max(1, len(rim2) // 10)]])
    open_pts = np.vstack([rim_c[None, :], open_pts])
    # opposite skin (skip cavity floor); also sample remaining wall from deep floor verts
    thick_opp = wall_thickness_along(mesh, open_pts, -fn, max_t=40.0, which="opposite")
    deep_pts = P[np.argsort(P @ fn)[: max(5, len(P) // 5)]]
    thick_rem = wall_thickness_along(mesh, deep_pts, -fn, max_t=40.0, which="first")
    thick_all = np.r_[thick_opp, thick_rem]
    thick_ok = thick_all[np.isfinite(thick_all) & (thick_all > 0.3)]
    thick_min = float(np.percentile(thick_ok, 20)) if len(thick_ok) else np.nan
    # Non-split: tool depth + remnant <= opposite thickness. Remnant ~ max(0.8, 0.25*thick).
    thick_use = float(thick_min) if np.isfinite(thick_min) else np.nan
    if np.isfinite(thick_use):
        remnant = max(0.8, 0.20 * thick_use)
        depth = float(min(depth, max(0.4, thick_use - remnant - float(depth_pad))))
    if depth < 0.25:
        return None, {"ok": False, "reason": f"depth too small {depth:.3f}",
                      "depth_floor": depth_floor, "depth_mesh": depth_mesh}
    depth_use = depth + float(depth_pad)
    tool = None
    tool_mode = "prism"
    # Multi-station rails along pocket long axis in the opening plane
    if multi_station and len(rim2) >= 6 and n_stations >= 3:
        try:
            tool = _pocket_tool_multi_station(
                rim2, rim_c, e1, e2, fn, depth_use, depth_pad, n_stations=int(n_stations))
            if tool is not None:
                tool_mode = "multi_station"
        except Exception:
            tool = None
    if tool is None:
        opening_p = np.array([rim_c + u * e1 + v * e2 + fn * float(depth_pad) for u, v in rim2])
        area2 = 0.0
        for i in range(len(opening_p)):
            aa, bb = opening_p[i] - rim_c, opening_p[(i + 1) % len(opening_p)] - rim_c
            area2 += float(np.dot(fn, np.cross(aa, bb)))
        if area2 < 0:
            opening_p = opening_p[::-1]
        from OCP.gp import gp_Pln, gp_Ax3, gp_Dir
        origin = rim_c + fn * float(depth_pad)
        pln = gp_Pln(gp_Ax3(gp_Pnt(float(origin[0]), float(origin[1]), float(origin[2])),
                           gp_Dir(float(fn[0]), float(fn[1]), float(fn[2]))))
        mw = BRepBuilderAPI_MakePolygon()
        for p in opening_p:
            mw.Add(gp_Pnt(float(p[0]), float(p[1]), float(p[2])))
        mw.Close()
        if not mw.IsDone():
            return None, {"ok": False, "reason": "rim wire failed"}
        mf = BRepBuilderAPI_MakeFace(pln, mw.Wire())
        if not mf.IsDone():
            return None, {"ok": False, "reason": "rim face not done"}
        face = mf.Face()
        if face.IsNull() or not BRepCheck_Analyzer(face).IsValid():
            return None, {"ok": False, "reason": "rim face invalid"}
        vec = gp_Vec(float(-fn[0] * depth_use), float(-fn[1] * depth_use), float(-fn[2] * depth_use))
        tool = BRepPrimAPI_MakePrism(face, vec).Shape()
        tool_mode = "prism"
    # Lateral containment: Common with a local OBB around the pocket so the tool
    # cannot reach past the cavity and split the wing (nsol=2).
    try:
        tool2 = _clamp_tool_to_pocket_obb(tool, P, fn, pad=1.2)
        if tool2 is not None:
            tool = tool2
            tool_mode = tool_mode + "+obb"
    except Exception:
        pass
    info = {
        "ok": True,
        "n_faces": int(len(face_ids)),
        "n_floor": int(len(fit["floor_ids"])),
        "n_walls": int(len(fit["wall_ids"])),
        "n_wall_planes": len(fit["walls"]),
        "rim_n": int(len(rim2)),
        "rim_src": rim_src,
        "tool_mode": tool_mode,
        "depth": round(float(depth), 4),
        "depth_use": round(float(depth_use), 4),
        "depth_floor": round(float(depth_floor), 4),
        "depth_mesh": round(float(depth_mesh), 4),
        "thick_min": round(float(thick_min), 4) if np.isfinite(thick_min) else None,
        "thick_use": round(float(thick_use), 4) if np.isfinite(thick_use) else None,
        "mean_side": round(float(mean_side), 4),
        "floor_normal": [round(float(x), 5) for x in fn],
        "floor_rms": round(float(fit["floor"].rms), 5) if fit["floor"].rms is not None else None,
        "centroid": [round(float(x), 2) for x in rim_c],
    }
    return tool, info


def cut_pockets_from_loft(mesh, shape, dists, cp_on_solid, clusters, top_k=3,
                          crease_deg=45.0, min_cluster_n=8, t_stop=None):
    """Grow pockets from top residual clusters, cut tools from `shape`. Returns (shape, log)."""
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    from OCP.TopAbs import TopAbs_SOLID
    from OCP.TopExp import TopExp_Explorer

    log = []
    cur = shape
    mesh_vol = float(mesh.volume)
    for i, mem in enumerate(list(clusters)[:top_k]):
        if t_stop is not None and time.time() > t_stop:
            log.append({"i": i, "ok": False, "reason": "time budget"})
            break
        if len(mem) < min_cluster_n:
            log.append({"i": i, "ok": False, "reason": f"cluster too small n={len(mem)}"})
            continue
        faces = grow_pocket_faces(mesh, mem, dists, crease_deg=crease_deg, max_faces=160)
        if len(faces) < 4:
            log.append({"i": i, "ok": False, "reason": f"few faces {len(faces)}", "n_seed": int(len(mem))})
            continue
        attempts = [
            dict(multi_station=False, rim_pad=-0.05, depth_pad=0.45, n_stations=5),
            dict(multi_station=False, rim_pad=-0.10, depth_pad=0.35, n_stations=5),
            dict(multi_station=False, rim_pad=-0.25, depth_pad=0.30, n_stations=3),
            dict(multi_station=True, rim_pad=-0.10, depth_pad=0.40, n_stations=7),
        ]
        accepted = False
        last_info = None
        for ai, kwargs in enumerate(attempts):
            tool, info = build_pocket_tool(mesh, faces, dists, cp_on_solid, **kwargs)
            info = dict(info)
            info["i"] = i
            info["attempt"] = ai
            info["n_seed"] = int(len(mem))
            info["n_region_faces"] = int(len(faces))
            last_info = info
            if not info.get("ok"):
                continue
            try:
                cut = BRepAlgoAPI_Cut(cur, tool)
                cut.Build()
                if not cut.IsDone():
                    info["ok"] = False
                    info["reason"] = "cut not done"
                    continue
                sh = cut.Shape()
                if not BRepCheck_Analyzer(sh).IsValid():
                    info["ok"] = False
                    info["reason"] = "invalid after cut"
                    continue
                ex = TopExp_Explorer(sh, TopAbs_SOLID)
                nsol = 0
                while ex.More():
                    nsol += 1
                    ex.Next()
                pg = GProp_GProps()
                BRepGProp.VolumeProperties_s(sh, pg)
                vol = float(pg.Mass())
                vr = vol / max(mesh_vol, 1e-12)
                info["n_solids"] = nsol
                info["vol_ratio"] = round(vr, 4)
                if nsol != 1 or not (0.75 < vr < 1.15):
                    info["ok"] = False
                    info["reason"] = f"nsol={nsol} vol={vr:.3f}"
                    continue
                cur = sh
                info["ok"] = True
                log.append(info)
                accepted = True
                break
            except Exception as e:
                info["ok"] = False
                info["reason"] = str(e)[:160]
                continue
        if not accepted:
            if last_info is None:
                last_info = {"i": i, "ok": False, "reason": "no attempts"}
            log.append(last_info)
    return cur, log



def _pocket_tool_multi_station(rim2, rim_c, e1, e2, fn, depth_use, depth_pad, n_stations=5):
    """Fuse station prisms along the rim long axis (elongated pocket rail)."""
    from OCP.gp import gp_Pnt, gp_Vec, gp_Pln, gp_Ax3, gp_Dir
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakePolygon, BRepBuilderAPI_MakeFace
    from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse
    from OCP.BRepCheck import BRepCheck_Analyzer

    UV = np.asarray(rim2, float)
    c = UV.mean(0)
    X = UV - c
    cov = X.T @ X / max(len(UV), 1)
    evals, evecs = np.linalg.eigh(cov)
    major = evecs[:, int(np.argmax(evals))]
    minor = evecs[:, int(np.argmin(evals))]
    along = X @ major
    a0, a1 = float(along.min()), float(along.max())
    if a1 - a0 < 1e-3:
        return None
    solids = []
    ts = np.linspace(a0, a1, int(n_stations))
    half_span = 0.55 * (a1 - a0) / max(n_stations - 1, 1)
    for t in ts:
        sel = np.abs(along - t) <= max(half_span, 0.8)
        nearby = np.abs(along - t) <= max(half_span * 2, 1.5)
        use = sel if sel.sum() >= 2 else nearby
        if use.sum() < 2:
            continue
        w = float(np.max(np.abs(X[use] @ minor))) + 0.3
        corners = np.array([
            c + major * (t - half_span) + minor * w,
            c + major * (t + half_span) + minor * w,
            c + major * (t + half_span) - minor * w,
            c + major * (t - half_span) - minor * w,
        ])
        opening_p = np.array([rim_c + u * e1 + v * e2 + fn * float(depth_pad)
                              for u, v in corners])
        area2 = 0.0
        for i in range(4):
            aa, bb = opening_p[i] - rim_c, opening_p[(i + 1) % 4] - rim_c
            area2 += float(np.dot(fn, np.cross(aa, bb)))
        if area2 < 0:
            opening_p = opening_p[::-1]
        origin = opening_p.mean(0)
        pln = gp_Pln(gp_Ax3(gp_Pnt(float(origin[0]), float(origin[1]), float(origin[2])),
                           gp_Dir(float(fn[0]), float(fn[1]), float(fn[2]))))
        mw = BRepBuilderAPI_MakePolygon()
        for pt in opening_p:
            mw.Add(gp_Pnt(float(pt[0]), float(pt[1]), float(pt[2])))
        mw.Close()
        if not mw.IsDone():
            continue
        mf = BRepBuilderAPI_MakeFace(pln, mw.Wire())
        if not mf.IsDone():
            continue
        face = mf.Face()
        if face.IsNull() or not BRepCheck_Analyzer(face).IsValid():
            continue
        vec = gp_Vec(float(-fn[0] * depth_use), float(-fn[1] * depth_use),
                     float(-fn[2] * depth_use))
        solids.append(BRepPrimAPI_MakePrism(face, vec).Shape())
    if not solids:
        return None
    out = solids[0]
    for s in solids[1:]:
        fuse = BRepAlgoAPI_Fuse(out, s)
        fuse.Build()
        if fuse.IsDone():
            out = fuse.Shape()
    return out


def _clamp_tool_to_pocket_obb(tool, region_pts, fn, pad=1.5, depth_extra=2.5):
    """Intersect tool with a world-axis AABB around the pocket (expanded along ±fn).

    Oriented-box transform was fragile (zero-volume Common). AABB is enough to keep
    elongated multi-station tools from reaching past the cavity and splitting the solid.
    """
    from OCP.gp import gp_Pnt
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Common

    P = np.asarray(region_pts, float)
    fn = np.asarray(fn, float)
    fn = fn / (np.linalg.norm(fn) + 1e-12)
    lo = P.min(0) - pad
    hi = P.max(0) + pad
    lo = lo - np.abs(fn) * depth_extra
    hi = hi + np.abs(fn) * 0.6
    a = np.minimum(lo, hi)
    b = np.maximum(lo, hi)
    box = BRepPrimAPI_MakeBox(
        gp_Pnt(float(a[0]), float(a[1]), float(a[2])),
        gp_Pnt(float(b[0]), float(b[1]), float(b[2]))).Shape()
    common = BRepAlgoAPI_Common(tool, box)
    common.Build()
    if not common.IsDone():
        return None
    return common.Shape()


def wall_thickness_along(mesh, origins, direction, max_t=50.0, which="opposite"):
    """Local wall thickness: ray from origins along direction through the mesh.

    which:
      "first" — first hit (cavity floor when casting from opening into a pocket)
      "opposite" — prefer second hit (= opposite skin when first hit is the pocket
        floor); fall back to first if only one hit. This is the non-splitting clamp
        for pocket tools (must not breach the opposite skin).
    """
    d = np.asarray(direction, float)
    d = d / (np.linalg.norm(d) + 1e-12)
    origins = np.asarray(origins, float).reshape(-1, 3)
    try:
        locs, ri, tri = mesh.ray.intersects_location(
            origins + d[None, :] * 1e-4, np.repeat(d[None, :], len(origins), 0), multiple_hits=True)
    except Exception:
        return np.full(len(origins), np.nan)
    out = np.full(len(origins), np.nan)
    if len(ri) == 0:
        return out
    for i in range(len(origins)):
        hits = locs[ri == i]
        if len(hits) == 0:
            continue
        s = np.sort((hits - origins[i]) @ d)
        s = s[s > 1e-6]
        if len(s) == 0:
            continue
        if which == "opposite" and len(s) >= 2:
            out[i] = float(s[1])  # skip cavity floor → opposite skin
        else:
            out[i] = float(s[0])
        if out[i] > max_t:
            out[i] = np.nan
    return out


def zone_masks(mesh, d_span, chord_frac_te=0.12, chord_frac_le=0.18):
    """Vertex masks: TE_band, LE_band, mid_skin, tip_5pct (along span)."""
    V = np.asarray(mesh.vertices, float)
    d = np.asarray(d_span, float); d = d / (np.linalg.norm(d) + 1e-12)
    # chord plane axes
    a = np.array([1.0, 0, 0]) if abs(d[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(d, a); u /= np.linalg.norm(u)
    v = np.cross(d, u)
    # project to (chord≈major in u-v of sections). Use PCA of verts orthogonal to span.
    X = V - V.mean(0)
    X = X - np.outer(X @ d, d)
    evals, evecs = np.linalg.eigh(X.T @ X / max(len(V), 1))
    chord = evecs[:, int(np.argmax(evals))]
    c = V @ chord
    c0, c1 = float(c.min()), float(c.max())
    # TE = min chord end, LE = max (convention may flip — use both ends as TE/LE by name after)
    # Match prior zone maps: TE sharp end often min or max — label by which has sharper turns later.
    # Here: LE_band = verts near the end farther from TE seam hint: use max|c| extreme with
    # smaller local radius of curvature approx = end with fewer verts density... Simple:
    # LE = verts in top chord_frac of chord range from the CHORD MAX end; TE from MIN end.
    # (WINGTIP maps used this convention in prior probes.)
    te = c <= c0 + chord_frac_te * (c1 - c0)
    le = c >= c1 - chord_frac_le * (c1 - c0)
    mid = ~(te | le)
    h = V @ d
    h0, h1 = float(h.min()), float(h.max())
    tip = (h >= h1 - 0.05 * (h1 - h0)) | (h <= h0 + 0.05 * (h1 - h0))
    return {"TE_band": te, "LE_band": le, "mid_skin": mid, "tip_5pct": tip, "chord_dir": chord}


def zone_rms(dists, masks):
    out = {}
    for k, m in masks.items():
        if k == "chord_dir":
            continue
        if m is None or not np.any(m):
            continue
        dd = dists[m]
        out[k] = {
            "n": int(m.sum()),
            "rms": float(np.sqrt((dd ** 2).mean())),
            "p95": float(np.percentile(dd, 95)),
            "max": float(dd.max()),
        }
    return out
