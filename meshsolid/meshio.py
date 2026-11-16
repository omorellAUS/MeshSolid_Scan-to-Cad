"""Mesh loading and basic repair, keeping triangle connectivity."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import trimesh


@dataclass
class RepairReport:
    input_vertices: int = 0
    input_faces: int = 0
    merged_vertices: int = 0
    removed_degenerate: int = 0
    removed_duplicate: int = 0
    removed_internal_walls: int = 0   # opposite-winding duplicate faces: glued bodies, both copies removed
    holes_found: int = 0
    holes_filled: int = 0
    holes_left_open: int = 0
    non_manifold_edges: int = 0
    watertight: bool = False
    winding_consistent: bool = False
    flipped_inside_out: bool = False
    notes: list = field(default_factory=list)


def _boundary_loops(faces: np.ndarray):
    """Return list of vertex loops for open boundaries (edges used once)."""
    edges = np.sort(np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    directed = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    uniq, inv, cnt = np.unique(edges, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    once = cnt[inv] == 1
    bd = directed[once]
    # boundary loop walks the reverse of face edge direction so fill faces get consistent winding
    nxt = {}
    for a, b in bd:
        nxt.setdefault(int(b), []).append(int(a))
    loops, used = [], set()
    for start in list(nxt.keys()):
        if start in used:
            continue
        loop, cur, ok = [start], start, True
        used.add(start)
        while True:
            cands = [c for c in nxt.get(cur, []) if c not in used or c == start]
            if not cands:
                ok = False
                break
            n = cands[0]
            if n == start:
                break
            loop.append(n)
            used.add(n)
            cur = n
        if ok and len(loop) >= 3:
            loops.append(loop)
    return loops, int(cnt[cnt > 2].size)


class MeshInputError(ValueError):
    """The input file is not a usable triangle mesh (missing, empty, unreadable, degenerate)."""


def check_mesh_file(path: str):
    """Load the mesh or raise MeshInputError with a plain reason.  Returns the trimesh."""
    import os
    if not os.path.exists(path):
        raise MeshInputError(f"file not found: {path}")
    if os.path.isdir(path):
        raise MeshInputError(f"{path} is a folder, not a mesh file")
    if os.path.getsize(path) == 0:
        raise MeshInputError(f"{path} is empty (0 bytes)")
    try:
        raw = trimesh.load(path, process=False, force="mesh")
    except Exception as e:  # noqa: BLE001  (trimesh raises many types on bad files)
        raise MeshInputError(f"{path}: not a readable STL/OBJ/PLY/3MF mesh ({type(e).__name__}: {e})") from None
    if not isinstance(raw, trimesh.Trimesh) or len(raw.faces) == 0 or len(raw.vertices) == 0:
        raise MeshInputError(f"{path}: the file holds no triangles (not a mesh, or an empty/point-only file)")
    v = np.asarray(raw.vertices, float)
    if not np.isfinite(v).all():
        raise MeshInputError(f"{path}: {int((~np.isfinite(v)).any(axis=1).sum())} vertices have NaN/inf coordinates")
    ext = v.max(0) - v.min(0)
    if np.linalg.norm(ext) <= 0:
        raise MeshInputError(f"{path}: all vertices coincide (zero size)")
    if len(raw.faces) < 4:
        raise MeshInputError(f"{path}: only {len(raw.faces)} triangles; a solid needs at least 4")
    if np.sort(ext)[0] <= 1e-9 * ext.max():
        raise MeshInputError(f"{path}: the mesh is flat or a line (extents {ext.round(6).tolist()}); no volume to rebuild")
    return raw


def load_mesh(path: str, merge_tol: float = 1e-6, max_hole_edges: int = 32):
    """Load STL/OBJ and repair. Returns (trimesh.Trimesh, RepairReport).  Raises MeshInputError for
    files that are not a usable mesh."""
    raw = check_mesh_file(path)
    mesh, rep = repair_mesh(raw.vertices, raw.faces, merge_tol, max_hole_edges)
    if len(mesh.faces) < 4:
        raise MeshInputError(f"{path}: repair left {len(mesh.faces)} triangles (all degenerate or duplicate)")
    return mesh, rep


def _edge_counts(f):
    e = np.sort(np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    uniq, inv, cnt = np.unique(e, axis=0, return_inverse=True, return_counts=True)
    return inv.ravel().reshape(3, -1).T, cnt                 # per face its 3 edge ids, uses per edge


def _remove_seam_walls(v, f):
    """Glued bodies whose shared caps are triangulated differently on each side (mirrored halves exported
    as separate CAD bodies): every seam edge then has 4 faces.  Planar face components bounded only by
    such non-manifold edges are the bodies' caps inside the union: removing them leaves the union's skin.
    Applied only when it makes the mesh manifold.  Returns (faces, removed count)."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components
    fe, cnt = _edge_counts(f)
    if not (cnt > 2).any():
        return f, 0
    # faces joined across manifold edges only
    rows, cols = [], []
    ef = {}
    for k in range(3):
        for i, e in enumerate(fe[:, k]):
            if cnt[e] == 2:
                ef.setdefault(e, []).append(i)
    for a in ef.values():
        if len(a) == 2:
            rows.append(a[0]); cols.append(a[1])
    G = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(f), len(f)))
    ncomp, lab = connected_components(G, directed=False)
    n = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-300)
    scale = float(np.linalg.norm(v.max(0) - v.min(0)))
    drop = np.zeros(len(f), bool)
    soft = np.zeros(len(f), bool)
    for c in range(ncomp):
        idx = np.where(lab == c)[0]
        if len(idx) > 0.25 * len(f):
            continue
        n0 = n[idx[0]]
        if np.abs(np.abs(n[idx] @ n0) - 1).max() > 1e-6:
            continue                                          # not planar
        d = v[f[idx]].reshape(-1, 3) @ n0
        if np.ptp(d) > 1e-6 * scale:
            continue
        e_in = fe[idx].ravel()
        u, k = np.unique(e_in, return_counts=True)
        b = cnt[u[k == 1]]
        if (b > 2).all():                                     # its whole boundary is seam
            drop[idx] = True
        elif (b != 2).all() and (b > 2).any():
            soft[idx] = True                                  # seam + open edges: a leftover cap piece
    if drop.any():
        g = f[~drop]
        _, c2 = _edge_counts(g)
        if not ((c2 > 2).any() or (c2 == 1).sum() > (cnt == 1).sum()):
            return g, int(drop.sum())                         # manifold: done
    # partial caps (the halves' caps triangulated differently, opposite copies already removed): drop the
    # planar seam pieces when that strictly reduces the edges that are not shared by exactly two faces
    both = drop | soft
    if not both.any():
        return f, 0
    g = f[~both]
    _, c2 = _edge_counts(g)
    if (c2 != 2).sum() < (cnt != 2).sum():
        return g, int(both.sum())
    return f, 0


def repair_mesh(vertices, faces, merge_tol: float = 1e-6, max_hole_edges: int = 32):
    rep = RepairReport(input_vertices=len(vertices), input_faces=len(faces))
    v = np.asarray(vertices, dtype=float)
    f = np.asarray(faces, dtype=np.int64)
    # merge coincident vertices (grid rounding at merge_tol)
    key = np.round(v / merge_tol).astype(np.int64)
    _, idx, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = inv.ravel()
    rep.merged_vertices = len(v) - len(idx)
    v = v[idx]
    f = inv[f]
    # degenerate faces (repeated index or ~zero area)
    rep_idx = (f[:, 0] == f[:, 1]) | (f[:, 1] == f[:, 2]) | (f[:, 0] == f[:, 2])
    area = 0.5 * np.linalg.norm(np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), axis=1)
    scale = np.linalg.norm(v.max(0) - v.min(0))
    degen = rep_idx | (area < (1e-12 * scale * scale))
    rep.removed_degenerate = int(degen.sum())
    f = f[~degen]
    # duplicate faces.  A pair with opposite windings is an internal wall: two bodies glued face to face
    # (CAD exports of mirrored halves, multi-body prints), each closed by its own copy of the shared cap.
    # Both copies go, which leaves the union's closed skin; same-winding copies are plain duplicates (one kept).
    sf = np.sort(f, axis=1)
    _, ui, inv2, cnt = np.unique(sf, axis=0, return_index=True, return_inverse=True, return_counts=True)
    inv2 = inv2.ravel()
    r0 = np.argmin(f, axis=1)
    rot = np.stack([f[np.arange(len(f)), (r0 + k) % 3] for k in range(3)], 1)
    par = (rot[:, 1] < rot[:, 2]).astype(int)               # winding of each face w.r.t. its sorted key
    pos = np.bincount(inv2, par, minlength=len(ui))
    wall = (cnt >= 2) & (pos > 0) & (pos < cnt)               # copies with both windings
    keep = np.zeros(len(f), bool)
    keep[ui[~wall]] = True
    rep.removed_internal_walls = int(cnt[wall].sum())
    rep.removed_duplicate = int(len(f) - keep.sum() - rep.removed_internal_walls)
    f = f[np.sort(np.where(keep)[0])]
    f, nwall = _remove_seam_walls(v, f)
    rep.removed_internal_walls += nwall
    # holes
    loops, nm = _boundary_loops(f)
    rep.non_manifold_edges = nm
    rep.holes_found = len(loops)
    new = []
    for loop in loops:
        if len(loop) > max_hole_edges:
            rep.holes_left_open += 1
            continue
        c = v[loop].mean(0)
        ci = len(v) + len(new)
        new.append(c)
        for i in range(len(loop)):
            f = np.vstack([f, [[loop[i], loop[(i + 1) % len(loop)], ci]]])
        rep.holes_filled += 1
    if new:
        v = np.vstack([v, np.array(new)])
    m = trimesh.Trimesh(v, f, process=False)
    m.remove_unreferenced_vertices()
    rep.winding_consistent = bool(m.is_winding_consistent)
    if not rep.winding_consistent:
        trimesh.repair.fix_winding(m)
        rep.notes.append("winding made consistent")
        rep.winding_consistent = bool(m.is_winding_consistent)
    if m.is_watertight and m.volume < 0:
        m.invert()
        rep.flipped_inside_out = True
    rep.watertight = bool(m.is_watertight)
    if nm:
        rep.notes.append(f"{nm} non-manifold edges (>2 faces) - not repaired")
    return m, rep
