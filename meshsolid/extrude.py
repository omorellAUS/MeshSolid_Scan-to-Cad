"""2.5D builder: parts that are a stack of straight extrusions (plates, printed brackets, covers).

When (almost) every mesh face is either perpendicular to one direction d (a cap) or parallel to it (a
side wall), the solid is the union of prisms between consecutive cap levels. Each prism's profile is
the mesh cross-section at mid-slab, split into lines and circular arcs (vertex runs with uniform
turning, as CAD tessellation produces), so holes and slot ends become true circles. The result goes
through the same deviation gate as every other builder; nothing here is tuned per part."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class ExtrudeResult:
    ok: bool = False
    method: str = "extrude_profile"
    shape: object = None
    reason: str = ""
    extrude_dir: list = None
    prismatic_area_frac: float = 0.0
    levels: list = field(default_factory=list)
    slabs: list = field(default_factory=list)       # [{"from", "to", "loops", "lines", "arcs"}]
    brepcheck_valid: bool = False
    n_solids: int = 0
    n_faces: int = 0
    volume: float = 0.0
    mesh_volume: float = 0.0
    seconds: float = 0.0

    def to_json(self):
        return {k: v for k, v in self.__dict__.items() if k != "shape"}


def prismatic_axis(mesh, ang_deg=1.0):
    """(d, frac): the direction for which the largest area fraction of faces is a cap (normal = +-d) or
    a side wall (normal _|_ d), candidates being the normals of the largest faces."""
    N = np.asarray(mesh.face_normals)
    A = np.asarray(mesh.area_faces)
    tot = float(A.sum()) or 1.0
    c_cap, s_side = math.cos(math.radians(ang_deg)), math.sin(math.radians(ang_deg))
    best = (0.0, None)
    tried = []
    for i in np.argsort(-A)[:400]:
        d = N[i]
        if any(abs(float(d @ t)) > c_cap for t in tried):
            continue
        tried.append(d)
        dot = np.abs(N @ d)
        f = float(A[(dot > c_cap) | (dot < s_side)].sum() / tot)
        if f > best[0]:
            best = (f, d)
        if len(tried) >= 12:
            break
    return best[1], best[0]


def _levels(mesh, d, merge):
    N = np.asarray(mesh.face_normals)
    A = np.asarray(mesh.area_faces)
    h = np.asarray(mesh.triangles_center) @ d
    cap = np.abs(N @ d) > math.cos(math.radians(1.0))
    hv = np.asarray(mesh.vertices) @ d
    hs = np.sort(np.r_[h[cap], hv.min(), hv.max()])
    out = []
    for x in hs:
        if not out or x - out[-1][-1] > merge:
            out.append([x])
        else:
            out[-1].append(x)
    return [float(np.mean(g)) for g in out]


def _loops(segs, key_tol):
    """Chain plane-section segments (n,2,3) into closed vertex loops."""
    def k(p):
        return tuple(np.round(p / key_tol).astype(np.int64))
    adj = {}
    pts = {}
    for a, b in segs:
        ka, kb = k(a), k(b)
        if ka == kb:
            continue
        pts[ka], pts[kb] = a, b
        adj.setdefault(ka, []).append(kb)
        adj.setdefault(kb, []).append(ka)
    loops, seen = [], set()
    for s in adj:
        if s in seen or len(adj[s]) != 2:
            continue
        loop, prev, cur = [s], None, s
        seen.add(s)
        while True:
            nx = [n for n in adj[cur] if n != prev]
            if not nx:
                break
            prev, cur = cur, nx[0]
            if cur == s:
                loops.append(np.array([pts[q] for q in loop]))
                break
            if cur in seen or len(adj[cur]) != 2:
                break
            seen.add(cur)
            loop.append(cur)
    bad = sum(1 for s in adj if len(adj[s]) != 2)
    return loops, bad


def _cross2(a, b):
    return float(a[0] * b[1] - a[1] * b[0])


def _circle(P):
    A = np.c_[2 * P, np.ones(len(P))]
    b = (P * P).sum(1)
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    c = x[:2]
    return c, float(math.sqrt(max(x[2] + c @ c, 0.0)))


def simplify_loop(Q, eps):
    """Drop vertices that lie on the chord of their neighbours within eps (section points on the
    diagonal of a side-wall quad, collinear tessellation points)."""
    keep = np.ones(len(Q), bool)
    changed = True
    while changed and keep.sum() > 3:
        changed = False
        idx = np.where(keep)[0]
        P, A, B = Q[idx], Q[np.roll(idx, 1)], Q[np.roll(idx, -1)]
        u = B - A
        nu = np.linalg.norm(u, axis=1)
        dist = np.abs(u[:, 0] * (P - A)[:, 1] - u[:, 1] * (P - A)[:, 0]) / np.maximum(nu, 1e-300)
        cand = np.where(dist <= eps)[0]
        # drop every other candidate per pass so neighbours are re-tested against the new chord
        last = -2
        for c in cand:
            if c - last >= 2:
                keep[idx[c]] = False
                last = c
                changed = True
    return Q[keep]


def split_profile(Q, tol, corner_deg=50.0, max_sweep_deg=180.0):
    """Closed 2D vertex loop (already simplified) -> [('line', i, j) | ('arc', i, m, j)] over vertex
    indices (taken mod n). Arcs grow greedily over runs of >= 3 segments that turn the same way, each
    turn below corner_deg, with every run vertex on one circle within tol/10 (CAD tessellation vertices
    lie on the design curve); everything else becomes lines."""
    n = len(Q)
    e = np.roll(Q, -1, 0) - Q
    ang = np.arctan2(e[:, 1], e[:, 0])
    turn = (ang - np.roll(ang, 1) + np.pi) % (2 * np.pi) - np.pi   # turning at vertex i
    corner = np.abs(turn) > math.radians(corner_deg)
    start = int(np.argmax(corner)) if corner.any() else int(np.argmax(np.abs(turn)))
    order = [(start + i) % n for i in range(n)] + [start]
    out = []
    i0 = 0
    while i0 < n:
        j = i0
        best = None
        while j + 1 <= n:
            j2 = j + 1
            inner = [order[t] for t in range(i0 + 1, j2)]
            if inner and (corner[inner].any() or len(set(np.sign(turn[inner]))) > 1
                          or (np.abs(turn[inner]) < 1e-4).any()):
                break
            if abs(float(turn[inner].sum())) > math.radians(max_sweep_deg) if inner else False:
                break
            if j2 - i0 >= 3:
                idx = [order[t] for t in range(i0, j2 + 1)]
                c, r = _circle(Q[idx])
                if np.abs(np.linalg.norm(Q[idx] - c, axis=1) - r).max() > 0.1 * tol:
                    break
                best = j2
            j = j2
        if best is not None:
            out.append(("arc", order[i0], order[(i0 + best) // 2], order[best]))
            i0 = best
        else:
            out.append(("line", order[i0], order[i0 + 1]))
            i0 += 1
    return out


def _wire(Q3, segs):
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire
    from OCP.GC import GC_MakeArcOfCircle
    from OCP.gp import gp_Pnt
    n = len(Q3)
    P = [gp_Pnt(*map(float, q)) for q in Q3]
    mw = BRepBuilderAPI_MakeWire()
    for s in segs:
        if s[0] == "line":
            mw.Add(BRepBuilderAPI_MakeEdge(P[s[1] % n], P[s[2] % n]).Edge())
        else:
            arc = GC_MakeArcOfCircle(P[s[1] % n], P[s[2] % n], P[s[3] % n]).Value()
            mw.Add(BRepBuilderAPI_MakeEdge(arc).Edge())
    if not mw.IsDone():
        return None
    return mw.Wire()


def _area2(Q):
    x, y = Q[:, 0], Q[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _inside(p, Q):
    x, y = Q[:, 0], Q[:, 1]
    x2, y2 = np.roll(x, -1), np.roll(y, -1)
    c = ((y > p[1]) != (y2 > p[1])) & (p[0] < (x2 - x) * (p[1] - y) / (y2 - y + 1e-300) + x)
    return bool(c.sum() % 2)


def build_extrusion(mesh, tol, min_frac=0.995, max_slabs=12):
    t0 = time.time()
    res = ExtrudeResult(mesh_volume=float(mesh.volume))
    d, frac = prismatic_axis(mesh)
    res.prismatic_area_frac = round(frac, 5)
    if d is None or frac < min_frac:
        res.reason = f"not a 2.5D extrusion: cap/side-wall area {100 * frac:.1f}% < {100 * min_frac:.1f}%"
        return res
    d = d / np.linalg.norm(d)
    res.extrude_dir = np.round(d, 6).tolist()
    ext = float(np.linalg.norm(mesh.extents))
    lv = _levels(mesh, d, merge=max(2 * tol, 1e-4 * ext))
    res.levels = [round(x, 4) for x in lv]
    if len(lv) < 2 or len(lv) - 1 > max_slabs:
        res.reason = f"{len(lv) - 1} extrusion slabs (limit {max_slabs})"
        return res
    from trimesh.intersections import mesh_plane
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse
    from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.gp import gp_Vec, gp_Pln, gp_Pnt, gp_Dir
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_SOLID, TopAbs_FACE
    u = np.cross(d, [1.0, 0, 0] if abs(d[0]) < 0.9 else [0, 1.0, 0])
    u /= np.linalg.norm(u)
    v = np.cross(d, u)
    solid = None
    for a, b in zip(lv[:-1], lv[1:]):
        hm = 0.5 * (a + b)
        segs = mesh_plane(mesh, d, d * hm)
        info = {"from": round(a, 4), "to": round(b, 4), "loops": 0, "lines": 0, "arcs": 0}
        res.slabs.append(info)
        if len(segs) == 0:
            continue
        loops, bad = _loops(np.asarray(segs), key_tol=max(1e-7 * ext, 1e-9))
        if bad:
            res.reason = f"open cross-section at height {hm:.3f} ({bad} dangling points)"
            return res
        info["loops"] = len(loops)
        # loops in plane coordinates, projected onto the slab's bottom level
        L2 = [np.c_[(lp - d * hm) @ u, (lp - d * hm) @ v] for lp in loops]
        depth = [sum(_inside(L2[i][0], L2[j]) for j in range(len(L2)) if j != i) for i in range(len(L2))]
        for i in range(len(L2)):
            if depth[i] % 2:
                continue
            outer = simplify_loop(L2[i] if _area2(L2[i]) > 0 else L2[i][::-1], 1e-3 * tol)
            holes = [j for j in range(len(L2)) if depth[j] == depth[i] + 1 and _inside(L2[j][0], L2[i])]
            base = d * a
            def to3(Q):
                return base[None] + Q[:, :1] * u + Q[:, 1:2] * v
            so = split_profile(outer, tol)
            info["lines"] += sum(s[0] == "line" for s in so)
            info["arcs"] += sum(s[0] == "arc" for s in so)
            wo = _wire(to3(outer), so)
            if wo is None:
                res.reason = f"profile wire failed at height {hm:.3f}"
                return res
            mf = BRepBuilderAPI_MakeFace(gp_Pln(gp_Pnt(*map(float, base)), gp_Dir(*map(float, d))), wo, True)
            for j in holes:
                hq = simplify_loop(L2[j] if _area2(L2[j]) < 0 else L2[j][::-1], 1e-3 * tol)
                sh = split_profile(hq, tol)
                info["lines"] += sum(s[0] == "line" for s in sh)
                info["arcs"] += sum(s[0] == "arc" for s in sh)
                wh = _wire(to3(hq), sh)
                if wh is None:
                    res.reason = f"hole wire failed at height {hm:.3f}"
                    return res
                mf.Add(wh)
            if not mf.IsDone():
                res.reason = f"profile face failed at height {hm:.3f}"
                return res
            pr = BRepPrimAPI_MakePrism(mf.Face(), gp_Vec(*map(float, d * (b - a)))).Shape()
            if solid is None:
                solid = pr
            else:
                fu = BRepAlgoAPI_Fuse(solid, pr)
                fu.SetFuzzyValue(1e-6 * ext)
                fu.Build()
                if not fu.IsDone():
                    res.reason = "slab fuse failed"
                    return res
                solid = fu.Shape()
    if solid is None:
        res.reason = "empty cross-sections"
        return res
    un = ShapeUpgrade_UnifySameDomain(solid, True, True, True)
    un.Build()
    solid = un.Shape()
    res.brepcheck_valid = bool(BRepCheck_Analyzer(solid).IsValid())
    ex = TopExp_Explorer(solid, TopAbs_SOLID)
    while ex.More():
        res.n_solids += 1
        ex.Next()
    ex = TopExp_Explorer(solid, TopAbs_FACE)
    while ex.More():
        res.n_faces += 1
        ex.Next()
    p = GProp_GProps()
    BRepGProp.VolumeProperties_s(solid, p)
    res.volume = float(p.Mass())
    res.seconds = time.time() - t0
    res.shape = solid
    if not res.brepcheck_valid:
        res.reason = "BRepCheck_Analyzer reports invalid shape"
    elif res.volume <= 0:
        res.reason = "non-positive volume"
    elif abs(res.volume - res.mesh_volume) > 0.02 * abs(res.mesh_volume) + 1e-9:
        res.reason = f"volume {res.volume:.4g} differs from mesh {res.mesh_volume:.4g} by more than 2%"
    else:
        res.ok = True
    return res
