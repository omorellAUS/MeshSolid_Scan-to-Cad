"""Local trimming B-rep construction.

For every segmented region:
  1. build its fitted surface as a bounded face (region extent + margin),
  2. split that face only by the bounded faces of its neighbouring regions
     (vertex adjacency in the segmented mesh, so corner neighbours are included),
  3. keep the pieces whose surface samples project onto mesh faces of that region.
All kept pieces are sewn into a shell, closed into a solid and validated
(free edges, BRepCheck, orientation, volume).  No global cell decomposition.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import os

import numpy as np
import trimesh
from OCP.BRep import BRep_Tool
from OCP.BRepAlgoAPI import BRepAlgoAPI_Splitter
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeSolid, BRepBuilderAPI_Sewing
from OCP.BRepLib import BRepLib
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRepPrimAPI import (BRepPrimAPI_MakeCone, BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakeSphere,
                             BRepPrimAPI_MakeTorus)
from OCP.ShapeFix import ShapeFix_Solid, ShapeFix_Shell
from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED, TopAbs_SHELL, TopAbs_SOLID
from OCP.TopExp import TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS
from OCP.collections import List_TopoDS_Shape
from OCP.gp import gp_Ax3, gp_Circ, gp_Pln

from . import occ_util as U
from .tangency import contact
from .build import _ax2, _gd, _gp, _lateral_face


def _pt_tri_dist(P, A, B, C):
    """Exact point-triangle distances, vectorised (Ericson's closest-point algorithm)."""
    ab, ac, ap = B - A, C - A, P - A
    d1, d2 = np.einsum("ij,ij->i", ab, ap), np.einsum("ij,ij->i", ac, ap)
    bp = P - B
    d3, d4 = np.einsum("ij,ij->i", ab, bp), np.einsum("ij,ij->i", ac, bp)
    cp = P - C
    d5, d6 = np.einsum("ij,ij->i", ab, cp), np.einsum("ij,ij->i", ac, cp)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
    den = va + vb + vc
    den = np.where(np.abs(den) < 1e-30, 1e-30, den)
    v, w = vb / den, vc / den
    Q = A + ab * v[:, None] + ac * w[:, None]          # interior
    # edges / vertices
    m = (d1 <= 0) & (d2 <= 0); Q[m] = A[m]
    m = (d3 >= 0) & (d4 <= d3); Q[m] = B[m]
    m = (d6 >= 0) & (d5 <= d6); Q[m] = C[m]
    m = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    t = d1 / np.where(np.abs(d1 - d3) < 1e-30, 1e-30, d1 - d3); Q[m] = (A + ab * t[:, None])[m]
    m = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    t = d2 / np.where(np.abs(d2 - d6) < 1e-30, 1e-30, d2 - d6); Q[m] = (A + ac * t[:, None])[m]
    m = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    t = (d4 - d3) / np.where(np.abs((d4 - d3) + (d5 - d6)) < 1e-30, 1e-30, (d4 - d3) + (d5 - d6))
    Q[m] = (B + (C - B) * t[:, None])[m]
    return np.linalg.norm(P - Q, axis=1)


class NearestFace:
    """Memory-bounded nearest-triangle query that is robust to long sliver triangles
    (exact CAD tessellations): candidates come from a KD-tree over dense samples of every
    triangle (spacing <= `step`), then exact point-triangle distances are evaluated."""

    def __init__(self, mesh, k=24, step=None, max_samples=400_000):
        from scipy.spatial import cKDTree
        V = np.asarray(mesh.vertices)
        F = np.asarray(mesh.faces)
        self.T = V[F]
        e = np.linalg.norm(self.T - np.roll(self.T, 1, axis=1), axis=2).max(1)
        area = 0.5 * np.linalg.norm(np.cross(self.T[:, 1] - self.T[:, 0], self.T[:, 2] - self.T[:, 0]), axis=1)
        if step is None:
            step = max(np.sqrt(area.sum() / max_samples) * 1.5, 1e-6)
        n = np.clip(np.ceil(e / step).astype(int), 1, 200)   # subdivisions per triangle edge
        pts, tid = [], []
        for nn in np.unique(n):
            sel = np.where(n == nn)[0]
            ij = [(i, j) for i in range(nn + 1) for j in range(nn + 1 - i)]
            Bc = np.array([(i / nn, j / nn, 1 - (i + j) / nn) for i, j in ij]) if nn > 1 else np.array([[1 / 3, 1 / 3, 1 / 3]])
            Pt = np.einsum("bk,tkd->tbd", Bc, self.T[sel]).reshape(-1, 3)
            pts.append(Pt)
            tid.append(np.repeat(sel, len(Bc)))
        self.S = np.vstack(pts)
        self.tid = np.concatenate(tid)
        self.tree = cKDTree(self.S)
        self.k = min(k, len(self.S))

    def query(self, P, chunk=20000):
        P = np.asarray(P, float)
        dist = np.empty(len(P))
        tri = np.empty(len(P), int)
        for s0 in range(0, len(P), chunk):
            Q = P[s0:s0 + chunk]
            _, idx = self.tree.query(Q, k=self.k)
            cand = self.tid[idx.reshape(len(Q), -1)]
            D = np.empty(cand.shape)
            for j in range(cand.shape[1]):
                t = self.T[cand[:, j]]
                D[:, j] = _pt_tri_dist(Q, t[:, 0], t[:, 1], t[:, 2])
            best = D.argmin(1)
            dist[s0:s0 + chunk] = D[np.arange(len(Q)), best]
            tri[s0:s0 + chunk] = cand[np.arange(len(Q)), best]
        return dist, tri


def region_neighbours(mesh, labels):
    """Vertex-adjacency between region labels (unassigned faces bridge their neighbours)."""
    F = np.asarray(mesh.faces)
    nv = len(mesh.vertices)
    lab = labels.copy()
    # give unassigned faces the label of an adjacent labelled face (iteratively) for adjacency only
    adj = np.asarray(mesh.face_adjacency)
    for _ in range(10):
        un = lab < 0
        if not un.any():
            break
        a, b = adj[:, 0], adj[:, 1]
        m1 = un[a] & ~un[b]
        lab[a[m1]] = lab[b[m1]]
        m2 = un[b] & ~un[a]
        lab[b[m2]] = lab[a[m2]]
    vl = [set() for _ in range(nv)]
    for f, l in zip(F, lab):
        if l >= 0:
            for v in f:
                vl[v].add(int(l))
    nb = {}
    for s in vl:
        for a in s:
            nb.setdefault(a, set()).update(s - {a})
    return nb, lab


def bounded_face(prim, P, margin, P_own=None, own_margin=None):
    """Fitted surface bounded to the region's points P (+margin)."""
    q, k = prim.params, prim.kind
    if k == "plane":
        n = q["normal"]
        ax3 = gp_Ax3(_gp(q["point"]), _gd(n))
        xd = ax3.XDirection(); yd = ax3.YDirection()
        xu = np.array([xd.X(), xd.Y(), xd.Z()]); yu = np.array([yd.X(), yd.Y(), yd.Z()])
        d = P - q["point"]
        u, v = d @ xu, d @ yu
        return BRepBuilderAPI_MakeFace(gp_Pln(ax3), u.min() - margin, u.max() + margin,
                                       v.min() - margin, v.max() + margin).Face()
    if k == "cylinder":
        a = q["axis"]
        h = (P - q["point"]) @ a
        base = q["point"] + a * (h.min() - margin)
        return _lateral_face(BRepPrimAPI_MakeCylinder(_ax2(base, a), float(q["radius"]),
                                                      float(np.ptp(h) + 2 * margin)).Shape())
    if k == "cone":
        a, ha = q["axis"], float(q["half_angle"])
        h = (P - q["apex"]) @ a
        h0 = max(h.min() - margin, 1e-3)
        h1 = h.max() + margin
        return _lateral_face(BRepPrimAPI_MakeCone(_ax2(q["apex"] + a * h0, a), h0 * np.tan(ha),
                                                  h1 * np.tan(ha), h1 - h0).Shape())
    if k == "sphere":
        c, R = q["center"], float(q["radius"])
        U = P - c
        U = U / np.maximum(np.linalg.norm(U, axis=1, keepdims=True), 1e-12)
        m = U.mean(0)
        if np.linalg.norm(m) > 0.5:
            a = m / np.linalg.norm(m)             # cap around the mean direction
        else:
            a = np.linalg.eigh(U.T @ U)[1][:, 0]  # band: axis = least-spread direction
        lat = np.arcsin(np.clip(U @ a, -1, 1))
        dm = margin / R
        a1 = max(lat.min() - dm, -np.pi / 2)
        a2 = min(lat.max() + dm, np.pi / 2)
        return _lateral_face(BRepPrimAPI_MakeSphere(_ax2(c, a), R, float(a1), float(a2)).Shape())
    if k == "bspline":
        # the fitted surface already extends past its data by the fit margin
        from .freeform import to_occ
        if getattr(prim, "_occ", None) is None:
            prim._occ = to_occ(prim)
        return BRepBuilderAPI_MakeFace(prim._occ, 1e-7).Face()
    if k == "pipe":
        raise ValueError("pipe faces are built by pipe_face_and_contacts (needs its support cylinder)")
    if k == "torus":
        # bounded in the tube angle to the points (+margin): keeps the v-seam (outer equator by
        # default) away from tangent contact circles, which otherwise cannot split the face
        c, a, R, r = q["center"], q["axis"], float(q["major_radius"]), float(q["minor_radius"])
        ax = _ax2(c, a)
        xd = ax.XDirection()
        X = np.array([xd.X(), xd.Y(), xd.Z()])
        # tube angle: from the torus region's own points when given (a tool sized to the union
        # of two regions must not wrap round the tube onto the neighbour's side)
        Pt = P if P_own is None else P_own
        mt = margin if own_margin is None else own_margin
        d = Pt - c
        h = d @ a
        rho = np.linalg.norm(d - np.outer(h, a), axis=1)
        v = np.arctan2(h, rho - R)
        v0 = float(np.angle(np.exp(1j * v).mean()))
        dv = np.angle(np.exp(1j * (v - v0)))
        lo, hi = dv.min() - mt / r, dv.max() + mt / r
        if hi - lo >= 2 * np.pi - 0.2:
            lo, hi = -np.pi + 0.1, np.pi - 0.1   # (nearly) full tube: seam opposite the mean
        a1, a2 = v0 + lo, v0 + hi
        sh = np.floor(a1 / (2 * np.pi)) * 2 * np.pi
        a1, a2 = a1 - sh, a2 - sh
        if a2 - a1 >= 2 * np.pi - 1e-6:
            a2 = a1 + 2 * np.pi - 1e-3
        del X
        return _lateral_face(BRepPrimAPI_MakeTorus(ax, R, r, float(a1), float(a2), 2 * np.pi).Shape())
    raise ValueError(k)


def _pipe_curves(q, cy, P, margin):
    """Spine ellipse arc, plane-contact ellipse arc and cylinder-contact curve of a pipe blend,
    trimmed to the parameter range of points P (+margin in mm). Same parametrisation for all three."""
    from OCP.Geom import Geom_Ellipse, Geom_TrimmedCurve
    from OCP.GeomAPI import GeomAPI_Interpolate
    from OCP.collections import HArray1_double, HArray1_gp_Pnt
    from OCP.gp import gp_Ax2, gp_Elips
    from .blend import ellipse_param, ellipse_pts
    t = ellipse_param(q, P)
    t0 = float(np.angle(np.exp(1j * t).mean()))
    tt = np.angle(np.exp(1j * (t - t0)))
    dm = margin / q["B"]
    a, b = t0 + tt.min() - dm, t0 + tt.max() + dm
    if b - a > 2 * np.pi - 1e-3:
        a, b = t0 - np.pi + 1e-3, t0 + np.pi - 1e-3
    sh = np.floor(a / (2 * np.pi)) * 2 * np.pi
    a, b = float(a - sh), float(b - sh)

    def ell(center):
        return Geom_Ellipse(gp_Elips(gp_Ax2(_gp(center), _gd(q["normal"]), _gd(q["u"])), float(q["A"]), float(q["B"])))
    spine = Geom_TrimmedCurve(ell(q["center"]), a, b)
    c1 = Geom_TrimmedCurve(ell(q["center"] - q["s_plane"] * q["radius"] * q["normal"]), a, b)
    ts = np.linspace(a, b, max(40, int((b - a) / 0.01)))
    Sp = ellipse_pts(q, ts)
    ax = cy["axis"]
    foot = cy["point"] + np.outer((Sp - cy["point"]) @ ax, ax)
    rad = Sp - foot
    C2 = foot + rad * (cy["radius"] / np.linalg.norm(rad, axis=1))[:, None]
    arr = HArray1_gp_Pnt(1, len(C2))
    par = HArray1_double(1, len(ts))
    for i, (p, x) in enumerate(zip(C2, ts)):
        arr.SetValue(i + 1, _gp(p))
        par.SetValue(i + 1, float(x))
    it = GeomAPI_Interpolate(arr, par, False, 1e-9)
    it.Perform()
    return spine, c1, it.Curve()


def pipe_face_and_contacts(prim, cy, P, margin):
    from OCP.GeomFill import GeomFill_Pipe
    q = prim.params
    spine, c1, c2 = _pipe_curves(q, cy, P, margin)
    pipe = GeomFill_Pipe(spine, c1, c2, float(q["radius"]))
    pipe.Perform(1e-7, False)
    if not pipe.IsDone():
        raise RuntimeError("GeomFill_Pipe failed")
    face = BRepBuilderAPI_MakeFace(pipe.Surface(), 1e-7).Face()
    e1 = BRepBuilderAPI_MakeEdge(c1).Edge()
    e2 = BRepBuilderAPI_MakeEdge(c2).Edge()
    return face, e1, e2


def contact_on(prim, c):
    """Move a contact curve (computed from two nearly tangent surfaces) exactly onto prim's
    surface, so that it splits prim's face (edges off the surface by > 1e-7 are ignored)."""
    kind, d, gap = c
    q, k = prim.params, prim.kind
    d = dict(d)
    if kind == "line":
        p0, u = d["point"], d["dir"]
        if k == "plane":
            n = q["normal"]
            d["point"] = p0 - ((p0 - q["point"]) @ n) * n
            u2 = u - (u @ n) * n
            d["dir"] = u2 / np.linalg.norm(u2)
        elif k == "cylinder":
            ax = q["axis"]
            d["dir"] = ax if ax @ u >= 0 else -ax
            w = p0 - q["point"]
            rv = w - (w @ q["axis"]) * q["axis"]
            d["point"] = p0 + rv / np.linalg.norm(rv) * (q["radius"] - np.linalg.norm(rv))
        elif k == "cone":
            # the cone generator through the apex closest in direction to u
            ax, al = q["axis"], q["half_angle"]
            uu = u if u @ ax >= 0 else -u
            rr = uu - (uu @ ax) * ax
            if np.linalg.norm(rr) > 1e-12:
                rr = rr / np.linalg.norm(rr)
                d["point"] = q["apex"]
                d["dir"] = np.cos(al) * ax + np.sin(al) * rr
        return kind, d, gap
    if kind == "circle":
        c0, a, rad = d["center"], d["axis"], d["radius"]
        if k == "plane":
            d["center"] = c0 - ((c0 - q["point"]) @ q["normal"]) * q["normal"]
            d["axis"] = q["normal"]
        elif k == "cylinder":
            d["radius"] = float(q["radius"])
            d["axis"] = q["axis"]
            d["center"] = q["point"] + ((c0 - q["point"]) @ q["axis"]) * q["axis"]
        elif k == "cone":
            h = (c0 - q["apex"]) @ q["axis"]
            # nearest point of the cone's meridian line to (rad, h)
            al = q["half_angle"]
            uu = np.array([np.sin(al), np.cos(al)])
            t = np.array([rad, h]) @ uu
            d["radius"] = float(t * uu[0])
            d["center"] = q["apex"] + t * uu[1] * q["axis"]
            d["axis"] = q["axis"]
        elif k == "sphere":
            v = np.array([rad, (c0 - q["center"]) @ a])
            v = v / np.linalg.norm(v) * q["radius"]
            d["radius"] = float(v[0])
            d["center"] = q["center"] + v[1] * a
        elif k == "torus":
            ta = q["axis"]
            h = (c0 - q["center"]) @ ta
            w = np.array([rad - q["major_radius"], h])
            w = w / np.linalg.norm(w) * q["minor_radius"]
            d["radius"] = float(q["major_radius"] + w[0])
            d["center"] = q["center"] + w[1] * ta
            d["axis"] = ta
        return kind, d, gap
    return c


def contact_edge(c, PU, margin):
    kind, d, _gap = c
    if kind == "line":
        t = (PU - d["point"]) @ d["dir"]
        a = d["point"] + d["dir"] * (t.min() - margin)
        b = d["point"] + d["dir"] * (t.max() + margin)
        return BRepBuilderAPI_MakeEdge(_gp(a), _gp(b)).Edge()
    if kind == "circle":
        return BRepBuilderAPI_MakeEdge(gp_Circ(_ax2(d["center"], d["axis"]), float(d["radius"]))).Edge()
    raise ValueError(kind)


def _assemble_volume(kept, fuzzy, mesh, res):
    """BOPAlgo_MakerVolume on the trimmed pieces; keeps the volumes whose interior lies inside the
    mesh. Records the attempt in res.assembly_attempts; on success fills res and returns True."""
    from OCP.BOPAlgo import BOPAlgo_MakerVolume
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse
    from OCP.BRepClass3d import BRepClass3d_SolidClassifier
    from OCP.TopAbs import TopAbs_IN
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib
    att = {"fuzzy": round(float(fuzzy), 6)}
    res.assembly_attempts.append(att)
    try:
        mv = BOPAlgo_MakerVolume()
        L = List_TopoDS_Shape()
        for f in kept:
            L.Append(f)
        mv.SetArguments(L)
        mv.SetFuzzyValue(float(fuzzy))
        mv.SetIntersect(True)
        mv.Perform()
        if mv.HasErrors():
            att["error"] = "MakerVolume reported errors"
            return False
        sols = []
        ex = TopExp_Explorer(mv.Shape(), TopAbs_SOLID)
        while ex.More():
            sols.append(TopoDS.Solid(ex.Current()))
            ex.Next()
        att["n_volumes"] = len(sols)
        mv_vol = abs(float(res.mesh_volume))
        inside = []
        rng = np.random.default_rng(0)
        for so in sols:
            v = U.volume(so)
            if abs(v) < 1e-6 * mv_vol:
                continue
            bb = Bnd_Box()
            BRepBndLib.Add_s(so, bb)
            a, b = bb.CornerMin(), bb.CornerMax()
            lo, hi = np.array([a.X(), a.Y(), a.Z()]), np.array([b.X(), b.Y(), b.Z()])
            pts = []
            for q in lo + rng.random((4000, 3)) * (hi - lo):
                cl = BRepClass3d_SolidClassifier(so, _gp(q), 1e-6)
                if cl.State() == TopAbs_IN:
                    pts.append(q)
                    if len(pts) >= 25:
                        break
            if not pts:
                continue
            frac = float(np.mean(mesh.contains(np.array(pts))))
            if frac > 0.5:
                inside.append(so)
        att["n_inside"] = len(inside)
        if not inside:
            att["error"] = "no enclosed volume inside the mesh"
            return False
        shape = inside[0]
        for so in inside[1:]:
            fu = BRepAlgoAPI_Fuse(shape, so)
            fu.Build()
            shape = fu.Shape()
        valid0 = bool(U.check(shape))
        att["valid_before_unify"] = valid0
        att["volume_before_unify"] = round(float(U.volume(shape)), 3)
        try:
            # UnifySameDomain can modify shared sub-shapes in place, so run it on a deep copy and
            # keep the original untouched when the unified result is not valid.
            from OCP.BRepBuilderAPI import BRepBuilderAPI_Copy
            cp = BRepBuilderAPI_Copy(shape, True, False)
            un = ShapeUpgrade_UnifySameDomain(cp.Shape(), True, True, True)
            un.Build()
            sh2 = un.Shape()
            att["unify_valid"] = bool(U.check(sh2))
            if not att["unify_valid"]:
                res.unify_failed = True
            else:
                shape = sh2
        except Exception:  # noqa: BLE001
            res.unify_failed = True
        # tolerances after a fuzzy Boolean are inflated to about the fuzzy value; recompute them from the
        # actual curve/pcurve/vertex deviations (reset, SameParameter, UpdateTolerances) and keep the
        # recomputed shape only if it is still valid
        try:
            from OCP.ShapeFix import ShapeFix_ShapeTolerance
            from OCP.ShapeAnalysis import ShapeAnalysis_ShapeTolerance
            from OCP.BRepLib import BRepLib
            from OCP.BRepBuilderAPI import BRepBuilderAPI_Copy
            from OCP.TopAbs import TopAbs_SHAPE, TopAbs_EDGE, TopAbs_VERTEX
            sat = ShapeAnalysis_ShapeTolerance()
            att["max_tol_fuzzy"] = float(max(sat.Tolerance(shape, 1, TopAbs_EDGE), sat.Tolerance(shape, 1, TopAbs_VERTEX)))
            s2 = BRepBuilderAPI_Copy(shape, True, False).Shape()
            ShapeFix_ShapeTolerance().SetTolerance(s2, 1e-7, TopAbs_SHAPE)
            BRepLib.SameParameter_s(s2, 1e-7, True)
            BRepLib.UpdateTolerances_s(s2, True)
            if U.check(s2) and U.count(s2, TopAbs_SOLID) == U.count(shape, TopAbs_SOLID):
                shape = s2
                att["max_tol_recomputed"] = float(max(sat.Tolerance(s2, 1, TopAbs_EDGE), sat.Tolerance(s2, 1, TopAbs_VERTEX)))
        except Exception as ex:  # noqa: BLE001
            att["tol_recompute_error"] = str(ex)
        n_sol = U.count(shape, TopAbs_SOLID)
        valid = bool(U.check(shape))
        vol = float(U.volume(shape))
        att.update({"n_solids": n_sol, "valid": valid, "volume": round(vol, 3)})
        probs = []
        if n_sol != 1:
            probs.append(f"{n_sol} solids")
        if not valid:
            probs.append("BRepCheck_Analyzer reports invalid shape")
        if vol <= 0 or abs(vol - mv_vol) > 0.02 * mv_vol:
            probs.append(f"volume {vol:.1f} vs mesh {mv_vol:.1f}")
        if probs:
            att["error"] = "; ".join(probs)
            return False
        res.shape = shape
        res.n_solids, res.n_faces = n_sol, U.count(shape, TopAbs_FACE)
        res.brepcheck_valid, res.volume = valid, vol
        res.n_shells = U.count(shape, TopAbs_SHELL)
        res.free_edges = 0
        res.ok = True
        res.reason = ""
        res.assembly_fuzzy = float(fuzzy)
        return True
    except Exception as e:  # noqa: BLE001
        att["error"] = f"{type(e).__name__}: {e}"
        return False


def _wire_points(w, n=8):
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.TopAbs import TopAbs_EDGE
    out = []
    ee = TopExp_Explorer(w, TopAbs_EDGE)
    while ee.More():
        c = BRepAdaptor_Curve(TopoDS.Edge(ee.Current()))
        for t in np.linspace(c.FirstParameter(), c.LastParameter(), n):
            p = c.Value(t)
            out.append((p.X(), p.Y(), p.Z()))
        ee.Next()
    return np.array(out)


def _free_edge_clusters(sew, link):
    """Cluster free-edge endpoints (single linkage, distance `link`); returns cluster centres."""
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    pts = []
    for i in range(1, sew.NbFreeEdges() + 1):
        c = BRepAdaptor_Curve(TopoDS.Edge(sew.FreeEdge(i)))
        for t in (c.FirstParameter(), c.LastParameter()):
            p = c.Value(t)
            pts.append((p.X(), p.Y(), p.Z()))
    if not pts:
        return []
    P = np.array(pts)
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    pairs = cKDTree(P).query_pairs(link, output_type="ndarray")
    import scipy.sparse as sps
    A = sps.coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(P), len(P))) if len(pairs) else sps.coo_matrix((len(P), len(P)))
    n, lab = connected_components(A, directed=False)
    return [P[lab == k].mean(0) for k in range(n)]


SETBACK_MAX_CORNERS = 150


def wall_thickness(mesh, q=10, n=2000):
    """q-th percentile of the inward ray thickness (face centroid along -normal to the opposite
    side) over a random face sample; 0 when no ray hits (open or degenerate mesh)."""
    try:
        C = np.asarray(mesh.triangles_center)
        N = np.asarray(mesh.face_normals)
        idx = np.random.default_rng(0).choice(len(C), min(n, len(C)), replace=False)
        eps = 1e-4 * float(np.linalg.norm(mesh.extents))
        loc, ri, _ = mesh.ray.intersects_location(C[idx] - N[idx] * eps, -N[idx], multiple_hits=False)
        if len(ri) < 10:
            return 0.0
        d = np.linalg.norm(loc - C[idx][ri], axis=1)
        return float(np.percentile(d, q))
    except Exception:  # noqa: BLE001
        return 0.0


def _setback_fill(kept, kept_rid, centers, rho, V, res, drop_small=False, pq=None, g1=False, t_stop=None):
    """t_stop: wall-clock time after which no more balls are cut / holes patched (the remaining corners
    stay open and the build reports them; res.budget_hit = 'setback')."""
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeSphere
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCP.BRepFill import BRepFill_Filling
    from OCP.GeomAbs import GeomAbs_C0
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Sewing as _Sew
    from OCP.ShapeAnalysis import ShapeAnalysis_FreeBounds
    from OCP.TopAbs import TopAbs_WIRE, TopAbs_EDGE
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib
    from scipy.spatial import cKDTree
    res.setback_corners = len(centers)
    res.setback_radius = round(float(rho), 4)
    C = np.array(centers)
    balls = [BRepPrimAPI_MakeSphere(_gp(c), float(rho)).Shape() for c in C]
    k2, r2 = [], []
    for f, rid in zip(kept, kept_rid):
        bb = Bnd_Box()
        BRepBndLib.Add_s(f, bb)
        a, b = bb.CornerMin(), bb.CornerMax()
        lo = np.array([a.X(), a.Y(), a.Z()]) - rho
        hi = np.array([b.X(), b.Y(), b.Z()]) + rho
        near = [i for i, c in enumerate(C) if np.all(c >= lo) and np.all(c <= hi)]
        if t_stop is not None and time.time() > t_stop:
            res.budget_hit = "setback"
            near = []
        cur = [f]
        for i in near:
            nxt = []
            for g in cur:
                cut = BRepAlgoAPI_Cut(g, balls[i])
                if cut.IsDone():
                    nxt += [x for x in _faces(cut.Shape()) if _area(x) > 1e-3 * rho ** 2]
                else:
                    nxt.append(g)
            cur = nxt
        if near and drop_small:
            # small leftovers at a corner are the overlapping slivers that caused the mismatch:
            # drop them, the corner patch covers that area (checked against the mesh later)
            keep = []
            for g in cur:
                if _area(g) < 0.25 * np.pi * rho ** 2:
                    gb = Bnd_Box()
                    BRepBndLib.Add_s(g, gb)
                    ga, gc = gb.CornerMin(), gb.CornerMax()
                    gm = 0.5 * np.array([ga.X() + gc.X(), ga.Y() + gc.Y(), ga.Z() + gc.Z()])
                    if np.min(np.linalg.norm(C[near] - gm, axis=1)) < 2 * rho:
                        res.setback_dropped += 1
                        continue
                keep.append(g)
            cur = keep
        k2 += cur
        r2 += [rid] * len(cur)
    # boundary loops of the set-back holes
    sw = _Sew(1e-4)
    for f in k2:
        sw.Add(f)
    sw.Perform()
    fb = ShapeAnalysis_FreeBounds(sw.SewedShape(), 1e-4, False, False)
    # point constraints come from dense samples of the mesh triangles (an exact CAD mesh has
    # few vertices inside a small corner hole)
    SP = pq.S if pq is not None else V
    tree = cKDTree(SP) if pq is None else pq.tree
    patches = []
    from OCP.GeomAbs import GeomAbs_G1
    from OCP.ShapeFix import ShapeFix_Face
    from OCP.BRepCheck import BRepCheck_Analyzer
    face_edge_index = []
    if g1:
        for sf in _faces(sw.SewedShape()):
            es = []
            e2 = TopExp_Explorer(sf, TopAbs_EDGE)
            while e2.More():
                es.append(e2.Current())
                e2.Next()
            face_edge_index.append((sf, es))
    ex = TopExp_Explorer(fb.GetClosedWires(), TopAbs_WIRE)
    while ex.More():
        w = TopoDS.Wire(ex.Current())
        ex.Next()
        if t_stop is not None and time.time() > t_stop:
            res.budget_hit = "setback"
            res.setback_patch_failures += 1
            continue
        bb = Bnd_Box()
        BRepBndLib.Add_s(w, bb)
        a, b = bb.CornerMin(), bb.CornerMax()
        mid = 0.5 * np.array([a.X() + b.X(), a.Y() + b.Y(), a.Z() + b.Z()])
        wp = _wire_points(w)
        if len(wp) == 0 or np.min(np.linalg.norm(wp[:, None, :] - C[None], axis=2)) > 1.2 * rho:
            continue   # not a set-back hole
        try:
            fl = BRepFill_Filling(3, 15, 3, False, 1e-5, 1e-4, 0.01, 0.1, 8, 12)
            ee = TopExp_Explorer(w, TopAbs_EDGE)
            while ee.More():
                ed = TopoDS.Edge(ee.Current())
                sup = None
                if g1:
                    for sf, sedges in face_edge_index:
                        if any(x.IsSame(ed) for x in sedges):
                            sup = sf
                            break
                if sup is not None:
                    fl.Add(ed, sup, GeomAbs_G1, True)
                else:
                    fl.Add(ed, GeomAbs_C0, True)
                ee.Next()
            # interior mesh points: within the hole's bbox and away from its boundary
            # the hole is (surface inside the balls): constrain only by mesh points inside them
            dist_c = np.linalg.norm(C - mid, axis=1)
            near_c = C[dist_c < 2 * rho + float(np.linalg.norm([b.X() - a.X(), b.Y() - a.Y(), b.Z() - a.Z()]))]
            idx = sorted({j for cc in near_c for j in tree.query_ball_point(cc, 0.92 * rho)})
            if idx:
                idx = np.array(idx)
                if len(idx) > 4000:
                    idx = np.random.default_rng(0).choice(idx, 4000, replace=False)
                dd = np.min(np.linalg.norm(SP[idx][:, None, :] - wp[None], axis=2), axis=1)
                idx = idx[dd > 0.1 * rho]
                # spread the constraints: farthest-point subsample
                if len(idx) > 60:
                    sel = [0]
                    dmin = np.linalg.norm(SP[idx] - SP[idx[0]], axis=1)
                    for _ in range(59):
                        k = int(dmin.argmax())
                        sel.append(k)
                        dmin = np.minimum(dmin, np.linalg.norm(SP[idx] - SP[idx[k]], axis=1))
                    idx = idx[sel]
            for j in idx:
                fl.Add(_gp(SP[j]))
            pf = _filling_face(fl, t_stop, res)
            if pf is not None:
                if not BRepCheck_Analyzer(pf).IsValid():
                    sff = ShapeFix_Face(pf)
                    sff.Perform()
                    pf = sff.Face()
                # a filling patch must stay inside its hole: a surface that runs far past the boundary loop
                # is a failed fill (and meshing it finely can take minutes); deflection follows the hole size
                pb = Bnd_Box()
                BRepBndLib.Add_s(pf, pb)
                pa, pz = pb.CornerMin(), pb.CornerMax()
                hd = float(np.linalg.norm([b.X() - a.X(), b.Y() - a.Y(), b.Z() - a.Z()]))
                pd = float(np.linalg.norm([pz.X() - pa.X(), pz.Y() - pa.Y(), pz.Z() - pa.Z()]))
                if pd > 3 * hd + 2 * rho:
                    res.setback_patch_failures += 1
                    continue
                Sx, Wx = _face_samples(pf, max(rho / 20, hd / 60))
                if Sx is not None and pq is not None:
                    if len(Sx) > 20000:
                        Sx = Sx[np.random.default_rng(0).choice(len(Sx), 20000, replace=False)]
                    dx, _t = pq.query(Sx)
                    res.setback_patch_stats.append({"area": round(_area(pf), 3), "max_dev": round(float(dx.max()), 4),
                                                    "p95_dev": round(float(np.percentile(dx, 95)), 4),
                                                    "center": np.round(mid, 2).tolist()})
                patches.append(pf)
                res.setback_patches += 1
            else:
                res.setback_patch_failures += 1
        except Exception:  # noqa: BLE001
            res.setback_patch_failures += 1
    res.debug_setback = (k2, r2, patches, C)
    return k2, r2, patches


SETBACK_PATCH_TIMEOUT = 20.0   # s per N-sided patch (BRepFill_Filling can run for many minutes)


def _filling_face(fl, t_stop, res):
    """Build a BRepFill_Filling and return its face or None. On POSIX the build runs in a forked child
    with a time limit (a hang becomes a counted patch timeout); elsewhere inline."""
    def _build():
        fl.Build()
        return fl.Face() if fl.IsDone() else None
    from .occ_child import _use_fork
    if not _use_fork():
        return _build()
    from .impeller import _in_child
    lim = SETBACK_PATCH_TIMEOUT
    if t_stop is not None:
        lim = min(lim, max(t_stop - time.time(), 1.0))
    try:
        sh = _in_child(_build, lim)
    except TimeoutError:
        res.setback_patch_timeouts += 1
        return None
    return None if sh is None else TopoDS.Face(sh)


def _fill_small_holes(shape, tol, fill_max, res):
    """Close free-boundary loops no larger than fill_max (bbox diagonal) with BRepFill_Filling
    patches (C0 to the boundary). Returns the patch faces; sizes are reported in res."""
    from OCP.ShapeAnalysis import ShapeAnalysis_FreeBounds
    from OCP.BRepFill import BRepFill_Filling
    from OCP.GeomAbs import GeomAbs_C0
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib
    from OCP.TopAbs import TopAbs_WIRE, TopAbs_EDGE
    fb = ShapeAnalysis_FreeBounds(shape, tol, False, False)
    out = []
    ex = TopExp_Explorer(fb.GetClosedWires(), TopAbs_WIRE)
    while ex.More():
        w = TopoDS.Wire(ex.Current())
        ex.Next()
        bb = Bnd_Box()
        BRepBndLib.Add_s(w, bb)
        a, b = bb.CornerMin(), bb.CornerMax()
        size = float(np.linalg.norm([b.X() - a.X(), b.Y() - a.Y(), b.Z() - a.Z()]))
        if size > fill_max:
            res.holes_unfilled_sizes.append(round(size, 3))
            continue
        try:
            fl = BRepFill_Filling()
            ee = TopExp_Explorer(w, TopAbs_EDGE)
            n = 0
            while ee.More():
                fl.Add(TopoDS.Edge(ee.Current()), GeomAbs_C0, True)
                n += 1
                ee.Next()
            if n < 2:
                continue
            fl.Build()
            if not fl.IsDone():
                res.holes_unfilled_sizes.append(round(size, 3))
                continue
            out.append(fl.Face())
            res.holes_filled += 1
            res.hole_fill_max_size = max(res.hole_fill_max_size, round(size, 4))
        except Exception:  # noqa: BLE001
            res.holes_unfilled_sizes.append(round(size, 3))
    return out


def _invalid_faces(shape, limit=30):
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    out = []
    for f in _faces(shape):
        if not BRepCheck_Analyzer(f).IsValid():
            p = GProp_GProps()
            BRepGProp.SurfaceProperties_s(f, p)
            c = p.CentreOfMass()
            out.append({"type": str(BRepAdaptor_Surface(f).GetType()).split("_")[-1], "area": round(p.Mass(), 3),
                        "center": [round(c.X(), 2), round(c.Y(), 2), round(c.Z(), 2)]})
            if len(out) >= limit:
                break
    return out


def _area(f):
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    p = GProp_GProps()
    BRepGProp.SurfaceProperties_s(f, p)
    return abs(p.Mass())


def _faces(shape):
    out = []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        out.append(TopoDS.Face(exp.Current()))
        exp.Next()
    return out


def _face_samples(face, lin):
    BRepMesh_IncrementalMesh(face, lin, False, 0.2, True)
    loc = TopLoc_Location()
    tri = BRep_Tool.Triangulation_s(face, loc)
    if tri is None or tri.NbTriangles() == 0:
        return None, None
    tr = loc.Transformation()
    V = np.array([[*(lambda p: (p.X(), p.Y(), p.Z()))(tri.Node(i).Transformed(tr))]
                  for i in range(1, tri.NbNodes() + 1)])
    T = np.array([tri.Triangle(i).Get() for i in range(1, tri.NbTriangles() + 1)]) - 1
    A, B, C = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    w = 0.5 * np.linalg.norm(np.cross(B - A, C - A), axis=1)
    # centroid + 3 points towards each vertex: more robust coverage on slim pieces
    pts = [(A + B + C) / 3, (4 * A + B + C) / 6, (A + 4 * B + C) / 6, (A + B + 4 * C) / 6]
    return np.vstack(pts), np.tile(w, 4)


@dataclass
class LocalBuildResult:
    ok: bool = False
    method: str = "local_trim"
    shape: object = None
    reason: str = ""
    n_regions: int = 0
    pieces_kept: int = 0
    pieces_dropped: int = 0
    pieces_degenerate: int = 0
    regions_without_piece: list = field(default_factory=list)
    split_failures: list = field(default_factory=list)
    unify_failed: bool = False
    tangent_contacts: int = 0
    no_piece_diag: dict = field(default_factory=dict)
    kept_area: dict = field(default_factory=dict)
    free_edge_info: list = field(default_factory=list)
    holes_filled: int = 0
    assembly_attempts: list = field(default_factory=list)
    assembly_fuzzy: float = 0.0
    setback_corners: int = 0
    setback_radius: float = 0.0
    setback_patches: int = 0
    setback_dropped: int = 0
    setback_patch_stats: list = field(default_factory=list)
    setback_free_info: list = field(default_factory=list)
    setback_patch_failures: int = 0
    setback_patch_timeouts: int = 0
    setback_capped_by_wall: float = 0.0  # thin-wall thickness (p10) that capped the setback radius
    setback_free_edges_before: int = -1
    setback_free_edges_after: int = -1
    wireframe_healed: bool = False
    wireframe_error: str = ""
    budget_hit: str = ""             # stage that ran out of the local-build time budget (trim / setback / maker_volume)
    setback_skipped: str = ""
    faces_refixed: int = 0
    invalid_pieces: list = field(default_factory=list)
    valid_before_unify: bool = False
    volume_before_unify: float = 0.0
    invalid_faces_before_unify: list = field(default_factory=list)
    hole_fill_max_size: float = 0.0
    holes_unfilled_sizes: list = field(default_factory=list)
    free_edges_near_regions: dict = field(default_factory=dict)
    free_edge_midpoints: list = field(default_factory=list)
    free_edges: int = -1
    multiple_edges: int = -1
    n_shells: int = 0
    n_solids: int = 0
    n_faces: int = 0
    brepcheck_valid: bool = False
    volume: float = 0.0
    mesh_volume: float = 0.0
    seconds: float = 0.0

    def to_json(self):
        return {k: v for k, v in self.__dict__.items() if k != "shape" and not k.startswith("debug_")}


def build_local(mesh, labels, regions, tol, sew_tol=None, keep_frac=0.5, fuzzy=1e-6, tangent_tol=None, only=None, fill_max=None, setback=None, assembly="auto", volume_offset=0.0):
    t0 = time.time()
    from . import budget as _B
    cap = _B.stage_budget(_B.env_cap("MESHSOLID_LOCAL_BUDGET", 300))
    t_stop = t0 + cap

    def over():
        return time.time() > t_stop
    # volume_offset: material the model adds on purpose (thread zones modelled as plain cylinders), so
    # the volume check compares with mesh volume + offset instead of the raw mesh volume
    res = LocalBuildResult(n_regions=len(regions), mesh_volume=float(mesh.volume) + float(volume_offset))
    nb, lab = region_neighbours(mesh, labels)
    V = np.asarray(mesh.vertices)
    Fc = np.asarray(mesh.faces)
    diag = float(np.linalg.norm(np.ptp(V, axis=0)))
    sew_tol = sew_tol or max(1e-3, 0.5 * tol)
    tangent_tol = tangent_tol if tangent_tol is not None else tol
    fill_max = fill_max if fill_max is not None else max(0.5, 50 * tol)
    if setback is None and os.environ.get("MESHSOLID_SETBACK"):
        setback = float(os.environ["MESHSOLID_SETBACK"])   # 0 disables the (slow) setback fallback
    if setback is None:
        # ball radius at multi-blend corners: a bit more than the largest small blend radius
        br = [float(r.primitive.params.get("radius", r.primitive.params.get("minor_radius", 0)))
              for r in regions if r.primitive.kind in ("cylinder", "torus", "pipe")]
        br = [b for b in br if 0 < b < 0.05 * diag]
        setback = 2.5 * max(br) if br else 0.0
        if setback > 0:
            # thin walls (printed shells, plates): a corner ball wider than the wall cuts through to the far
            # side and the patch cannot close the hole; keep it under half the thin-wall thickness
            t10 = wall_thickness(mesh, q=10)
            if t10 > 0 and setback > 0.45 * t10:
                res.setback_capped_by_wall = round(float(t10), 4)
                setback = 0.45 * t10
    pq = NearestFace(mesh)
    rid_index = {r.id: r for r in regions}
    faces = {}
    pts_of = {}
    pipe_contacts = {}
    pipe_tool = {}
    for r in regions:
        P = V[np.unique(Fc[r.faces])]
        pts_of[r.id] = P
        rd = float(np.linalg.norm(np.ptp(P, axis=0)))
        margin = max(0.5, 0.1 * rd, 5 * tol)
        try:
            if r.primitive.kind == "pipe":
                cy = rid_index[int(r.primitive.params["support_cyl"])].primitive.params
                faces[r.id] = pipe_face_and_contacts(r.primitive, cy, P, margin)[0]
                # wider copies: splitting tools for the neighbours (contact curves for the supports)
                wf, e1, e2 = pipe_face_and_contacts(r.primitive, cy, P, max(1.0, 0.5 * rd))
                pipe_tool[r.id] = wf
                pipe_contacts[r.id] = {int(r.primitive.params["support_plane"]): e1,
                                       int(r.primitive.params["support_cyl"]): e2}
            else:
                faces[r.id] = bounded_face(r.primitive, P, margin)
        except Exception as e:  # noqa: BLE001
            res.split_failures.append({"region": r.id, "error": f"surface: {e}"})
    kept = []
    lin = max(diag / 500, tol)
    res.tangent_contacts = 0
    dbg = []
    kept_rid = []
    for i_r, r in enumerate(regions):
        if r.id not in faces or (only is not None and r.id not in only):
            continue
        if over():
            res.budget_hit = "trim"
            res.reason = f"local trim exceeded its {cap:.0f} s budget after trimming {i_r} of {len(regions)} regions"
            res.seconds = time.time() - t0
            return res
        tools = List_TopoDS_Shape()
        n_t = 0
        PA = pts_of[r.id]
        for j in sorted(nb.get(r.id, ())):
            o = rid_index[j]
            if o.primitive is r.primitive:
                continue
            PU = np.vstack([PA, pts_of[j]])
            ud = float(np.linalg.norm(np.ptp(PU, axis=0)))
            if r.primitive.kind == "pipe" and j in pipe_contacts.get(r.id, {}):
                continue   # the pipe face is already bounded by its contact curves
            if o.primitive.kind == "pipe":
                if r.id in pipe_contacts.get(j, {}):
                    tools.Append(pipe_contacts[j][r.id])
                    res.tangent_contacts += 1
                    n_t += 1
                elif j in pipe_tool:
                    tools.Append(pipe_tool[j])
                    n_t += 1
                continue
            c = contact(r.primitive, o.primitive, tangent_tol) if r.primitive.kind != "pipe" else None
            try:
                if c is not None:
                    tools.Append(contact_edge(contact_on(r.primitive, c), PU, max(1.0, 0.2 * ud)))
                    res.tangent_contacts += 1
                else:
                    if o.primitive.kind == "torus":
                        tools.Append(bounded_face(o.primitive, PU, max(1.0, 0.3 * ud, 15 * tol), P_own=pts_of[j],
                                                  own_margin=max(0.3 * float(o.primitive.params["minor_radius"]), 5 * tol)))
                    else:
                        tools.Append(bounded_face(o.primitive, PU, max(1.0, 0.3 * ud, 15 * tol)))
                n_t += 1
            except Exception as e:  # noqa: BLE001
                res.split_failures.append({"region": r.id, "neighbour": j, "error": f"tool: {e}"})
        args = List_TopoDS_Shape()
        args.Append(faces[r.id])
        if n_t:
            sp = BRepAlgoAPI_Splitter()
            sp.SetArguments(args)
            sp.SetTools(tools)
            sp.SetFuzzyValue(fuzzy)
            sp.Build()
            if not sp.IsDone():
                res.split_failures.append({"region": r.id, "error": "splitter failed"})
                continue
            pieces = _faces(sp.Shape())
        else:
            pieces = [faces[r.id]]
        got = 0
        diag_r = []
        if os.environ.get("MESHSOLID_TRIM_TRACE"):
            print(f"  trim region {r.id} {r.primitive.kind}: {len(pieces)} pieces, {n_t} tools, t={time.time() - t0:.1f}s", flush=True)
        for pc in pieces:
            if over():
                break
            # degenerate split remnants (zero-area slivers along tangent seams) are not surface; keeping
            # them breaks sewing / MakerVolume
            if _area(pc) < max(1e-6, (0.05 * tol) ** 2):
                res.pieces_degenerate += 1
                continue
            S, W = _face_samples(pc, lin)
            if S is None:
                res.pieces_dropped += 1
                continue
            if len(S) > 3000:
                sel = np.random.default_rng(0).choice(len(S), 3000, replace=False)
                S, W = S[sel], W[sel]
            dist, tri = pq.query(S)
            ok = (lab[tri] == r.id) & (dist < max(10 * tol, 0.2, getattr(r, "keep_dist", 0.0)))
            frac = float(W[ok].sum() / W.sum()) if W.sum() > 0 else 0.0
            diag_r.append((round(float(W.sum()), 3), round(frac, 3), round(float(np.median(dist)), 4)))
            if only is not None:
                dbg.append((r.id, pc, S, W, dist, lab[tri], frac))
            if frac >= keep_frac:
                kept.append(pc)
                kept_rid.append(r.id)
                from OCP.BRepCheck import BRepCheck_Analyzer as _BA
                if not _BA(pc).IsValid():
                    res.invalid_pieces.append({"region": r.id, "kind": r.primitive.kind, "area": round(_area(pc), 4)})
                got += 1
                res.kept_area[r.id] = res.kept_area.get(r.id, 0.0) + float(W.sum() / 4)
            else:
                res.pieces_dropped += 1
        res.pieces_kept += got
        if got == 0:
            res.regions_without_piece.append(r.id)
            res.no_piece_diag[r.id] = {"kind": r.primitive.kind, "area": round(r.area, 2), "n_tools": n_t,
                                       "pieces(area,frac,median_dist)": sorted(diag_r, key=lambda t: -t[1])[:6]}
    res.area_ratio = {rid: round(res.kept_area.get(rid, 0.0) / max(rid_index[rid].area, 1e-9), 3) for rid in rid_index}
    if only is not None:
        res.debug_pieces = dbg
        return res
    res.debug_kept = (list(kept), list(kept_rid))
    if not kept:
        res.reason = "no trimmed faces"
        return res
    if assembly in ("auto", "volume") and not res.regions_without_piece:
        # General-Fuse assembly (BOPAlgo_MakerVolume): the trimmed pieces are intersected
        # together with a fuzzy tolerance, so coincident or nearly coincident boundaries become
        # shared edges, then the enclosed volumes are built. Robust at multi-blend vertices
        # where independent sewing leaves tiny mismatched edges.
        for fz in (1.1 * tol, 0.75 * tol, 1.5 * tol, 0.5 * tol, 2.0 * tol, 1.0 * tol):
            if over():
                res.budget_hit = "maker_volume"
                break
            ok_mv = _assemble_volume(kept, fz, mesh, res)
            if ok_mv:
                res.method = "local_trim+maker_volume"
                res.seconds = time.time() - t0
                return res
        if assembly == "volume":
            res.seconds = time.time() - t0
            return res
    sew = BRepBuilderAPI_Sewing(sew_tol)
    for f in kept:
        sew.Add(f)
    sew.Perform()
    shell_shape = sew.SewedShape()
    if sew.NbFreeEdges() and setback > 0:
        # multi-blend corners (several fillets meeting at one vertex) are where independent
        # trimming disagrees: cut a ball of radius `setback` out of every piece at each corner
        # with open edges and close the hole with an N-sided patch fitted to the mesh there
      for _sb_round in range(3):
        if not sew.NbFreeEdges() or over():
            break
        centers = _free_edge_clusters(sew, link=setback)
        if len(centers) > SETBACK_MAX_CORNERS:
            # hundreds of open corners: the trim itself disagrees almost everywhere (e.g. a faceted skin
            # cut into strips); ball-and-patch repair cannot fix that and would run for hours
            res.setback_skipped = f"{len(centers)} open corners > {SETBACK_MAX_CORNERS}"
            break
        if centers:
            k2, r2, patches = _setback_fill(kept, kept_rid, centers, setback, V, res, pq=pq, t_stop=t_stop)
            sew2 = BRepBuilderAPI_Sewing(sew_tol)
            for f in k2 + patches:
                sew2.Add(f)
            sew2.Perform()
            res.setback_free_edges_before = int(sew.NbFreeEdges())
            res.setback_free_edges_after = int(sew2.NbFreeEdges())
            from OCP.BRepAdaptor import BRepAdaptor_Curve as _BC
            res.setback_free_info = []
            for i in range(1, sew2.NbFreeEdges() + 1):
                cc = _BC(TopoDS.Edge(sew2.FreeEdge(i)))
                a0, a1 = cc.Value(cc.FirstParameter()), cc.Value(cc.LastParameter())
                res.setback_free_info.append([[a0.X(), a0.Y(), a0.Z()], [a1.X(), a1.Y(), a1.Z()]])
            if sew2.NbFreeEdges() < sew.NbFreeEdges():
                sew = sew2
                shell_shape = sew.SewedShape()
                kept = k2 + patches
                kept_rid = r2 + [-4] * len(patches)
            else:
                break
        else:
            break
    if sew.NbFreeEdges() and fill_max > 0:
        # micro-holes where several blends meet (numerical mismatch of near-tangent
        # intersections): close each small free-boundary loop with an n-sided filling patch
        patches = _fill_small_holes(shell_shape, sew_tol, fill_max, res)
        if patches:
            sew = BRepBuilderAPI_Sewing(sew_tol)
            for f in kept + patches:
                sew.Add(f)
            sew.Perform()
            shell_shape = sew.SewedShape()
            kept_rid = kept_rid + [-2] * len(patches)
            kept = kept + patches
    if sew.NbFreeEdges():
        # heal small edges / wire gaps left at multi-blend vertices, then re-sew and re-fill
        from OCP.ShapeFix import ShapeFix_Wireframe
        fw = ShapeFix_Wireframe(shell_shape)
        fw.SetPrecision(sew_tol)
        fw.SetMaxTolerance(2 * sew_tol)
        import os as _os
        _wf = _os.environ.get("MESHSOLID_WF", "drop,small,gaps")
        fw.ModeDropSmallEdges = "drop" in _wf
        try:
            if "small" in _wf:
                fw.FixSmallEdges()
            if "gaps" in _wf:
                fw.FixWireGaps()
            healed_src = fw.Shape()
        except Exception as e:   # OCCT raises Standard_NullObject on some degenerate wires: skip this heal
            healed_src = shell_shape
            res.wireframe_error = f"{type(e).__name__}: {e}"
        healed = []
        from OCP.ShapeFix import ShapeFix_Face
        from OCP.BRepCheck import BRepCheck_Analyzer
        for f in _faces(healed_src):
            if not BRepCheck_Analyzer(f).IsValid():
                sff = ShapeFix_Face(f)
                sff.Perform()
                f = sff.Face()
                res.faces_refixed += 1
            if _area(f) > 1e-7:
                healed.append(f)
        res.wireframe_healed = True
        heal_tol = 2 * sew_tol    # healed edges carry up to 2*sew_tol (SetMaxTolerance above)
        for _round in range(3):
            sew = BRepBuilderAPI_Sewing(heal_tol if _round else sew_tol)
            for f in healed:
                sew.Add(f)
            sew.Perform()
            shell_shape = sew.SewedShape()
            if not sew.NbFreeEdges() or fill_max <= 0:
                break
            patches = _fill_small_holes(shell_shape, sew_tol, fill_max, res)
            if not patches:
                break
            healed = [f for f in _faces(shell_shape) if _area(f) > 1e-7] + patches
        kept = _faces(shell_shape)
        kept_rid = [-3] * len(kept)
    res.free_edges = int(sew.NbFreeEdges())
    res.multiple_edges = int(sew.NbMultipleEdges())
    shells = []
    exp = TopExp_Explorer(shell_shape, TopAbs_SHELL)
    while exp.More():
        shells.append(TopoDS.Shell(exp.Current()))
        exp.Next()
    res.n_shells = len(shells)
    problems = []
    if res.regions_without_piece:
        problems.append(f"{len(res.regions_without_piece)} regions produced no trimmed face")
    if res.free_edges:
        problems.append(f"{res.free_edges} free (unsewn) edges - shell not closed"
                        + (f" (corner repair stopped: {res.budget_hit} time budget)" if res.budget_hit else "")
                        + (f" (corner repair skipped: {res.setback_skipped})" if res.setback_skipped else ""))
    if res.multiple_edges:
        problems.append(f"{res.multiple_edges} non-manifold edges after sewing")
    if len(shells) != 1:
        problems.append(f"{len(shells)} shells after sewing")
    if not shells:
        res.reason = "; ".join(problems)
        return res
    if res.free_edges:
        from OCP.BRepAdaptor import BRepAdaptor_Curve
        mids = []
        for i in range(1, res.free_edges + 1):
            c = BRepAdaptor_Curve(TopoDS.Edge(sew.FreeEdge(i)))
            pm = c.Value(0.5 * (c.FirstParameter() + c.LastParameter()))
            mids.append((pm.X(), pm.Y(), pm.Z()))
        dist, tri = pq.query(np.array(mids))
        near = {}
        for l, d in zip(lab[tri], dist):
            near[int(l)] = near.get(int(l), 0) + 1
        res.free_edges_near_regions = dict(sorted(near.items(), key=lambda t: -t[1]))
        res.free_edge_midpoints = np.round(np.array(mids), 3).tolist()[:400]
        try:
            from OCP.TopAbs import TopAbs_EDGE as _E
            face_edges = []
            for f, rid in zip(kept, kept_rid):
                sf = sew.Modified(f)
                ex = TopExp_Explorer(sf, _E)
                while ex.More():
                    face_edges.append((ex.Current(), rid))
                    ex.Next()
            info = []
            for i in range(1, res.free_edges + 1):
                e = sew.FreeEdge(i)
                owners = sorted({rid for ee, rid in face_edges if ee.IsSame(e)})
                c = BRepAdaptor_Curve(TopoDS.Edge(e))
                p0, p1 = c.Value(c.FirstParameter()), c.Value(c.LastParameter())
                info.append({"owner": owners, "p0": [round(p0.X(), 3), round(p0.Y(), 3), round(p0.Z(), 3)],
                             "p1": [round(p1.X(), 3), round(p1.Y(), 3), round(p1.Z(), 3)], "near": int(lab[tri[i - 1]])})
            res.free_edge_info = info[:600]
        except Exception as ex:  # noqa: BLE001
            res.free_edge_info = [{"error": str(ex)}]
    if res.free_edges or len(shells) != 1:
        res.reason = "; ".join(problems)
        res.seconds = time.time() - t0
        res.debug_shape = shell_shape
        return res
    try:
        mk = BRepBuilderAPI_MakeSolid()
        for sh in shells:
            fx = ShapeFix_Shell(sh)
            fx.Perform()
            mk.Add(TopoDS.Shell(fx.Shell()) if fx.Shell().ShapeType() == TopAbs_SHELL else sh)
        solid = mk.Solid()
        fs = ShapeFix_Solid(solid)
        fs.Perform()
        shape = fs.Shape()
        try:
            so = TopoDS.Solid(shape) if shape.ShapeType() == TopAbs_SOLID else None
            if so is not None:
                BRepLib.OrientClosedSolid_s(so)
                shape = so
        except Exception:  # noqa: BLE001
            pass
        res.valid_before_unify = bool(U.check(shape))
        res.volume_before_unify = float(U.volume(shape))
        res.invalid_faces_before_unify = _invalid_faces(shape)
        try:
            un = ShapeUpgrade_UnifySameDomain(shape, True, True, True)
            un.Build()
            sh2 = un.Shape()
            if res.valid_before_unify and not U.check(sh2):
                res.unify_failed = True   # unify broke a valid solid: keep the un-merged faces
            else:
                shape = sh2
        except Exception:  # noqa: BLE001
            res.unify_failed = True   # keep the un-merged (still valid) faces
        res.n_solids = U.count(shape, TopAbs_SOLID)
        res.n_faces = U.count(shape, TopAbs_FACE)
        res.brepcheck_valid = U.check(shape)
        res.volume = float(U.volume(shape))
        if res.n_solids != 1:
            problems.append(f"{res.n_solids} solids")
        if not res.brepcheck_valid:
            problems.append("BRepCheck_Analyzer reports invalid shape")
        if res.volume <= 0:
            problems.append("non-positive volume")
        elif abs(res.volume - res.mesh_volume) > 0.02 * abs(res.mesh_volume):
            problems.append(f"volume {res.volume:.1f} differs from mesh volume {res.mesh_volume:.1f} by >2%")
    except Exception as e:  # noqa: BLE001
        problems.append(f"solid construction raised {type(e).__name__}: {e}")
        res.reason = "; ".join(problems)
        res.seconds = time.time() - t0
        return res
    res.shape = shape
    res.seconds = time.time() - t0
    res.ok = not problems
    res.reason = "; ".join(problems)
    return res
