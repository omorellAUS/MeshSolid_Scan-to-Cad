"""Build a B-rep solid from fitted primitives by cell decomposition.

All fitted surfaces (as untrimmed, oversized faces) split an enlarged bounding
box into cells (BRepAlgoAPI_Splitter).  Each cell is classified inside/outside
the input mesh by sampling interior points and evaluating the generalized
winding number of the mesh.  Inside cells are fused and same-domain faces
unified.  The result is validated (single solid, BRepCheck valid, volume>0).
Nothing is exported if validation fails.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse, BRepAlgoAPI_Splitter
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
from OCP.BRepClass3d import BRepClass3d_SolidClassifier
from OCP.BRepPrimAPI import (BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCone,
                             BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakeSphere)
from OCP.Bnd import Bnd_Box
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.GeomAbs import GeomAbs_Plane
from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
from OCP.TopAbs import TopAbs_FACE, TopAbs_IN, TopAbs_SOLID
from OCP.TopExp import TopExp_Explorer
from OCP.collections import List_TopoDS_Shape as TopTools_ListOfShape
from OCP.TopoDS import TopoDS
from OCP.gp import gp_Ax2, gp_Ax3, gp_Dir, gp_Pln, gp_Pnt

from . import occ_util as U


def winding_number(V, Fc, P, chunk=64):
    """Generalized winding number of closed-ish triangle mesh at points P (~1 inside, ~0 outside)."""
    T = V[Fc]
    out = np.empty(len(P))
    for s in range(0, len(P), chunk):
        p = P[s:s + chunk, None, None, :]
        a, b, c = T[None, :, 0] - p[:, :, 0], T[None, :, 1] - p[:, :, 0], T[None, :, 2] - p[:, :, 0]
        la, lb, lc = (np.linalg.norm(x, axis=-1) for x in (a, b, c))
        num = np.einsum("pfi,pfi->pf", a, np.cross(b, c))
        den = la * lb * lc + np.einsum("pfi,pfi->pf", a, b) * lc + \
            np.einsum("pfi,pfi->pf", b, c) * la + np.einsum("pfi,pfi->pf", c, a) * lb
        out[s:s + chunk] = np.arctan2(num, den).sum(1) / (2 * np.pi)
    return out


def _gp(p):
    return gp_Pnt(*map(float, p))


def _gd(d):
    return gp_Dir(*map(float, d))


def _ax2(p, d):
    d = np.asarray(d, float)
    t = np.array([1.0, 0, 0]) if abs(d[0]) < 0.9 else np.array([0, 1.0, 0])
    x = np.cross(d, t)
    return gp_Ax2(_gp(p), _gd(d), _gd(x))


def _lateral_face(shape, skip_planes=True):
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        f = TopoDS.Face(exp.Current())
        if not (skip_planes and BRepAdaptor_Surface(f).GetType() == GeomAbs_Plane):
            return f
        exp.Next()
    raise RuntimeError("no lateral face")


def tool_face(prim, S):
    q = prim.params
    k = prim.kind
    if k == "plane":
        n = q["normal"]
        pl = gp_Pln(gp_Ax3(_gp(q["point"]), _gd(n)))
        return BRepBuilderAPI_MakeFace(pl, -S, S, -S, S).Face()
    if k == "cylinder":
        a = q["axis"]
        base = q["point"] - a * S
        return _lateral_face(BRepPrimAPI_MakeCylinder(_ax2(base, a), float(q["radius"]), 2 * S).Shape())
    if k == "sphere":
        return _lateral_face(BRepPrimAPI_MakeSphere(_gp(q["center"]), float(q["radius"])).Shape())
    if k == "cone":
        a, ha = q["axis"], float(q["half_angle"])
        return _lateral_face(BRepPrimAPI_MakeCone(_ax2(q["apex"], a), 0.0, S * np.tan(ha), S).Shape())
    if k == "torus":
        from OCP.BRepPrimAPI import BRepPrimAPI_MakeTorus
        return _lateral_face(BRepPrimAPI_MakeTorus(_ax2(q["center"], q["axis"]), float(q["major_radius"]),
                                                   float(q["minor_radius"])).Shape())
    raise ValueError(k)


def _bbox(shape):
    b = Bnd_Box()
    BRepBndLib.Add_s(shape, b)
    lo, hi = b.CornerMin(), b.CornerMax()
    return np.array([lo.X(), lo.Y(), lo.Z(), hi.X(), hi.Y(), hi.Z()])


def interior_points(cell, rng, want=24, tries=4000):
    lo_hi = _bbox(cell)
    lo, hi = lo_hi[:3], lo_hi[3:]
    pts = []
    batch = 200
    for _ in range(tries // batch):
        for p in rng.uniform(lo, hi, (batch, 3)):
            cl = BRepClass3d_SolidClassifier(cell, _gp(p), 1e-7)
            if cl.State() == TopAbs_IN:
                pts.append(p)
                if len(pts) >= want:
                    return np.array(pts)
    return np.array(pts).reshape(-1, 3)


@dataclass
class BuildResult:
    ok: bool
    shape: object = None
    reason: str = ""
    n_tools: int = 0
    n_cells: int = 0
    n_inside: int = 0
    ambiguous_cells: list = field(default_factory=list)
    brepcheck_valid: bool = False
    n_solids: int = 0
    n_faces: int = 0
    volume: float = 0.0
    mesh_volume: float = 0.0
    seconds: float = 0.0

    def to_json(self):
        d = {k: v for k, v in self.__dict__.items() if k != "shape"}
        return d


def build_solid(mesh, prims, max_tools=80, fuzzy=1e-5, seed=0, budget_s=None):
    """budget_s: wall-time limit for the cell classification loop (default env MESHSOLID_CELL_BUDGET or
    240 s). The fallback gives up with a reason instead of running unbounded."""
    import os
    t0 = time.time()
    if budget_s is None:
        from . import budget as _B
        budget_s = _B.stage_budget(_B.env_cap("MESHSOLID_CELL_BUDGET", 240))
    res = BuildResult(ok=False, n_tools=len(prims), mesh_volume=float(mesh.volume))
    if len(prims) > max_tools:
        res.reason = f"{len(prims)} surfaces exceeds max_tools={max_tools} (cell decomposition would explode)"
        return res
    lo, hi = mesh.bounds
    diag = float(np.linalg.norm(hi - lo))
    m = 0.05 * diag + 1.0
    lo, hi = lo - m, hi + m
    box = BRepPrimAPI_MakeBox(_gp(lo), _gp(hi)).Shape()
    S = 3 * diag + 10
    args, tools = TopTools_ListOfShape(), TopTools_ListOfShape()
    args.Append(box)
    for p in prims:
        tools.Append(tool_face(p, S))
    sp = BRepAlgoAPI_Splitter()
    sp.SetArguments(args)
    sp.SetTools(tools)
    sp.SetFuzzyValue(fuzzy)
    sp.SetRunParallel(True)
    sp.Build()
    if not sp.IsDone():
        res.reason = "BRepAlgoAPI_Splitter failed"
        return res
    cells = []
    exp = TopExp_Explorer(sp.Shape(), TopAbs_SOLID)
    while exp.More():
        cells.append(TopoDS.Solid(exp.Current()))
        exp.Next()
    res.n_cells = len(cells)
    rng = np.random.default_rng(seed)
    V, Fc = mesh.vertices, mesh.faces
    keep = []
    for i, c in enumerate(cells):
        if time.time() - t0 > budget_s:
            res.reason = (f"cell decomposition exceeded its {budget_s:.0f} s budget after classifying {i} of "
                          f"{len(cells)} cells")
            res.seconds = time.time() - t0
            return res
        vol = U.volume(c)
        pts = interior_points(c, rng)
        if len(pts) == 0:
            res.ambiguous_cells.append({"cell": i, "volume": vol, "inside_frac": None,
                                        "note": "no interior sample found (sliver)"})
            continue
        w = winding_number(V, Fc, pts)
        frac = float(np.mean(w > 0.5))
        if 0.2 < frac < 0.8:
            res.ambiguous_cells.append({"cell": i, "volume": vol, "inside_frac": frac})
        if frac > 0.5:
            keep.append(c)
    res.n_inside = len(keep)
    if not keep:
        res.reason = "no cell classified inside the mesh"
        return res
    if len(keep) == 1:
        shape = keep[0]
    else:
        a, t = TopTools_ListOfShape(), TopTools_ListOfShape()
        a.Append(keep[0])
        for c in keep[1:]:
            t.Append(c)
        fu = BRepAlgoAPI_Fuse()
        fu.SetArguments(a)
        fu.SetTools(t)
        fu.SetFuzzyValue(fuzzy)
        fu.Build()
        if not fu.IsDone():
            res.reason = "fusing inside cells failed"
            return res
        shape = fu.Shape()
    un = ShapeUpgrade_UnifySameDomain(shape, True, True, True)
    un.Build()
    shape = un.Shape()
    res.n_solids = U.count(shape, TopAbs_SOLID)
    res.n_faces = U.count(shape, TopAbs_FACE)
    res.brepcheck_valid = U.check(shape)
    res.volume = float(U.volume(shape))
    res.seconds = time.time() - t0
    res.shape = shape
    problems = []
    if res.n_solids != 1:
        problems.append(f"{res.n_solids} solids instead of 1")
    if not res.brepcheck_valid:
        problems.append("BRepCheck_Analyzer reports invalid shape")
    if res.volume <= 0:
        problems.append("non-positive volume")
    elif res.mesh_volume > 0 and abs(res.volume - res.mesh_volume) > 0.02 * abs(res.mesh_volume) + 1e-9:
        # same 2 % volume gate as extrude / local / loft: a 25 mm³ sliver must not be
        # accepted for a 23 000 mm³ turned body (turned_40x30x29_a cell fallback).
        # Only when the mesh volume is known (watertight); a zero mesh volume would
        # otherwise reject every positive solid.
        problems.append(
            f"volume {res.volume:.4g} differs from mesh {res.mesh_volume:.4g} by more than 2%"
        )
    res.ok = not problems
    res.reason = "; ".join(problems)
    return res
