"""Faceted CAD STL test parts (like 3D-print exports of the drone/VTOL set): exact vertices on the true
surface, coarse facets with visible chord sag, long thin triangles along extruded skins.

  wing_section   NACA 00xx airfoil (B-spline through the analytic thickness distribution, closed by a
                 small trailing-edge flat), extruded along the span
  thin_shell     open-top box shell with 1.2 mm walls (print-style thin wall)
"""
from __future__ import annotations

import math

import numpy as np


def _naca(t, c, n=60):
    x = c * (1 - np.cos(np.linspace(0, math.pi, n))) / 2
    xx = x / c
    yt = 5 * t * c * (0.2969 * np.sqrt(xx) - 0.1260 * xx - 0.3516 * xx ** 2 + 0.2843 * xx ** 3 - 0.1015 * xx ** 4)
    return x, yt


def wing_section(chord=60.0, t=0.12, span=40.0):
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeWire
    from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
    from OCP.GeomAPI import GeomAPI_PointsToBSpline
    from OCP.collections import Array1_gp_Pnt as TColgp_Array1OfPnt
    from OCP.gp import gp_Pnt, gp_Vec
    x, y = _naca(t, chord)
    up = list(zip(x[::-1], y[::-1]))            # TE -> LE on the upper side
    lo = list(zip(x[1:], -y[1:]))               # LE -> TE lower side
    pts = up + lo
    arr = TColgp_Array1OfPnt(1, len(pts))
    for i, (a, b) in enumerate(pts):
        arr.SetValue(i + 1, gp_Pnt(float(a), float(b), 0.0))
    bs = GeomAPI_PointsToBSpline(arr, 3, 8).Curve()
    e1 = BRepBuilderAPI_MakeEdge(bs).Edge()
    p0, p1 = gp_Pnt(*map(float, (*pts[0], 0))), gp_Pnt(*map(float, (*pts[-1], 0)))
    e2 = BRepBuilderAPI_MakeEdge(p1, p0).Edge()                  # trailing-edge flat
    w = BRepBuilderAPI_MakeWire(e1, e2).Wire()
    f = BRepBuilderAPI_MakeFace(w).Face()
    sh = BRepPrimAPI_MakePrism(f, gp_Vec(0, 0, span)).Shape()
    return sh, dict(chord=chord, thickness=t, span=span)


def thin_shell(L=60.0, W=40.0, H=20.0, wall=1.2):
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.gp import gp_Pnt
    outer = BRepPrimAPI_MakeBox(L, W, H).Shape()
    inner = BRepPrimAPI_MakeBox(gp_Pnt(wall, wall, wall), L - 2 * wall, W - 2 * wall, H).Shape()
    op = BRepAlgoAPI_Cut(outer, inner); op.Build()
    return op.Shape(), dict(L=L, W=W, H=H, wall=wall)


def faceted_mesh(shape, lin=0.05, ang=0.35):
    """CAD-export-like tessellation: coarse angular deflection (long facets along extrusions)."""
    from . import occ_util as U
    return U.shape_to_trimesh(shape, lin=lin, ang=ang)
