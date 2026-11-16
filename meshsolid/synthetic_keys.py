"""Synthetic keyed shafts and hubs with known keyway sizes (M9 test data)."""
from __future__ import annotations

import math

import numpy as np
from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt


def _cyl(r, z0, h, x=0.0, y=0.0, axis=(0, 0, 1)):
    return BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(x, y, z0), gp_Dir(*axis)), r, h).Shape()


def _cut(a, b):
    op = BRepAlgoAPI_Cut(a, b); op.Build(); return op.Shape()


def _fuse(a, b):
    op = BRepAlgoAPI_Fuse(a, b); op.Build(); return op.Shape()


def keyed_shaft(d=40.0, L=120.0, b=12.0, t1=5.0, key_len=50.0, z_key=35.0, end_milled=True, d2=None, L2=30.0):
    """Shaft of diameter d (z = 0..L) with a keyseat of width b and depth t1 (from the top of the circle,
    the DIN 6885 convention) on +x, from z_key to z_key + key_len; round ends when end_milled.
    Optional journal step of diameter d2 and length L2 at z = L."""
    s = _cyl(d / 2, 0.0, L)
    if d2:
        s = _fuse(s, _cyl(d2 / 2, L, L2))
    x0 = d / 2 - t1
    if end_milled:
        straight = key_len - b
        box = BRepPrimAPI_MakeBox(gp_Pnt(x0, -b / 2, z_key + b / 2), d, b, straight).Shape()
        c1 = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(x0, 0, z_key + b / 2), gp_Dir(1, 0, 0)), b / 2, d).Shape()
        c2 = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(x0, 0, z_key + key_len - b / 2), gp_Dir(1, 0, 0)), b / 2, d).Shape()
        tool = _fuse(_fuse(box, c1), c2)
    else:
        tool = BRepPrimAPI_MakeBox(gp_Pnt(x0, -b / 2, z_key), d, b, key_len).Shape()
    return _cut(s, tool)


def keyed_hub(d=40.0, D=80.0, H=40.0, b=12.0, t2=3.3):
    """Hub of bore d, outside D, length H with a through keyway of width b and depth t2 into the hub."""
    s = _cut(_cyl(D / 2, 0.0, H), _cyl(d / 2, -1.0, H + 2))
    box = BRepPrimAPI_MakeBox(gp_Pnt(0.0, -b / 2, -1.0), d / 2 + t2, b, H + 2).Shape()
    return _cut(s, box)


PARTS = {
    "shaft_din_40": (lambda: keyed_shaft(40.0, 120.0, 12.0, 5.0, 50.0, 35.0, True, 30.0, 30.0),
                     {"kind": "shaft", "d": 40.0, "b": 12.0, "t1": 5.0, "length": 50.0, "ends": "round",
                      "standard": "DIN 6885-1", "designation": "12 x 8"}),
    "shaft_ansi_1in": (lambda: keyed_shaft(25.4, 100.0, 6.35, 3.175, 38.1, 20.0, False),
                       {"kind": "shaft", "d": 25.4, "b": 6.35, "t1": 3.175, "length": 38.1, "ends": "square",
                        "standard": "ANSI B17.1 square", "designation": "1/4 in"}),
    "hub_din_40": (lambda: keyed_hub(40.0, 80.0, 40.0, 12.0, 3.3),
                   {"kind": "hub", "d": 40.0, "b": 12.0, "t2": 3.3, "length": 40.0, "ends": "through",
                    "standard": "DIN 6885-1", "designation": "12 x 8"}),
}


def part(name, seed=0, noise=0.02, edge=0.8):
    import trimesh
    from .synthetic import to_scan
    fn, truth = PARTS[name]
    shape = fn()
    rng = np.random.default_rng(seed)
    T = trimesh.transformations.random_rotation_matrix(rng.random(3)); T[:3, 3] = rng.uniform(-40, 40, 3)
    return to_scan(shape, edge=edge, noise=noise, seed=seed, rotate=T), dict(truth, pose=T, shape=shape)
