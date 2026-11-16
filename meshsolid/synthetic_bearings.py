"""Synthetic bearings and labyrinth seals with known parameters (M10 test data)."""
from __future__ import annotations

import math

import numpy as np
from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakePolygon, BRepBuilderAPI_MakeFace
from OCP.BRepPrimAPI import BRepPrimAPI_MakeRevol, BRepPrimAPI_MakeSphere
from OCP.gp import gp_Ax1, gp_Dir, gp_Pnt

from .standards import BEARINGS


def _revolve(rz):
    """Solid of revolution about z of a closed polygon given as (r, z) points (x = r, y = 0)."""
    poly = BRepBuilderAPI_MakePolygon()
    for r, z in rz:
        poly.Add(gp_Pnt(float(r), 0.0, float(z)))
    poly.Close()
    face = BRepBuilderAPI_MakeFace(poly.Wire()).Face()
    return BRepPrimAPI_MakeRevol(face, gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), 2 * math.pi).Shape()


def shielded_bearing(designation="6205", c=0.6, recess=0.4):
    """A sealed/shielded (2Z) bearing as a scanner sees it: one ring-shaped body (bore d, outside D,
    width B, chamfers c) whose faces carry the shield recess between the rings."""
    d, D, B = BEARINGS[designation]
    ri, ro = d / 2, D / 2
    s0, s1 = ri + 0.28 * (ro - ri), ro - 0.22 * (ro - ri)       # shield annulus
    rz = [(ri + c, 0.0), (s0, 0.0), (s0, recess), (s1, recess), (s1, 0.0), (ro - c, 0.0), (ro, c),
          (ro, B - c), (ro - c, B), (s1, B), (s1, B - recess), (s0, B - recess), (s0, B), (ri + c, B),
          (ri, B - c), (ri, c)]
    return _revolve(rz), {"designation": designation, "d": d, "D": D, "B": B}


def labyrinth_seal(bore=60.0, od=90.0, L=40.0, teeth=6, pitch=5.0, tooth_h=3.0, tip_w=0.8, flank_deg=10.0,
                   flange=(110.0, 8.0)):
    """Labyrinth seal ring: N teeth on the bore (tips at the bore radius), grooves tooth_h deep, tip land
    tip_w, flanks flank_deg from the radial; flange of diameter flange[0], thickness flange[1] at z = 0."""
    rb = bore / 2
    rg = rb + tooth_h
    fl = tooth_h * math.tan(math.radians(flank_deg))
    zs = 0.5 * (L - (teeth - 1) * pitch)                          # first tooth centre
    prof = [(rg, 0.0)]                                            # bore side, z = 0 -> L
    for k in range(teeth):
        zc = zs + k * pitch
        prof += [(rg, zc - tip_w / 2 - fl), (rb, zc - tip_w / 2), (rb, zc + tip_w / 2), (rg, zc + tip_w / 2 + fl)]
    prof += [(rg, L)]
    ro, (rf, tf) = od / 2, (flange[0] / 2, flange[1])
    outer = [(ro, L), (ro, tf), (rf, tf), (rf, 0.0)]
    rz = prof + outer
    truth = {"teeth": teeth, "pitch": pitch, "tip_diameter": bore, "groove_diameter": 2 * rg, "tooth_height": tooth_h,
             "tip_land": tip_w, "side": "bore", "length": L}
    return _revolve(rz), truth


def open_bearing(designation="6205", kind="ball", Z=9, Dw=7.938, Lw=None, c=0.6, intrude=0.15):
    """Open (unshielded) bearing as one scanned body: inner and outer ring with the rolling elements
    between them, each element sunk intrude * Dw into both rings (the scan fuses the contacts).
    kind 'ball': Z spheres of diameter Dw on the pitch circle at mid-width; 'roller': Z cylinders of
    diameter Dw and length Lw, axes parallel to the bearing axis."""
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder
    from OCP.gp import gp_Ax2
    d, D, B = BEARINGS[designation]
    ri, ro = d / 2, D / 2
    rp = 0.5 * (ri + ro)
    r_in = rp - (0.5 - intrude) * Dw; r_out = rp + (0.5 - intrude) * Dw   # ring lands either side of the elements
    inner = _revolve([(ri + c, 0.0), (r_in, 0.0), (r_in, B), (ri + c, B), (ri, B - c), (ri, c)])
    outer = _revolve([(r_out, 0.0), (ro - c, 0.0), (ro, c), (ro, B - c), (ro - c, B), (r_out, B)])
    s = _fuse(inner, outer)
    Lw = Lw or Dw
    for k in range(Z):
        a = 2 * math.pi * k / Z
        x, y = rp * math.cos(a), rp * math.sin(a)
        if kind == "ball":
            el = BRepPrimAPI_MakeSphere(gp_Pnt(x, y, B / 2), Dw / 2).Shape()
        else:
            el = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(x, y, B / 2 - Lw / 2), gp_Dir(0, 0, 1)), Dw / 2, Lw).Shape()
        s = _fuse(s, el)
    truth = {"designation": designation, "d": d, "D": D, "B": B, "elements": kind, "Z": Z, "Dw": Dw,
             "pitch_diameter": 2 * rp, **({"Lw": Lw} if kind == "roller" else {})}
    return s, truth


def _fuse(a, b):
    op = BRepAlgoAPI_Fuse(a, b); op.Build(); return op.Shape()


PARTS = {"bearing_6205_2z": lambda: shielded_bearing("6205"),
         "bearing_6308_2z": lambda: shielded_bearing("6308"),
         "seal_6t": lambda: labyrinth_seal(),
         "seal_9t": lambda: labyrinth_seal(bore=45.0, od=70.0, L=36.0, teeth=9, pitch=3.5, tooth_h=2.0, tip_w=0.5,
                                           flank_deg=0.0, flange=(85.0, 6.0)),
         "bearing_6205_open": lambda: open_bearing("6205", "ball", 9, 7.938),
         "bearing_nu205_open": lambda: open_bearing("NU205", "roller", 13, 6.5, 8.0)}


def part(name, seed=0, noise=0.02, edge=0.6):
    import trimesh
    from .synthetic import to_scan
    shape, truth = PARTS[name]()
    rng = np.random.default_rng(seed)
    T = trimesh.transformations.random_rotation_matrix(rng.random(3)); T[:3, 3] = rng.uniform(-40, 40, 3)
    return to_scan(shape, edge=edge, noise=noise, seed=seed, rotate=T), dict(truth, pose=T, shape=shape)
