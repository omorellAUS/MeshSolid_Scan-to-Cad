"""Synthetic spline shafts and hubs with known parameters (M9 test data).

  involute  30 deg flat-root involute splines.  DIN 5480 sizes: reference diameter d_B, module m, teeth z,
            pitch diameter d = m z, profile shift x m = (d_B - d) / 2 - 0.55 m, shaft tip d_a1 = d_B - 0.2 m,
            hub tip (minor) d_a2 = d_B - 2 m, shaft root d_f1 = d_B - 2.2 m, hub root (major) d_f2 = d_B + 0.2 m.
            ANSI B92.1 flat root side fit: pitch P/Ps, pitch diameter N / P, external major (N + 1) / P,
            external minor (N - 1.35) / P, internal minor (N - 1) / P, internal major (N + 1.35) / P,
            basic space width = tooth thickness = pi / (2 P) at the pitch diameter.
  straight  ISO 14: N teeth of width B with parallel sides, minor d, major D.
The transverse section is an exact polygon (involutes sampled finely); parts are a spline zone joined to
plain cylindrical journals (shafts) or a plain ring (hubs)."""
from __future__ import annotations

import math

import numpy as np
from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakePolygon
from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakePrism
from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt, gp_Vec

IN = 25.4
ALPHA = math.radians(30.0)


def inv(a):
    return math.tan(a) - a


def _cyl(r, z0, h):
    return BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(0, 0, z0), gp_Dir(0, 0, 1)), r, h).Shape()


def _bool(cls, a, b):
    op = cls(a, b); op.Build(); return op.Shape()


def _prism(poly, z0, h):
    mp = BRepBuilderAPI_MakePolygon()
    for x, y in poly:
        mp.Add(gp_Pnt(float(x), float(y), z0))
    mp.Close()
    f = BRepBuilderAPI_MakeFace(mp.Wire()).Face()
    return BRepPrimAPI_MakePrism(f, gp_Vec(0, 0, h)).Shape()


def involute_section(z, m, s_ref, r_a, r_f, n_flank=40, n_arc=12):
    """Closed polygon of an external involute tooth ring: z teeth, pressure angle 30 deg at the pitch radius
    m z / 2, tooth thickness s_ref there, tip radius r_a, root radius r_f (flat root, radial below base)."""
    rp = m * z / 2; rb = rp * math.cos(ALPHA)
    psi_b = s_ref / (2 * rp) + inv(ALPHA)              # half tooth angle at the base circle
    def half(r):                                       # half tooth angle at radius r (radial below rb)
        return psi_b - (inv(math.acos(rb / r)) if r > rb else 0.0)
    pitch = 2 * math.pi / z
    rs = np.linspace(max(r_f, min(rb, r_a)), r_a, n_flank) if r_f < rb else np.linspace(r_f, r_a, n_flank)
    pts = []
    for k in range(z):
        c = k * pitch
        fl = [(r, c - half(r)) for r in rs]            # rising flank (minus side)
        if r_f < rb:
            fl = [(r_f, c - psi_b)] + fl
        tip = [(r_a, a) for a in np.linspace(c - half(r_a), c + half(r_a), n_arc)[1:-1]]
        fr = [(r, c + half(r)) for r in rs[::-1]]
        if r_f < rb:
            fr = fr + [(r_f, c + psi_b)]
        a0 = c + (psi_b if r_f < rb else half(r_f)); a1 = c + pitch - (psi_b if r_f < rb else half(r_f))
        root = [(r_f, a) for a in np.linspace(a0, a1, n_arc)[1:-1]]
        pts += fl + tip + fr + root
    return [(r * math.cos(a), r * math.sin(a)) for r, a in pts]


def straight_section(N, d, D, B, n_arc=16):
    """ISO 14 straight-sided spline: N teeth of width B (sides parallel to the tooth centre line), minor d,
    major D."""
    rf, ra = d / 2, D / 2
    pitch = 2 * math.pi / N
    pts = []
    for k in range(N):
        c = k * pitch
        a_f = math.asin(B / 2 / rf); a_a = math.asin(B / 2 / ra)
        def P(r, a):
            return (r, c + a)
        pts += [P(rf, -a_f), P(ra, -a_a)]
        pts += [P(ra, a) for a in np.linspace(-a_a, a_a, n_arc)[1:-1]]
        pts += [P(ra, a_a), P(rf, a_f)]
        pts += [P(rf, a) for a in np.linspace(a_f, pitch - a_f, n_arc)[1:-1]]
    out = []
    for r, a in pts:
        out.append((r * math.cos(a), r * math.sin(a)))
    return out


def shaft(section, r_journal, L_spline, L_j=(20.0, 25.0)):
    s = _prism(section, 0.0, L_spline)
    s = _bool(BRepAlgoAPI_Fuse, s, _cyl(r_journal, -L_j[0], L_j[0]))
    return _bool(BRepAlgoAPI_Fuse, s, _cyl(r_journal, L_spline, L_j[1]))


def hub(section, r_out, H):
    return _bool(BRepAlgoAPI_Cut, _cyl(r_out, 0.0, H), _prism(section, -1.0, H + 2))


def din5480(dB, m, z):
    d = m * z
    xm = (dB - d) / 2 - 0.55 * m
    s_ext = math.pi * m / 2 + 2 * xm * math.tan(ALPHA)           # shaft tooth thickness at d
    return dict(d=d, xm=xm, s=s_ext, da1=dB - 0.2 * m, df1=dB - 2.2 * m, da2=dB - 2 * m, df2=dB + 0.2 * m)


def b92(N, P):
    m = IN / P
    return dict(m=m, d=N * m, s=math.pi * m / 2, Do=(N + 1) * m, Dre=(N - 1.35) * m, Dri=(N - 1) * m, Dmi=(N + 1.35) * m)


def _din_shaft():
    g = din5480(30.0, 2.0, 14)
    sec = involute_section(14, 2.0, g["s"], g["da1"] / 2, g["df1"] / 2)
    return shaft(sec, g["df1"] / 2 - 1.0, 40.0), dict(kind="involute", side="external", teeth=14, module=2.0,
               standard="DIN 5480", designation="W 30 x 2 x 14", d_B=30.0, tip_diameter=g["da1"],
               root_diameter=g["df1"], length=40.0)


def _din_hub():
    g = din5480(40.0, 1.25, 30)
    s_hub_space = math.pi * 1.25 / 2 + 2 * g["xm"] * math.tan(ALPHA)   # hub space = mating shaft tooth
    sec = involute_section(30, 1.25, s_hub_space, g["df2"] / 2, g["da2"] / 2)
    return hub(sec, 35.0, 30.0), dict(kind="involute", side="internal", teeth=30, module=1.25, standard="DIN 5480",
                                      designation="N 40 x 1.25 x 30", d_B=40.0, tip_diameter=g["da2"],
                                      root_diameter=g["df2"], length=30.0)


def _b92_shaft():
    g = b92(20, 16)
    sec = involute_section(20, g["m"], g["s"], g["Do"] / 2, g["Dre"] / 2)
    return shaft(sec, g["Dre"] / 2 - 1.0, 38.1), dict(kind="involute", side="external", teeth=20, module=g["m"],
               standard="ANSI B92.1", designation="20T 16/32", diametral_pitch=16, tip_diameter=g["Do"],
               root_diameter=g["Dre"], length=38.1)


def _iso_shaft():
    sec = straight_section(8, 32.0, 36.0, 6.0)
    return shaft(sec, 15.0, 45.0), dict(kind="straight", side="external", teeth=8, standard="ISO 14",
                                        designation="8 x 32 x 36", minor=32.0, major=36.0, width=6.0, length=45.0)


def _iso_hub():
    sec = straight_section(6, 23.0, 28.0, 6.0)
    return hub(sec, 26.0, 30.0), dict(kind="straight", side="internal", teeth=6, standard="ISO 14",
                                      designation="6 x 23 x 28", minor=23.0, major=28.0, width=6.0, length=30.0)


PARTS = {"din_shaft_30": _din_shaft, "din_hub_40": _din_hub, "b92_shaft_20t": _b92_shaft,
         "iso_shaft_8x32": _iso_shaft, "iso_hub_6x23": _iso_hub}


def part(name, edge=0.6, noise=0.03, seed=0):
    """(scan mesh, truth dict) with Artec-like remesh + noise + random pose, like the other test sets."""
    from . import occ_util as U
    from .synthetic_screw import scanify
    shape, tr = PARTS[name]()
    clean = U.shape_to_trimesh(shape, lin=0.005, ang=0.05)
    m, T = scanify(clean, edge=edge, noise=noise, seed=seed)
    tr.update(clean_mesh=clean, pose=T, shape=shape)
    return m, tr
