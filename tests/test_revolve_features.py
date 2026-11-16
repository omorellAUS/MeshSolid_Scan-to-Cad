"""Revolve gate: a turned body with a milled side-flat and cross-holes is still a revolve
(turned_40x30x29_a: raw coaxial 58.5 % failed the 85 % gate and fell through to a 25 mm³ cell sliver).
A rectangular block with a hole must not be accepted as turned."""
import math

import numpy as np
import pytest

from meshsolid.fit import Primitive
from meshsolid.revolve import _feature_area_credit
from meshsolid.segment import Region


def _R(i, kind, area, **params):
    return Region(id=i, faces=np.zeros(0, int), primitive=Primitive(kind=kind, params=params), area=float(area))


def test_feature_credit_recovers_turned_40x30x29_a_areas():
    """Exact region areas from the 04106c3 turned_40x30x29_a report.  Raw coaxial is 58.5 %;
    after crediting the four Z-holes, offset bores, fillets and the Z-flat, the gate passes."""
    # coaxial about Y: cone, two end planes, Ø24 and Ø12.15 cylinders (ids 0,1,3,13,15)
    regs = [
        _R(0, "cone", 71.9797),
        _R(1, "plane", 726.2184),
        _R(2, "cylinder", 204.9992),
        _R(3, "cylinder", 771.5138),
        _R(4, "torus", 27.3795),
        _R(5, "cylinder", 204.9992),
        _R(6, "cylinder", 204.9992),
        _R(7, "cylinder", 185.2731),
        _R(8, "torus", 27.3799),
        _R(9, "cylinder", 204.9992),
        _R(10, "cylinder", 185.0620),
        _R(11, "torus", 27.3608),
        _R(12, "cylinder", 34.7187),
        _R(13, "cylinder", 1090.0644),
        _R(14, "plane", 1149.9258),
        _R(15, "plane", 808.4614),
    ]
    coaxial = {0, 1, 3, 13, 15}
    tot = sum(r.area for r in regs)
    coax = sum(r.area for r in regs if r.id in coaxial)
    raw = coax / tot
    cut, flat, cred, tot_adj = _feature_area_credit(regs, coaxial, tot)
    frac = coax / tot_adj
    assert abs(raw - 0.5853) < 1e-3, raw
    assert raw < 0.85
    assert flat == pytest.approx(1149.9258, rel=1e-6)
    assert cut == pytest.approx(sum(r.area for r in regs if r.id not in coaxial and r.id != 14), rel=1e-9)
    assert cred == pytest.approx(flat, rel=1e-6)        # 19.4 % < 25 % cap
    assert frac >= 0.85 and frac == pytest.approx(1.0, abs=1e-3)


def test_feature_credit_box_with_hole_stays_below_gate():
    """A 40 x 30 x 20 block with a Ø10 through-hole: even with a 25 % side-flat credit the
    coaxial fraction stays under 85 %, so it is not taken as a turned part."""
    L, W, H, D = 40.0, 30.0, 20.0, 10.0
    hole = math.pi * D * H
    ends = 2 * (L * W - math.pi * (D / 2) ** 2)
    sides = 2 * L * H + 2 * W * H
    regs = [
        _R(0, "cylinder", hole),
        _R(1, "plane", ends / 2),
        _R(2, "plane", ends / 2),
        _R(3, "plane", L * H),
        _R(4, "plane", L * H),
        _R(5, "plane", W * H),
        _R(6, "plane", W * H),
    ]
    coaxial = {0, 1, 2}            # hole axis + the two pierced faces
    tot = sum(r.area for r in regs)
    coax = sum(r.area for r in regs if r.id in coaxial)
    _cut, _flat, _cred, tot_adj = _feature_area_credit(regs, coaxial, tot)
    assert coax / tot < 0.85
    assert coax / tot_adj < 0.85


def _turned_flat_holes():
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakeBox
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCP.gp import gp_Ax2, gp_Pnt, gp_Dir
    body = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 1, 0)), 12.0, 30.0).Shape()
    body = BRepAlgoAPI_Cut(body, BRepPrimAPI_MakeBox(gp_Pnt(9, -1, -20), gp_Pnt(25, 31, 20)).Shape()).Shape()
    for x, y in ((6.0, 8.0), (6.0, 22.0), (-6.0, 8.0), (-6.0, 22.0)):
        hole = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(x, y, -20), gp_Dir(0, 0, 1)), 2.0, 40.0).Shape()
        body = BRepAlgoAPI_Cut(body, hole).Shape()
    return body


def test_revolve_gate_passes_on_segmented_turned_body_with_flat_and_holes():
    """A Ø24 x 30 cylinder, one milled flat and four Ø4 cross-holes: the raw coaxial
    fraction is below 85 %, but after feature credit the gate passes.  (Full
    build_revolve hole-clustering is not run here: _cluster_holes can SIGSEGV on
    this synthetic tessellation in _on_surface.)"""
    from meshsolid.occ_util import shape_to_trimesh
    from meshsolid.revolve import _curve2d
    from meshsolid.segment import segment, SegParams
    body = _turned_flat_holes()
    mesh = shape_to_trimesh(body, lin=0.15, ang=0.15)
    regs, _labels, _info = segment(mesh, SegParams(tol=0.05))
    axial = [r for r in regs if r.primitive is not None and r.primitive.kind in ("cylinder", "cone", "torus")]
    assert axial, "expected a cylindrical body"
    ref = max(axial, key=lambda r: r.area)
    q = ref.primitive.params
    u = np.asarray(q["axis"], float); u = u / np.linalg.norm(u)
    a0 = np.asarray(q.get("point", q.get("apex", q.get("center"))), float)
    tot = float(sum(r.area for r in regs)) or 1.0
    curves, coax = {}, 0.0
    for r in regs:
        c = _curve2d(r.primitive, a0, u, 0.05) if r.primitive is not None else None
        if c is not None:
            curves[r.id] = c
            coax += r.area
    _cut, _flat, _cred, tot_adj = _feature_area_credit(regs, set(curves), tot)
    raw, frac = coax / tot, coax / max(tot_adj, coax)
    assert raw < 0.85, (raw, frac, {r.id: (r.primitive.kind if r.primitive else None, round(r.area, 1)) for r in regs})
    assert frac >= 0.85, (raw, frac, tot_adj)
