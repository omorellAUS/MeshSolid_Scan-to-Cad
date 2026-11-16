"""Step 8: solid -> scan gate.  A solid with a boss the scan does not have must FAIL; a scan with an
uncaptured hole must not be penalised; a matching pair passes."""
import numpy as np
import trimesh

from meshsolid.support_gate import solid_to_scan


def _box(ext=(40, 30, 10)):
    return trimesh.creation.box(extents=ext)


def _dense(m, size=0.5, noise=0.005, seed=0):
    v, f = trimesh.remesh.subdivide_to_size(m.vertices, m.faces, max_edge=size, max_iter=12)
    m = trimesh.Trimesh(v, f, process=True)
    m.vertices = m.vertices + np.random.default_rng(seed).normal(0, noise, m.vertices.shape)
    return m


def test_matching_pair_passes():
    g = solid_to_scan(_dense(_box()), _box(), {"n_samples": 50000})
    assert g["result"] == "PASS" and g["unsupported_pct"] < 0.1


def test_extra_boss_in_solid_fails():
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse
    from OCP.gp import gp_Pnt, gp_Ax2, gp_Dir
    from meshsolid.occ_util import shape_to_trimesh
    box = BRepPrimAPI_MakeBox(gp_Pnt(-20, -15, -5), gp_Pnt(20, 15, 5)).Shape()
    boss = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(5, 0, 4.0), gp_Dir(0, 0, 1)), 4.0, 7.0).Shape()   # 6 mm proud
    solid = shape_to_trimesh(BRepAlgoAPI_Fuse(box, boss).Shape(), lin=0.01, ang=0.1)
    g = solid_to_scan(_dense(_box()), solid, {"n_samples": 50000})
    assert g["result"] == "FAIL", g
    p = g["largest_unsupported_patch"]
    assert p["area_mm2"] > 50 and abs(p["centre"][0] - 5) < 1.5 and p["max_mm"] > 3


def test_scan_hole_is_not_penalised():
    scan = _dense(_box())
    c = scan.triangles_center
    keep = ~((np.abs(c[:, 0] - 5) < 6) & (np.abs(c[:, 1]) < 6) & (c[:, 2] > 4.9))   # 12 x 12 mm hole in the top skin
    scan = trimesh.Trimesh(scan.vertices, scan.faces[keep], process=True)
    g = solid_to_scan(scan, _box(), {"n_samples": 50000})
    assert g["result"] == "PASS", g
    assert g["uncaptured_pct"] > 0.5 and g["scan_boundary_vertices"] > 0
