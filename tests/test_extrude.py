"""2.5D extrusion builder: plates / stepped printed parts from CAD tessellation."""
import numpy as np
import pytest
import trimesh

from meshsolid.extrude import build_extrusion, prismatic_axis, split_profile, simplify_loop


def _tess(shape, lin=0.05):
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
    from OCP.TopoDS import TopoDS
    from OCP.BRep import BRep_Tool
    from OCP.TopLoc import TopLoc_Location
    BRepMesh_IncrementalMesh(shape, lin, False, 0.3, True)
    V, F = [], []
    ex = TopExp_Explorer(shape, TopAbs_FACE)
    while ex.More():
        f = TopoDS.Face(ex.Current()); ex.Next()
        loc = TopLoc_Location(); t = BRep_Tool.Triangulation_s(f, loc)
        o = len(V)
        for i in range(1, t.NbNodes() + 1):
            p = t.Node(i).Transformed(loc.Transformation()); V.append([p.X(), p.Y(), p.Z()])
        for i in range(1, t.NbTriangles() + 1):
            a, b, c = t.Triangle(i).Get()
            F.append([o + a - 1, o + c - 1, o + b - 1] if f.Orientation() == TopAbs_REVERSED else [o + a - 1, o + b - 1, o + c - 1])
    m = trimesh.Trimesh(np.array(V), np.array(F), process=True)
    m.merge_vertices()
    return m


def _plate(stepped=False):
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
    from OCP.gp import gp_Ax2, gp_Pnt, gp_Dir
    s = BRepPrimAPI_MakeBox(80.0, 50.0, 1.6).Shape()
    for x, y, r in ((15, 15, 3.0), (65, 15, 3.0), (40, 35, 6.0)):
        s = BRepAlgoAPI_Cut(s, BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(x, y, -1), gp_Dir(0, 0, 1)), r, 5).Shape()).Shape()
    # stadium slot: two circles + box
    slot = BRepAlgoAPI_Fuse(BRepPrimAPI_MakeBox(gp_Pnt(25, 10, -1), 20, 6, 5).Shape(),
                            BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(25, 13, -1), gp_Dir(0, 0, 1)), 3, 5).Shape()).Shape()
    slot = BRepAlgoAPI_Fuse(slot, BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(45, 13, -1), gp_Dir(0, 0, 1)), 3, 5).Shape()).Shape()
    s = BRepAlgoAPI_Cut(s, slot).Shape()
    if stepped:   # a 4 mm boss on top (second slab)
        boss = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(65, 38, 1.6), gp_Dir(0, 0, 1)), 7.0, 4.0).Shape()
        s = BRepAlgoAPI_Fuse(s, boss).Shape()
    return s


def test_split_profile_circle_and_square():
    t = np.linspace(0, 2 * np.pi, 33)[:-1]
    circ = np.c_[5 * np.cos(t), 5 * np.sin(t)]
    segs = split_profile(simplify_loop(circ, 1e-6), 0.02)
    assert all(s[0] == "arc" for s in segs) and 2 <= len(segs) <= 3
    sq = np.array([[0, 0], [5, 0], [10, 0], [10, 10], [0, 10]], float)
    segs = split_profile(simplify_loop(sq, 1e-6), 0.02)
    assert [s[0] for s in segs] == ["line"] * 4


@pytest.mark.parametrize("stepped", [False, True])
def test_plate_builds_as_extrusion(stepped):
    shape = _plate(stepped)
    m = _tess(shape)
    d, frac = prismatic_axis(m)
    assert frac > 0.999 and abs(abs(d[2]) - 1) < 1e-6
    r = build_extrusion(m, tol=0.05)
    assert r.ok, r.reason
    assert r.n_solids == 1 and r.brepcheck_valid
    arcs = sum(s["arcs"] for s in r.slabs)
    assert arcs >= 7                               # 3 holes (>=2 arcs each) + slot ends, true circles
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    p = GProp_GProps(); BRepGProp.VolumeProperties_s(shape, p)
    assert abs(r.volume - p.Mass()) / p.Mass() < 2e-3   # vs the CAD truth, not the chordal mesh
    assert r.n_faces <= 30


def test_non_prismatic_rejected():
    m = trimesh.creation.icosphere(subdivisions=3, radius=10)
    r = build_extrusion(m, tol=0.05)
    assert not r.ok and "not a 2.5D" in r.reason


def test_pipeline_uses_extrusion_for_plate(tmp_path):
    from meshsolid.pipeline import run, Params
    m = _tess(_plate(True))
    p = tmp_path / "plate.stl"; m.export(p)
    rep, *_ = run(str(p), str(tmp_path / "plate.step"), str(tmp_path / "plate.json"), Params())
    assert rep["status"] == "ok", rep.get("failure")
    assert rep["stages"]["build"]["method"] == "extrude_profile"
    assert rep["deviation"]["acceptance"]["passed"]


def test_multi_body_stl_builds_per_body(tmp_path):
    """Two separate printed parts in one STL: each body is built on its own, one multi-body STEP."""
    from meshsolid.pipeline import run, Params
    from meshsolid.occ_util import read_step
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_SOLID
    a = _tess(_plate(False))
    b = _tess(_plate(True)); b.apply_translation([0, 70, 0])
    tiny = trimesh.creation.box([0.05, 0.05, 0.05]); tiny.apply_translation([200, 0, 0])   # debris
    m = trimesh.util.concatenate([a, b, tiny])
    p = tmp_path / "two.stl"; m.export(p)
    rep, *_ = run(str(p), str(tmp_path / "two.step"), str(tmp_path / "two.json"), Params())
    assert rep["status"] == "ok", rep.get("failure")
    mb = rep["multi_body"]
    assert len(mb["bodies"]) == 2 and mb["built"] == 2 and len(mb["debris"]) == 1
    sh = read_step(str(tmp_path / "two.step"))
    ex = TopExp_Explorer(sh, TopAbs_SOLID); n = 0
    while ex.More():
        n += 1; ex.Next()
    assert n == 2
