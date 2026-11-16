"""Rim slots (step 7): flat-floored pockets in the back face open to the rim are detected from the scan and
cut into the solid.  Synthetic disc R 80 x 10 with 4 slots (STG1-like: floors 2.85 mm deep, 196-320 deg)."""
import numpy as np
import pytest
import trimesh

from meshsolid import rim_slots as RS

R, H = 80.0, 10.0
SLOTS = [(200.0, 6.0), (240.0, 5.0), (280.0, 6.0), (315.0, 4.0)]     # centre deg, width mm
R_IN, DEPTH = 74.0, 2.85


def _disc(slots=SLOTS):
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder
    s = BRepPrimAPI_MakeCylinder(R, H).Shape()
    sl = [{"centre_deg": c, "width_mm": w, "r_inner": R_IN, "z_floor": DEPTH, "z_end": 0.0, "material_side": 1}
          for c, w in slots]
    return s, (RS.cut(s, [0, 0, 0], [0, 0, 1], [1, 0, 0], sl, R) if slots else s)


def _scan(shape, noise=0.008, seed=0):
    from meshsolid.occ_util import shape_to_trimesh
    m = shape_to_trimesh(shape, lin=0.01, ang=0.05)
    v, f = trimesh.remesh.subdivide_to_size(m.vertices, m.faces, max_edge=0.6, max_iter=12)
    m = trimesh.Trimesh(v, f, process=True)
    rng = np.random.default_rng(seed)
    m.vertices = m.vertices + rng.normal(0, noise, m.vertices.shape)
    return m


@pytest.fixture(scope="module")
def slotted():
    plain, cut = _disc()
    return plain, cut, _scan(cut)


def test_detects_slots_angle_width_depth(slotted):
    _, _, scan = slotted
    rep, _ = RS.detect(np.asarray(scan.vertices), np.asarray(scan.vertex_normals), R)
    sl = sorted(rep["slots"], key=lambda x: x["centre_deg"])
    assert rep["confident"] and len(sl) == 4, rep
    for got, (c, w) in zip(sl, SLOTS):
        assert abs(got["centre_deg"] - c) < 0.3, got
        assert abs(got["width_mm"] - w) < 0.15, got
        assert abs(got["depth_mm"] - DEPTH) < 0.03, got
        assert abs(got["r_inner"] - R_IN) < 0.3, got
        assert got["open_to_rim"] and got["material_side"] == 1


def test_apply_cuts_one_valid_solid_matching_truth(slotted):
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.TopAbs import TopAbs_SOLID
    from meshsolid import occ_util as U
    plain, cut, scan = slotted
    new, rep = RS.apply(scan, plain, [0, 0, 0], [0, 0, 1], [1, 0, 0], R, flag="auto", log=lambda *a: None)
    assert rep["modelled"] and rep["n_cut"] == 4, rep
    assert BRepCheck_Analyzer(new).IsValid() and U.count(new, TopAbs_SOLID) == 1
    removed_truth = U.volume(plain) - U.volume(cut)
    removed = U.volume(plain) - U.volume(new)
    assert abs(removed - removed_truth) < 0.03 * removed_truth


def test_plain_disc_no_slots_and_flag_off():
    plain, _ = _disc(slots=[])
    scan = _scan(plain, seed=1)
    rep, _ = RS.detect(np.asarray(scan.vertices), np.asarray(scan.vertex_normals), R)
    assert rep["slots"] == [] and not rep["confident"]
    s, rep = RS.apply(scan, plain, [0, 0, 0], [0, 0, 1], [1, 0, 0], R, flag="off")
    assert s is plain and "skipped" in rep


def test_rough_floor_is_not_confident():
    """A scalloped (balance-cut-like) pocket: floor depth varies -> listed, not cut in auto mode."""
    rng = np.random.default_rng(3)
    th = np.radians(rng.uniform(250, 256, 3000)); r = rng.uniform(75, 80, 3000)
    z = 1.0 + 0.8 * np.sin((r - 75) * 1.2)                  # wavy floor
    P = np.c_[r * np.cos(th), r * np.sin(th), z]; N = np.tile([0, 0, -1.0], (len(P), 1))
    rep, _ = RS.detect(P, N, R, planes=[(0.0, 1)])
    assert rep["slots"] and not rep["confident"]


def test_rim_rebate_ring_is_not_a_slot():
    """An annular rebate at the rim (turned step, 360 deg) must not be cut as a slot."""
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    s = BRepPrimAPI_MakeCylinder(R, H).Shape()
    ring = BRepAlgoAPI_Cut(BRepPrimAPI_MakeCylinder(R + 1, 2.0).Shape(), BRepPrimAPI_MakeCylinder(R - 5, 2.0).Shape()).Shape()
    s = BRepAlgoAPI_Cut(s, ring).Shape()
    scan = _scan(s, seed=2)
    rep, _ = RS.detect(np.asarray(scan.vertices), np.asarray(scan.vertex_normals), R)
    assert rep["slots"] == []


def test_auto_cuts_only_confident_slots():
    """A wavy (non-flat) scallop next to real milled slots is listed and rejected; auto still cuts
    the confident slots.  Matches STG1: 4 back-face patches, only the flat-floored one is cut."""
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeSphere
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.TopAbs import TopAbs_SOLID
    from OCP.gp import gp_Pnt
    from meshsolid import occ_util as U
    plain, two = _disc(slots=SLOTS[:2])
    # spherical pocket on the end face, open to the rim, well away from the two milled slots
    th = np.radians(80.0)
    cx, cy = (R - 3.0) * np.cos(th), (R - 3.0) * np.sin(th)
    mixed = BRepAlgoAPI_Cut(two, BRepPrimAPI_MakeSphere(gp_Pnt(float(cx), float(cy), 0.0), 6.0).Shape()).Shape()
    scan = _scan(mixed, seed=4)
    rep, _ = RS.detect(np.asarray(scan.vertices), np.asarray(scan.vertex_normals), R)
    conf = [s for s in rep["slots"] if s["confident"]]
    bad = [s for s in rep["slots"] if not s["confident"]]
    assert len(conf) == 2, rep["slots"]
    assert bad and not rep["confident"], rep["slots"]
    assert all("floor_rms" in s["reject_reason"] for s in bad), bad
    new, arep = RS.apply(scan, plain, [0, 0, 0], [0, 0, 1], [1, 0, 0], R, flag="auto", log=lambda *a: None)
    assert arep["modelled"] and arep["n_cut"] == 2, arep
    assert BRepCheck_Analyzer(new).IsValid() and U.count(new, TopAbs_SOLID) == 1
    removed_truth = U.volume(plain) - U.volume(two)
    removed = U.volume(plain) - U.volume(new)
    assert abs(removed - removed_truth) < 0.05 * removed_truth
    # the scallop must not have been cut: the solid still occupies that sector at the end face
    assert U.volume(new) > U.volume(mixed) + 0.2 * (U.volume(two) - U.volume(mixed))
