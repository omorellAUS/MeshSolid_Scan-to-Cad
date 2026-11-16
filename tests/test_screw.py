"""M8 screw rotors: recognised parameters and built geometry against the known synthetic truth."""
import math

import numpy as np
import pytest
import trimesh

from meshsolid import screw as SC
from meshsolid import synthetic_screw as S
from meshsolid.gear import cyl

pytest.importorskip("pymeshlab")


def _truth_dev(spec, tr, sol):
    import igl
    from meshsolid import occ_util as U
    tm = U.shape_to_trimesh(sol, lin=0.01, ang=0.1)
    P = trimesh.transform_points(tr["clean_mesh"].vertices, tr["pose"])
    r, _, z = cyl(P, np.asarray(spec.axis_point), np.asarray(spec.axis_dir))
    k = (z > spec.z_a + 0.5) & (z < spec.z_b - 0.5) & (r > spec.r_root - 1.0)
    d = np.sqrt(igl.point_mesh_squared_distance(P[k], np.asarray(tm.vertices, float), np.asarray(tm.faces, np.int64))[0])
    return float(np.sqrt(np.mean(d ** 2))), float(np.percentile(d, 95))


_SPECS = {}


def _spec(name):
    """analyse() once per part (shared by the parameter and the build test)."""
    if name not in _SPECS:
        m, tr = S.part(name, edge=0.8)
        _SPECS[name] = (SC.analyse(m), tr)
    return _SPECS[name]


@pytest.mark.parametrize("name", ["male_4", "pump_twin", "pump_triple", "variable_pitch"])
def test_screw_params(name):
    """Recognised parameters against the known synthetic truth."""
    (spec, info), tr = _spec(name)
    assert spec is not None, info
    assert spec.lobes == tr["lobes"]
    assert spec.hand == tr["hand"]
    assert abs(spec.r_tip - tr["r_tip"]) < 0.1 and abs(spec.r_root - tr["r_root"]) < 0.15
    assert abs((spec.z_b - spec.z_a) - tr["length"]) < 0.3
    assert abs(spec.fit["wrap_deg"] - tr["wrap_deg"]) < 1.0
    if tr.get("variable_pitch"):
        assert spec.variable_pitch
        assert np.allclose(sorted(spec.lead_range_mm), sorted(tr["lead_mm"]), rtol=0.01)
    else:
        assert not spec.variable_pitch
        assert abs(spec.lead_mm - tr["lead_mm"]) / tr["lead_mm"] < 0.005


@pytest.mark.parametrize("name", ["male_4", "pump_twin", "pump_triple", "variable_pitch"])
def test_screw_build_deviation(name):
    """Built solid against the true surface (clean truth mesh vertices to the solid)."""
    from meshsolid import occ_util as U
    (spec, info), tr = _spec(name)
    assert spec is not None, info
    sol = SC.build_rotor(spec)
    assert U.check(sol)
    rms, p95 = _truth_dev(spec, tr, sol)
    assert rms < 0.04 and p95 < 0.09, (rms, p95)


def test_trapezoid_corners_found_smooth_profiles_untouched():
    (twin, _), _ = _spec("pump_twin")
    (male, _), _ = _spec("male_4")
    assert twin.fit["corners_per_lobe"] == 4          # crest and root land, both ends
    assert male.fit["corners_per_lobe"] == 0
    (vp, _), _ = _spec("variable_pitch")
    assert vp.fit["corners_per_lobe"] == 4
    (tri, _), _ = _spec("pump_triple")
    assert tri.fit["corners_per_lobe"] == 0           # cycloidal: curvature jumps, no corners


def test_straight_lobes_rejected():
    R = S.male_profile(4, 24.0, 40.0)
    m = S.rotor_mesh(R, 4, 0.0, 0.0, 80.0, shaft_r=15.0)
    s, _ = S.scanify(m, edge=1.0)
    spec, info = SC.analyse(s)
    assert spec is None and "straight" in info["reason"]


def test_round_shaft_not_a_screw():
    m = trimesh.creation.cylinder(radius=20, height=100, sections=256)
    s, _ = S.scanify(m, edge=1.0)
    spec, info = SC.analyse(s)
    assert spec is None


def test_keyed_shaft_not_a_screw():
    from meshsolid import synthetic_keys as K
    m, _ = K.part("shaft_ansi_1in", edge=1.0)
    spec, info = SC.analyse(m)
    assert spec is None and "straight" in info["reason"]
