"""M9 splines: recognised form, size and standard against synthetic parts with known parameters."""
import pytest

from meshsolid import spline as SP
from meshsolid import synthetic_splines as S

pytest.importorskip("pymeshlab")


@pytest.mark.parametrize("name", list(S.PARTS))
def test_spline_recognised_and_snapped(name):
    m, tr = S.part(name)
    sp, info = SP.analyse(m)
    assert sp is not None, info
    assert sp["teeth"] == tr["teeth"] and sp["side"] == tr["side"]
    assert sp["standard"] == tr["standard"] and sp["designation"] == tr["designation"], sp
    assert sp["flank_rms_mm"] < 0.06                       # noise is 0.03
    assert abs(sp["length"] - tr["length"]) < 0.5
    if tr["kind"] == "involute":
        assert sp["form"].startswith("involute") and abs(sp["module"] - tr["module"]) < 1e-6
        tip, root = sorted([sp["major_diameter"], sp["minor_diameter"]], reverse=tr["side"] == "external")
        assert abs(tip - tr["tip_diameter"]) < 0.1 and abs(root - tr["root_diameter"]) < 0.1
    else:
        assert sp["form"] == "straight-sided"
        assert abs(sp["width_B"] - tr["width"]) < 0.05
        assert abs(sp["minor_diameter"] - tr["minor"]) < 0.05 and abs(sp["major_diameter"] - tr["major"]) < 0.05


def test_twisted_lobes_are_not_splines():
    from meshsolid import synthetic_screw as SS
    m, _ = SS.part("male_4", edge=1.0)                     # twisted lobes
    sp, info = SP.analyse(m)
    assert sp is None
