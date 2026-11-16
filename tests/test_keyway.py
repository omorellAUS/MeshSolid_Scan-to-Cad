"""M9 keyways on synthetic keyed parts with known DIN/ANSI sizes."""
import pytest

from meshsolid import synthetic_keys as K
from meshsolid.pipeline import Params, run

pytest.importorskip("pymeshlab")


@pytest.mark.parametrize("name", list(K.PARTS))
def test_keyway_recognised_and_snapped(name, tmp_path):
    m, tr = K.part(name)
    p = tmp_path / f"{name}.stl"; m.export(p)
    rep, shape, *_ = run(str(p), str(tmp_path / "o.step"), str(tmp_path / "r.json"), Params())
    assert rep["status"] == "ok"
    kws = rep["keyways"]
    assert len(kws) == 1, kws
    k = kws[0]
    assert k["kind"].startswith(tr["kind"])
    assert abs(k["diameter"] - tr["d"]) < 0.05 and abs(k["width_b"] - tr["b"]) < 0.05
    dk = "depth_t1" if tr["kind"] == "shaft" else "depth_t2"
    assert abs(k[dk] - tr["t1" if tr["kind"] == "shaft" else "t2"]) < 0.05
    assert abs(k["length"] - tr["length"]) < 0.3
    want = {"round": "round", "square": "square", "through": "through"}[tr["ends"]]
    assert k["ends"] == want
    assert k["standard"]["standard"] == tr["standard"] and k["standard"]["designation"] == tr["designation"]
