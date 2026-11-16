"""Fillet in a child process: fork path and the Windows (spawned process, BRep file) path give the
same valid result; a missing edge is refused."""
import pytest
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCP.GProp import GProp_GProps
from OCP.BRepGProp import BRepGProp

from meshsolid import occ_child as C


def _vol(s):
    g = GProp_GProps(); BRepGProp.VolumeProperties_s(s, g); return g.Mass()


@pytest.mark.parametrize("nofork", ["0", "1"])
def test_fillet_in_child(monkeypatch, nofork):
    monkeypatch.setenv("MESHSOLID_NO_FORK", nofork)
    box = BRepPrimAPI_MakeBox(10.0, 20.0, 30.0).Shape()
    e = C._edges(box)[0]
    res = C.fillet_in_child(box, [(e, 2.0)], timeout=120)
    assert res is not None
    import math
    lost = (1 - math.pi / 4) * 2.0 ** 2          # area lost per unit length of the rounded edge
    v = _vol(res)
    assert min(abs(6000 - lost * L - v) for L in (10.0, 20.0, 30.0)) < 1e-3


def test_child_refuses_unknown_edge(tmp_path):
    import json
    from OCP.BRepTools import BRepTools
    box = BRepPrimAPI_MakeBox(10.0, 20.0, 30.0).Shape()
    BRepTools.Write_s(box, str(tmp_path / "in.brep"))
    (tmp_path / "job.json").write_text(json.dumps({"edges": [{"mid": [55.0, 55.0, 55.0], "r": 1.0}]}))
    assert C._main(str(tmp_path / "in.brep"), str(tmp_path / "job.json"), str(tmp_path / "out.brep")) == 3
