"""Conversion report carries the honest target verdict beside the engine-limit result; the add-in shows both."""
import sys
from pathlib import Path

import pytest

pytest.importorskip("pymeshlab")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "fusion_addin" / "MeshSolid"))


def test_engine_report_has_target_verdict_and_addin_line(tmp_path):
    from meshsolid import synthetic as S
    from meshsolid.pipeline import Params, run
    from mslib import outcome as O
    m = S.to_scan(S.box_with_hole()[0], edge=1.0, noise=0.01)
    p = tmp_path / "b.stl"; m.export(p)
    rep, *_ = run(str(p), str(tmp_path / "o.step"), str(tmp_path / "r.json"), Params())
    v = rep["verdict"]
    assert v["target"]["result"] == "PASS" and v["engine_limit"]["result"] == "PASS"
    assert "% excluded" in v["target"]["label"] and "% excluded" in v["engine_limit"]["label"]
    line = O.verdict_line(rep)
    assert line.startswith("Target verdict: PASS") and "engine-limit result: PASS" in line
    ex = O.explain(0, rep, str(tmp_path / "none.log"), step_path=str(tmp_path / "o.step"))
    assert ex["importable"] and "Target verdict" in ex["message"]


def test_engine_verdict_refuses_pass_above_5pct_excluded():
    import numpy as np
    from meshsolid.metrology_spec import engine_verdict
    ev = np.zeros(1000, bool); ev[:60] = True
    v = engine_verdict(np.full(1000, 0.01), ev, accepted=True)
    assert v["engine_limit"]["result"] == "REFUSED" and v["target"]["result"] == "FAIL"
