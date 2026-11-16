"""A recogniser that raises never ends the run: it becomes a 'failed' entry and the next method builds."""
import pytest

pytest.importorskip("pymeshlab")


def test_impeller_exception_falls_through(tmp_path, monkeypatch):
    from meshsolid import synthetic as S, impeller as I, impeller_run as IR
    from meshsolid.pipeline import Params, run
    m = S.to_scan(S.box_with_hole()[0], edge=1.0, noise=0.01)
    p = tmp_path / "b.stl"; m.export(p)
    monkeypatch.setattr(I, "probe", lambda mesh: (16, {"reason": "forced"}))

    def boom(*a, **k):
        raise ValueError("no crossing")
    monkeypatch.setattr(IR, "run_impeller", boom)
    rep, *_ = run(str(p), str(tmp_path / "o.step"), str(tmp_path / "r.json"), Params())
    rec = {r["name"]: r for r in rep["recognisers"]}
    assert rec["impeller"]["status"] == "failed" and "no crossing" in rec["impeller"]["error"]
    assert rec["impeller"]["where"] and rep["status"] == "ok"                 # the general method built it
    assert any("impeller recognition failed" in w for w in rep.get("warnings", []))
