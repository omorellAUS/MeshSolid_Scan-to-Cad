"""1.0.2a job guards: stay awake during a job (Windows), fall-through reasons in the engine log, stage time caps,
and Part type Impeller failing with the recogniser's reason instead of falling through (synthetic data only)."""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from meshsolid import power  # noqa: E402


def test_keep_awake_is_a_noop_off_windows(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    with power.keep_awake() as prev:
        assert prev is None


def test_keep_awake_sets_and_always_resets_on_windows(monkeypatch):
    calls = []

    class _F:
        restype = None
        argtypes = None

        def __call__(self, flags):
            calls.append(int(getattr(flags, "value", flags)))
            return 0x80000000

    fake_ctypes = types.SimpleNamespace(windll=types.SimpleNamespace(kernel32=types.SimpleNamespace(
        SetThreadExecutionState=_F())), c_uint=lambda v: types.SimpleNamespace(value=v))
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setitem(sys.modules, "ctypes", fake_ctypes)
    with pytest.raises(RuntimeError):
        with power.keep_awake():
            assert calls == [power.ES_CONTINUOUS | power.ES_SYSTEM_REQUIRED]
            raise RuntimeError("job failed")
    assert calls == [power.ES_CONTINUOUS | power.ES_SYSTEM_REQUIRED, power.ES_CONTINUOUS]


def test_recogniser_fall_through_reason_goes_to_the_engine_log(capsys):
    from meshsolid import pipeline as P
    rep = {}
    P._recogniser(rep, "impeller", "none", reason="not a bladed rotor (votes/tangential area)",
                  fell_through_to="gear / screw / prismatic")
    P._recogniser(rep, "impeller", "failed", ValueError("no crossing"), fell_through_to="gear / screw / prismatic")
    P._recogniser(rep, "impeller", "ok")
    err = capsys.readouterr().err
    assert "[recogniser] impeller: none - not a bladed rotor" in err and "falls through to gear" in err
    assert "[recogniser] impeller: failed - ValueError: no crossing" in err
    assert err.count("[recogniser]") == 2


def test_stage_caps_scale_with_the_budget():
    from meshsolid import pipeline as P
    pr = P.Params(time_budget_s=7560.0)
    assert P._stage_cap(pr, 0.10, 600.0) == 600.0 and P._stage_cap(pr, 0.20, 1200.0) == 1200.0
    pr = P.Params(time_budget_s=1800.0)
    assert P._stage_cap(pr, 0.10, 600.0) == 180.0 and P._stage_cap(pr, 0.20, 1200.0) == 360.0
    assert P._stage_cap(P.Params(), 0.10, 600.0) == 600.0


def test_coaxial_and_freeform_stop_at_their_deadline(monkeypatch):
    import trimesh
    from meshsolid.coaxial import coaxial_refit
    from meshsolid import freeform as FR
    from meshsolid.segment import MeshGraph
    m = trimesh.creation.icosphere(2)
    g = MeshGraph(m)
    labels = np.zeros(len(m.faces), int)
    regs = [types.SimpleNamespace(id=0, faces=np.arange(len(m.faces)), primitive=None)]
    log = coaxial_refit(g, labels, regs, 0.02, deadline=0.0)
    assert log and "time cap" in log[-1]["stopped"]
    monkeypatch.setattr(FR, "smooth_clusters", lambda *a, **k: [[0], [1]])
    _r, _l, flog = FR.apply_freeform(m, g, regs, labels, 0.02, None, deadline=0.0)
    assert flog and flog[-1]["stopped"].startswith("time cap") and not flog[-1]["accepted"]


def test_part_type_impeller_fails_with_reason_instead_of_falling_through(tmp_path, capsys):
    """A plain noisy cylinder with --impeller on: the probe finds no blades, the run fails at once with that
    reason (no segmentation / coaxial / freeform)."""
    import trimesh
    from meshsolid import pipeline as P
    m = trimesh.creation.cylinder(radius=20.0, height=30.0, sections=180)
    m = m.subdivide().subdivide()
    rng = np.random.default_rng(0)
    m.vertices = m.vertices + rng.normal(0, 0.01, m.vertices.shape)
    stl = tmp_path / "cyl.stl"
    m.export(stl)
    rep, shape, *_ = P.run(str(stl), str(tmp_path / "o.step"), str(tmp_path / "o.json"),
                           P.Params(impeller="on", gears="off", screws="off"))
    assert rep["status"] == "failed" and shape is None
    assert rep["failure"].startswith("Part type Impeller: the impeller recogniser could not build this part")
    assert "segmentation" not in rep.get("stages", {}) and "coaxial_refit" not in rep.get("stages", {})
    assert "falls through to none (Part type Impeller)" in capsys.readouterr().err


def test_part_type_impeller_maps_to_impeller_on():
    sys.path.insert(0, str(ROOT / "fusion_addin" / "MeshSolid"))
    from mslib import ui_options as UO
    assert UO.engine_options("impeller")["impeller"] == "on"
    assert UO.VERSION == "2.0.0"
