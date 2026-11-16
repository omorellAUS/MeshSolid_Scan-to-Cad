"""M4 freeform: B-spline fits (tube and patch), OCC conversion, and the ovalised pipe elbow end to end."""
import numpy as np
import pytest

from meshsolid import freeform as FF


def _oval_bend(n=20000, seed=0):
    rng = np.random.default_rng(seed)
    s, ph = rng.uniform(0, np.pi / 2, n), rng.uniform(0, 2 * np.pi, n)
    R, r = 40.0, 10 * (1 + 0.06 * np.sin(2 * s) * np.cos(2 * ph))
    P = np.c_[(R + r * np.cos(ph)) * np.cos(s), r * np.sin(ph), (R + r * np.cos(ph)) * np.sin(s)]
    c = np.c_[R * np.cos(s), np.zeros_like(s), R * np.sin(s)]
    N = (P - c) / np.linalg.norm(P - c, axis=1)[:, None]
    return P + rng.normal(0, 0.02, P.shape), N


def test_tube_fit_and_occ():
    P, N = _oval_bend()
    prim, info = FF.fit_freeform(P, N, 0.1, 2.0)
    assert info["param"] == "tube"
    assert info["rms_true"] < 0.05 and info["p95_true"] < 0.1
    S = FF.to_occ(prim)               # raises if OCC does not reproduce the fit
    assert S.IsVPeriodic()


def test_patch_fit():
    rng = np.random.default_rng(1)
    x = rng.uniform(-20, 20, (5000, 2))
    P = np.c_[x, 2 * np.sin(x[:, 0] / 7) * np.cos(x[:, 1] / 9)]
    prim, info = FF.fit_freeform(P, np.tile([0, 0, 1.0], (5000, 1)), 0.1, 2.0, kind="patch")
    assert info["rms_true"] < 0.05
    FF.to_occ(prim)


@pytest.mark.slow
def test_pipe_elbow_end_to_end(tmp_path):
    from meshsolid import synthetic as S
    from meshsolid.pipeline import Params, run
    shape, truth = S.pipe_elbow()
    m = S.to_scan(shape, noise=0.02, seed=1)
    stl = tmp_path / "elbow.stl"
    m.export(stl)
    rep, *_ = run(stl, tmp_path / "elbow.step", tmp_path / "elbow.json", Params())
    assert rep["status"] == "ok"
    assert rep["stages"]["freeform"]["accepted"] == 2
    b = rep["stages"]["build"]
    assert b["brepcheck_valid"]
    assert abs(b["volume"] - truth["volume"]) < 0.002 * truth["volume"]
