"""Spec-v2 alignment self-test, N-fold sectors, face ids and the extended deviation.csv (synthetic data only)."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from meshsolid import align_v2 as A  # noqa: E402
from meshsolid import synthetic_nfold as S  # noqa: E402
from meshsolid.metrology import pose_of  # noqa: E402

AXIS = ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))


@pytest.fixture(scope="module")
def nfold():
    solid, owner = S.part(15)
    br = S.balanced_scatter(15, 0.035, seed=3)          # per-blade rotation scatter (deg)
    bt = S.balanced_scatter(15, 0.010, seed=4)          # per-blade thickness scatter (mm)
    Tt = S.known_motion(rot_deg=0.1, axis=(0.05, -0.03, 1.0), t_mm=(0.03, -0.02, 0.015))
    scan, vb = S.scan_of(solid, owner, noise=0.013, blade_rot_deg=br, blade_thick_mm=bt, T=Tt)
    return solid, owner, scan, vb, Tt


def _err(E, scan):
    c = np.asarray(scan.vertices).mean(0)
    return pose_of(E)["rotation_deg"], float(np.linalg.norm(A.apply(E, c[None])[0] - c))


def _align(scan, solid):
    T, info = A.align_v2(scan, solid, n_fold=15, axis=AXIS)
    P = A.apply(T, A.sample_points(scan))
    dT, rinfo = A.refine_on_deviation(P, solid)
    return dT @ T, info, rinfo, A.apply(dT, P)


def test_self_test_recovers_known_rotation_and_translation(nfold):
    solid, _owner, scan, _vb, Tt = nfold
    T, info, rinfo, P = _align(scan, solid)
    assert info["n_points"] >= 500_000 and info["converged"] and not info["bound_hit"]
    rot, shift = _err(T @ Tt, scan)                       # T should undo Tt exactly
    assert rot <= 0.002 and shift <= 0.002, (rot, shift)
    assert rinfo["warning"] is None or "moved" in rinfo["warning"]
    # sign verified by direct re-measurement: the recovered pose is a measured minimum ...
    chk = A.direct_check(P[::4], solid, axis=AXIS[1])
    assert chk["ok"], chk
    # ... and applying the correction with the opposite sign measures clearly worse
    Pw = A.apply(np.linalg.inv(T), np.asarray(scan.vertices)[::4])
    r_good, *_ = A._closest(solid, A.apply(T, np.asarray(scan.vertices)[::4]))
    r_bad, *_ = A._closest(solid, Pw)
    assert np.sqrt(np.mean(r_good ** 2)) < 0.8 * np.sqrt(np.mean(r_bad ** 2))


def test_five_percent_trim_alone_is_biased_and_refinement_removes_it(nfold):
    """Documents why the refinement exists: the 5 % trimmed ICP leaves ~0.009 deg with per-blade scatter."""
    solid, _owner, scan, _vb, Tt = nfold
    T_icp, _info = A.align_v2(scan, solid, n_fold=15, axis=AXIS)
    T, *_ = _align(scan, solid)
    assert _err(T @ Tt, scan)[0] < _err(T_icp @ Tt, scan)[0]


def test_two_consecutive_runs_are_identical(nfold):
    solid, _owner, scan, _vb, _Tt = nfold
    T1, *_ = _align(scan, solid)
    T2, *_ = _align(scan, solid)
    assert np.array_equal(T1, T2)


def test_rotation_bound_half_pitch(nfold):
    solid, owner, _scan, _vb, _Tt = nfold
    z = np.array([0, 0, 1.0])
    # true offset 5 deg; a (wrong) 60-fold bound of +/-3 deg must clamp and say so
    off = S.known_motion(rot_deg=5.0, axis=(0, 0, 1), t_mm=(0, 0, 0))
    scan, _ = S.scan_of(solid, owner, noise=0.013, T=off, densify=1)
    T, info = A.align_v2(scan, solid, n_fold=60, axis=AXIS, min_points=100_000, max_iter=30)
    assert info["bound_hit"] and abs(np.degrees(A.axis_angle(T, z))) <= 3.0 + 1e-6
    # with the right 15-fold bound (+/-12 deg) the same scan is recovered, bound untouched
    T, info = A.align_v2(scan, solid, n_fold=15, axis=AXIS, min_points=100_000, max_iter=30)
    assert not info["bound_hit"] and np.degrees(A.axis_angle(T, z)) == pytest.approx(-5.0, abs=0.01)


@pytest.mark.parametrize("n", [15, 13])
def test_n_fold_estimate_and_sectors_follow_each_blade(n):
    solid, owner = S.part(n, sweep_deg=35.0)
    k, strength = A.estimate_n_fold(solid, AXIS)
    assert k == n and strength > 0.02
    scan, vb = S.scan_of(solid, owner, noise=0.0, densify=1)
    sec = A.SectorField(solid, AXIS, n)(scan.vertices)
    V = np.asarray(scan.vertices)
    mid = (vb >= 0) & (V[:, 2] > 1.0)                    # blade vertices above the hub
    got = []
    for b in range(n):
        vals, cnt = np.unique(sec[mid & (vb == b)], return_counts=True)
        assert cnt.max() / cnt.sum() > 0.99, (b, vals, cnt)   # a whole swept blade stays in one sector
        got.append(vals[np.argmax(cnt)])
    assert sorted(got) == list(range(n))


def test_metrology_v2_end_to_end_face_ids_sectors_csv(tmp_path, nfold):
    from meshsolid.metrology import run
    solid, _owner, scan, _vb, Tt = nfold
    solid.export(tmp_path / "solid.stl")
    scan.export(tmp_path / "scan.stl")
    kw = dict(align=True, align_method="v2", n_fold=15, device="cpu", do_map=False, do_zones=False,
              do_histogram=False, do_section=False, pdf=False, max_dist=0.2)
    rep = run(tmp_path / "scan.stl", tmp_path / "solid.stl", tmp_path / "a", **kw)
    rep2 = run(tmp_path / "scan.stl", tmp_path / "solid.stl", tmp_path / "b", **kw)
    assert rep["stats"] == rep2["stats"] and rep["pose"] == rep2["pose"]
    al = rep["alignment"]
    assert al["converged"] and al["direct_check"]["ok"] and al["refinement"]["n_points"] > 0
    assert rep["sectors"]["n_fold"] == 15 and len(rep["per_sector"]) == 15
    rows = list(csv.DictReader(open(rep["files"]["csv"])))
    assert {"solid_face", "sector", "deviation_mm"} <= set(rows[0])
    secs = {int(r["sector"]) for r in rows}
    assert secs == set(range(15))
    assert rep["stats"]["rms"] < 0.03                    # noise 0.013 + scatter only, pose removed


def test_step_face_ids_and_per_face_stats(tmp_path):
    from tests.test_metrology import _box_step
    from meshsolid.metrology import run
    import trimesh
    step = _box_step(tmp_path / "box.step")
    scan = trimesh.creation.box(extents=(20.0, 20.0, 20.0)).subdivide().subdivide()
    top = scan.vertices[:, 2] > 9.99
    scan.vertices[top, 2] += 0.1
    scan.export(tmp_path / "scan.stl")
    rep = run(tmp_path / "scan.stl", step, tmp_path / "o", device="cpu", do_map=False, do_zones=False,
              do_histogram=False, do_section=False, pdf=False)
    pf = rep["per_face"]
    assert pf and len(pf) == 6 and all(r["type"] == "plane" for r in pf)
    assert pf[0]["mean"] == pytest.approx(0.1, abs=0.01)          # the raised face ranks first
    rows = list(csv.DictReader(open(rep["files"]["csv"])))
    assert len({r["solid_face"] for r in rows}) == 6
