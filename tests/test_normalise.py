"""Normalise stage: exact 4x4 in the report; 'apply' builds axis-up / nose-up and maps the outputs back."""
import numpy as np
import pytest
import trimesh

pytest.importorskip("pymeshlab")


def test_estimate_axis_nose_units():
    from meshsolid.normalise import estimate
    cone = trimesh.creation.cone(radius=20.0, height=30.0, sections=128)       # nose (apex) at +Z
    T = trimesh.transformations.rotation_matrix(1.1, [1, 2, 0.5]); T[:3, 3] = [5, -7, 12]
    m = cone.copy(); m.apply_transform(T)
    N, info, _ = estimate(m)
    q = trimesh.transform_points(m.vertices, N)
    apex = trimesh.transform_points(np.array([[0, 0, 30.0]]), N @ T)[0]
    assert info["rotational"] and info["units"] == "mm"
    assert abs(q[:, 2].min()) < 1e-6 and apex[2] == pytest.approx(30.0, abs=1e-3)   # base at z=0, nose up
    assert np.hypot(apex[0], apex[1]) < 1e-3                                       # on the axis
    small = cone.copy(); small.apply_scale(0.001)                                    # an export in metres
    N2, info2, _ = estimate(small)
    assert "m (" in info2["units"] and np.cbrt(np.linalg.det(N2[:3, :3])) == pytest.approx(1000.0)


def test_apply_round_trips_outputs_to_scan_frame(tmp_path):
    from meshsolid import synthetic as S
    from meshsolid.occ_util import read_step, shape_to_trimesh
    from meshsolid.pipeline import Params, run
    m = S.to_scan(S.box_with_hole()[0], edge=1.0, noise=0.01)
    T = trimesh.transformations.rotation_matrix(0.7, [0.3, 1, 0.2]); T[:3, 3] = [40, -10, 25]
    m.apply_transform(T)
    p = tmp_path / "b.stl"; m.export(p)
    rep, *_ = run(str(p), str(tmp_path / "o.step"), str(tmp_path / "r.json"), Params(normalise="apply"),
                  deviation_ply=str(tmp_path / "d.ply"))
    nz = rep["normalise"]
    assert rep["status"] == "ok" and nz["applied"] and set(nz["restored"]) == {"step", "deviation_ply"}
    N = np.asarray(nz["transform_scan_to_normalised"])
    assert N.shape == (4, 4) and np.allclose(np.asarray(nz["transform_normalised_to_scan"]) @ N, np.eye(4), atol=1e-9)
    solid = shape_to_trimesh(read_step(tmp_path / "o.step"), lin=0.01)
    _, d, _ = trimesh.proximity.closest_point(solid, m.vertices[::20])
    assert np.sqrt(np.mean(d ** 2)) < 0.05                                          # STEP sits on the scan
    dply = trimesh.load(tmp_path / "d.ply", process=False)
    assert np.allclose(dply.bounds, m.bounds, atol=0.05)


def test_report_mode_changes_nothing(tmp_path):
    from meshsolid import synthetic as S
    from meshsolid.pipeline import Params, run
    m = S.to_scan(S.box_with_hole()[0], edge=1.0, noise=0.01)
    p = tmp_path / "b.stl"; m.export(p)
    rep, *_ = run(str(p), str(tmp_path / "o.step"), str(tmp_path / "r.json"), Params())
    assert rep["normalise"]["mode"] == "report" and not rep["normalise"]["applied"]
    assert len(rep["normalise"]["transform_scan_to_normalised"]) == 4
