"""M8b device layer: CPU-equivalence of the GPU closest-point path (run on Warp's CPU device on machines
without a GPU) and device resolution."""
import numpy as np
import pytest
import trimesh

from meshsolid import device as D

wp = pytest.importorskip("warp")


def _sliver_mesh():
    # a fan cap (long slivers meeting at one vertex) on a cylinder: the worst case for float32 distances
    m = trimesh.creation.cylinder(radius=30.0, height=80.0, sections=720)
    m.apply_translation([100.0, -50.0, 20.0])
    return m


def test_closest_points_match_libigl():
    igl = pytest.importorskip("igl")
    m = _sliver_mesh()
    rng = np.random.default_rng(1)
    P = trimesh.sample.sample_surface(m, 20000, seed=2)[0] + rng.normal(0, 0.05, (20000, 3))
    P = np.vstack([P, m.vertices[:500] + rng.normal(0, 0.03, (500, 3))])
    d2, tri, cp = igl.point_mesh_squared_distance(P, np.asarray(m.vertices, float), np.asarray(m.faces, np.int64))
    cpw, dw, fw = D.closest_points(m.vertices, m.faces, P, "warp-cpu")
    assert np.abs(np.sqrt(d2) - dw).max() < 1e-9
    assert np.linalg.norm(cp - cpw, axis=1).max() < 1e-6 or np.abs(np.sqrt(d2) - dw).max() < 1e-9


def test_deviation_same_on_warp_cpu(monkeypatch):
    from meshsolid import deviation as DV
    m = _sliver_mesh()
    P = trimesh.sample.sample_surface(m, 5000, seed=3)[0] * 1.0005
    D.set_device("cpu"); a = DV._on_surface(m, P)
    D.set_device("warp-cpu"); b = DV._on_surface(m, P)
    D.set_device("cpu")
    assert np.abs(a[1] - b[1]).max() < 1e-9


def test_device_resolution():
    ok, desc = D.cuda_info()
    assert D.set_device("cpu") == "cpu"
    if not ok:
        assert D.set_device("auto") == "cpu"
        with pytest.raises(RuntimeError, match="no usable GPU"):
            D.set_device("cuda")
    else:
        assert D.set_device("auto") == "cuda"
    with pytest.raises(ValueError):
        D.set_device("tpu")
    D.set_device("cpu")
    assert D.report()["used"] == "cpu"
