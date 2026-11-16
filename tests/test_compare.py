"""M11 compare tool and batch folder mode, against known offsets and poses."""
import json

import numpy as np
import pytest
import trimesh

from meshsolid import compare as CP
from meshsolid import occ_util as U
from meshsolid import synthetic as S


@pytest.fixture(scope="module")
def box(tmp_path_factory):
    d = tmp_path_factory.mktemp("cmp")
    sh, _ = S.box_with_hole()
    U.write_step(sh, str(d / "truth.step"))
    m = U.shape_to_trimesh(sh, lin=0.01, ang=0.1)
    m = trimesh.Trimesh(*trimesh.remesh.subdivide_to_size(m.vertices, m.faces, 0.5))
    return d, m


def _pose(m, seed):
    T = trimesh.transformations.random_rotation_matrix(np.random.default_rng(seed).random(3)); T[:3, 3] = [10, -5, 3]
    m = m.copy(); m.apply_transform(T); return m, T


def test_pose_recovered(box):
    d, m = box
    mp, T = _pose(m, 1)
    mp.export(d / "posed.stl")
    r = CP.compare(str(d / "posed.stl"), str(d / "truth.step"))
    assert r["test_to_ref"]["rms"] < 0.005 and r["hausdorff"] < 0.05
    # box_with_hole is symmetric, so the recovered pose may differ from T^-1 by a symmetry: transform @ T
    # must be a proper rigid motion that maps the part onto itself
    M = np.asarray(r["alignment"]["transform"]) @ T
    assert abs(np.linalg.det(M[:3, :3]) - 1) < 1e-3 and np.allclose(M[:3, :3] @ M[:3, :3].T, np.eye(3), atol=1e-3)
    v = m.vertices[::97]
    _, dist, _ = trimesh.proximity.closest_point(m, v @ M[:3, :3].T + M[:3, 3])
    assert dist.max() < 0.02
    assert set(r["zones"]) >= {"planar", "cylindrical", "edges"}
    assert abs(r["volume"]["rel_diff"]) < 1e-3


def test_known_offset_and_sign(box):
    d, m = box
    # every vertex moved 0.1 mm along its face's normal (faces of a subdivided box are flat: exact offset)
    off = trimesh.Trimesh(m.vertices + 0.1 * m.vertex_normals, m.faces)
    off.export(d / "offset.stl")
    r = CP.compare(str(d / "offset.stl"), str(d / "truth.step"), align=False)
    z = r["zones"]["away_from_edges"]
    assert abs(z["rms"] - 0.1) < 0.005 and abs(z["mean_signed"] - 0.1) < 0.005      # outside = positive
    assert r["bands"]["<= 0.05 mm"] < 0.1 and r["bands"]["<= 0.2 mm"] > 0.99
    assert r["volume"]["rel_diff"] > 0


def test_cli(box, tmp_path):
    d, m = box
    m.export(tmp_path / "a.stl")
    assert CP.main([str(tmp_path / "a.stl"), str(d / "truth.step"), "--report", str(tmp_path / "c.json"), "--no-align"]) == 0
    assert json.loads((tmp_path / "c.json").read_text())["test_to_ref"]["rms"] < 0.005
    assert CP.main([str(tmp_path / "missing.stl"), str(d / "truth.step")]) == 2


def test_batch_folder(tmp_path):
    from meshsolid import batch as B
    from meshsolid.synthetic import to_scan
    inp = tmp_path / "in"; inp.mkdir()
    sh, _ = S.box_with_hole()
    to_scan(sh, edge=1.0, noise=0.02, seed=0).export(inp / "Box.stl")
    U.write_step(sh, str(inp / "box.STEP"))                          # stem pairs case-insensitively
    sh2, _ = S.stepped_shaft()
    to_scan(sh2, edge=1.0, noise=0.02, seed=1).export(inp / "shaft.stl")
    (inp / "notes.txt").write_text("ignored")
    assert [(a.name, b.name if b else None) for a, b in B.find_pairs(inp)] == [("Box.stl", "box.STEP"), ("shaft.stl", None)]
    rows = B.run_batch(inp, tmp_path / "out", ["--tol", "0.05"], timeout=900, progress=lambda s: None)
    by = {r["name"]: r for r in rows}
    assert by["Box"]["status"] == "ok" and by["shaft"]["status"] == "ok"
    assert by["Box"]["solid_vs_truth_rms"] < 0.02 and "planar" in by["Box"]["zones"]
    assert by["Box"]["scan_vs_truth_rms"] < 0.04 and "solid_vs_truth_rms" not in by["shaft"]
    s = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert s["meshes"] == 2 and s["ok"] == 2 and s["scored"] == 1
    assert (tmp_path / "out" / "summary.csv").read_text().count("\n") == 3
    assert not any(p.suffix == ".json" for p in inp.iterdir())       # nothing written next to the inputs
