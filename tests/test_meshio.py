import numpy as np
import trimesh

from meshsolid.meshio import repair_mesh


def test_merge_and_fill_holes():
    m = trimesh.creation.box((10, 10, 10))
    m = m.subdivide().subdivide()
    soup_v = m.vertices[m.faces].reshape(-1, 3)        # STL-like triangle soup
    soup_f = np.arange(len(soup_v)).reshape(-1, 3)
    keep = np.ones(len(soup_f), bool)
    keep[[0, 40]] = False                              # punch two small holes
    deg = np.array([[0, 0, 1]])                        # plus one degenerate face
    f = np.vstack([soup_f[keep], deg])
    out, rep = repair_mesh(soup_v, f)
    assert rep.merged_vertices > 0
    assert rep.removed_degenerate == 1
    assert rep.holes_found == 2 and rep.holes_filled == 2
    assert rep.watertight and out.is_watertight
    assert abs(out.volume - 1000) < 1e-6


def test_non_manifold_reported():
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1]], float)
    f = np.array([[0, 1, 2], [1, 0, 3], [0, 1, 4]])   # edge 0-1 shared by 3 faces
    _, rep = repair_mesh(v, f)
    assert rep.non_manifold_edges == 1
    assert not rep.watertight
