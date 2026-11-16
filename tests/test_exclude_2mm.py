"""2 mm auto-exclusion: connected scan regions more than 2 mm from the solid, any part."""
from __future__ import annotations

import numpy as np
import pytest
import trimesh

from meshsolid.exclude_2mm import detect, public_report, outlines, THRESHOLD_MM


def _plane(n=80, span=20.0):
    xs = np.linspace(-span / 2, span / 2, n)
    xx, yy = np.meshgrid(xs, xs)
    V = np.c_[xx.ravel(), yy.ravel(), np.zeros(n * n)]
    # regular grid faces
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            a = i * n + j
            faces.append([a, a + 1, a + n + 1])
            faces.append([a, a + n + 1, a + n])
    return trimesh.Trimesh(V, np.asarray(faces), process=False)


def test_detect_finds_connected_5mm_region_with_area_and_pct():
    mesh = _plane()
    dv = np.zeros(len(mesh.vertices))
    # a compact 6 x 6 mm island raised 5 mm at the origin
    m = (np.abs(mesh.vertices[:, 0]) <= 3.0) & (np.abs(mesh.vertices[:, 1]) <= 3.0)
    dv[m] = 5.0
    # a single 6 mm spike far away must not count (too few points once isolated? add only 4 points)
    dv[0] = 6.0
    rep, mask = detect(mesh.vertices, dv, mesh=mesh)
    assert rep["n_regions"] == 1, rep
    r = rep["regions"][0]
    assert r["area_mm2"] >= 20.0 and r["pct_of_scan"] > 0
    assert r["depth_max_mm"] >= 4.9 and r["n_points"] == int(m.sum())
    assert "2 mm" in r["reason"] and r["boundary"]["type"] == "box"
    assert mask.sum() == int(m.sum())
    assert outlines(rep) and len(outlines(rep)[0]) >= 3
    pub = public_report(rep)
    assert "_ix" not in pub["regions"][0]


def test_over_5pct_flag_when_the_region_is_large():
    mesh = _plane()
    dv = np.zeros(len(mesh.vertices))
    # most of the plane (well over 5 % of points)
    m = mesh.vertices[:, 0] > 0
    dv[m] = 3.0
    rep, mask = detect(mesh.vertices, dv, mesh=mesh)
    assert rep["over_5pct"] and rep["pct_of_scan"] > 5.0
    assert rep["over_5pct_flag"] and "5" in rep["over_5pct_flag"]
    assert mask.mean() == pytest.approx(m.mean(), abs=0.02)


def test_metrology_auto_excludes_2mm_bump_and_lists_it(tmp_path):
    from meshsolid import occ_util as U
    from meshsolid.metrology import run
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.gp import gp_Pnt

    shape = BRepPrimAPI_MakeBox(gp_Pnt(-10, -10, -10), 20, 20, 20).Shape()
    U.write_step(shape, str(tmp_path / "part.step"))
    m = U.shape_to_trimesh(shape, lin=0.3, ang=0.2)
    v, f = trimesh.remesh.subdivide_to_size(m.vertices, m.faces, max_edge=1.0, max_iter=8)
    scan = trimesh.Trimesh(v, f, process=True)
    # almost the whole top face, +5 mm — well over 5 % of the vertices
    top = scan.vertices[:, 2] > 9.6
    scan.vertices[top, 2] += 5.0
    scan.export(tmp_path / "scan.stl")

    off = run(tmp_path / "scan.stl", tmp_path / "part.step", tmp_path / "off",
              device="cpu", n_probes=3, auto_exclude_2mm=False, pdf=False, max_dist=None)
    assert off["stats"]["max_abs"] > 4.0
    assert not (off.get("auto_exclusions_2mm") or {}).get("n_regions")

    on = run(tmp_path / "scan.stl", tmp_path / "part.step", tmp_path / "on",
             device="cpu", n_probes=3, pdf=False, max_dist=None)
    ex = on["auto_exclusions_2mm"]
    assert ex["n_regions"] >= 1 and ex["n_points"] > 0
    ni = [e for e in on["not_inspected"] if "auto-exclusion" in e.get("reason", "")]
    assert ni and ni[0]["area_mm2"] > 0 and ni[0]["pct_of_scan"] > 0
    assert "2 mm" in ni[0]["detail"] and ni[0]["n"] == ni[0]["n_new"]
    # the bump is out of the inspected set, so the remaining deviation is tessellation-scale
    assert on["stats"]["max_abs"] < 1.0
    assert on["stats"]["excluded_over_5pct"] or ex["over_5pct"] or ex.get("over_5pct_total")
    assert ex["over_5pct"] or ex.get("over_5pct_total")
    assert on["stats"]["pct_excluded"] > 5.0
