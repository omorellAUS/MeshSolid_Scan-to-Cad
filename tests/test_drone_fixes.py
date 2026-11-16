"""Fixes from the drone/VTOL test set: faceted-CAD STL mode, thin print walls, time budget / watchdog,
crash guards, gear plausibility.  Synthetic stand-ins (the user's STLs are not in git); the user-file
checks run only where those files exist."""
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from meshsolid import facets as FA  # noqa: E402
from meshsolid import noise as NZ  # noqa: E402
from meshsolid import synthetic_faceted as SF  # noqa: E402
from meshsolid.pipeline import Params, run  # noqa: E402

USER_STL = ROOT / "tests" / "user_scans" / "drone" / "stl"


def _naca_volume(chord, t, span, n=200001):
    x = np.linspace(0, chord, n); xx = x / chord
    yt = 5 * t * chord * (0.2969 * np.sqrt(xx) - 0.1260 * xx - 0.3516 * xx ** 2 + 0.2843 * xx ** 3 - 0.1015 * xx ** 4)
    return 2 * np.trapezoid(yt, x) * span


@pytest.fixture(scope="module")
def wing(tmp_path_factory):
    d = tmp_path_factory.mktemp("wing")
    sh, tr = SF.wing_section()
    m = SF.faceted_mesh(sh)
    m.export(d / "wing.stl")
    return d, m, tr


def test_faceted_cad_detected_and_scan_not(wing):
    _, m, _ = wing
    tol, sig, ex = NZ.auto_tol(m)
    f, st = FA.detect(m, sig, ex)
    assert f and st["cad_tessellation"] and st["sag_p90"] > 0.01
    from meshsolid.synthetic_screw import scanify
    s, _ = scanify(m.subdivide_to_size(1.0), edge=0.8, noise=0.02, seed=1)
    tol, sig, ex = NZ.auto_tol(s)
    f2, st2 = FA.detect(s, sig, ex)
    assert not st2["cad_tessellation"] and not f2


def test_faceted_wing_builds_with_freeform_skin(wing):
    d, m, tr = wing
    rep, *_ = run(str(d / "wing.stl"), str(d / "wing.step"), str(d / "wing.json"), Params(time_budget_s=600))
    assert rep["status"] == "ok", rep.get("failure")
    assert rep["tolerance"]["from"].startswith("facet chord sag")
    b = rep["stages"]["build"]
    # section loft (faceted-CAD early path) or freeform skin both valid
    if b.get("method") == "section_loft":
        assert (rep.get("stages") or {}).get("loft_skin", {}).get("accepted")
    else:
        assert rep["stages"]["freeform"]["accepted"] >= 1
    assert b["n_solids"] == 1 and b.get("brepcheck_valid", b.get("ok"))
    v = _naca_volume(tr["chord"], tr["thickness"], tr["span"])
    assert abs(b["volume"] - v) / v < 0.01
    # no huge quadric left standing for a skin strip (freeform path only)
    if b.get("method") != "section_loft":
        diag = float(np.linalg.norm(m.extents))
        for g in rep.get("regions") or []:
            if g["kind"] == "cylinder":
                assert g["params"]["diameter"] < 4 * diag


def test_thin_wall_shell_builds(tmp_path):
    sh, tr = SF.thin_shell()
    m = SF.faceted_mesh(sh)
    m.export(tmp_path / "shell.stl")
    rep, *_ = run(str(tmp_path / "shell.stl"), str(tmp_path / "shell.step"), None, Params(time_budget_s=300))
    assert rep["status"] == "ok", rep.get("failure")
    v = tr["L"] * tr["W"] * tr["H"] - (tr["L"] - 2 * tr["wall"]) * (tr["W"] - 2 * tr["wall"]) * (tr["H"] - tr["wall"])
    assert abs(rep["stages"]["build"]["volume"] - v) / v < 1e-3


def test_watchdog_marks_timeout_with_stage(tmp_path):
    from meshsolid import synthetic as S
    sh, _ = S.box_with_hole()
    S.to_scan(sh, edge=0.4, noise=0.02, seed=0).export(tmp_path / "b.stl")
    rp = tmp_path / "b.json"
    p = subprocess.run([sys.executable, "-m", "meshsolid", str(tmp_path / "b.stl"), "-o", str(tmp_path / "b.step"),
                        "--report", str(rp), "--time-budget", "4"], cwd=ROOT, capture_output=True, text=True, timeout=300)
    rep = json.loads(rp.read_text())
    assert rep["status"] == "timeout" and rep["timeout"]["by"] == "watchdog" and rep["timeout"]["stage"]
    assert "exceeded during stage" in rep["failure"] and p.returncode != 0
    assert not (tmp_path / "b.step").exists()


def test_watchdog_leaves_finished_report_alone(tmp_path):
    from meshsolid import watchdog as W
    rp = tmp_path / "r.json"
    rp.write_text(json.dumps({"status": "ok", "last_stage": "done"}))
    assert not W.mark_timeout(rp, 10)
    assert json.loads(rp.read_text())["status"] == "ok"
    rp.write_text(json.dumps({"status": "incomplete", "last_stage": "build"}))
    assert W.mark_timeout(rp, 10)
    r = json.loads(rp.read_text())
    assert r["status"] == "timeout" and r["timeout"]["stage"] == "build"


def test_stage_budget_follows_run_deadline():
    import time
    from meshsolid import budget as B
    B.set_deadline(None)
    assert B.stage_budget(300) == 300
    B.set_deadline(time.time() + 200)
    assert 100 < B.stage_budget(300) <= 170
    B.set_deadline(time.time() + 20)
    assert B.stage_budget(300) == 5.0
    B.set_deadline(None)


def test_freeform_skips_cluster_without_samples(monkeypatch):
    from meshsolid import freeform as FF
    from meshsolid.segment import Region
    m = trimesh.creation.box((10, 10, 10))
    labels = np.zeros(len(m.faces), int)
    regs = [Region(id=0, faces=np.arange(len(m.faces)), primitive=None, area=float(m.area))]
    monkeypatch.setattr(FF, "smooth_clusters", lambda *a, **k: [[7, 8]])   # ids with no faces
    out_regs, out_labels, log = FF.apply_freeform(m, None, regs, labels, 0.05, lambda g, f: (np.zeros((0, 3)), np.zeros((0, 3))))
    assert log and log[0]["accepted"] is False and "samples" in log[0]["reason"]


@pytest.mark.skipif(not (USER_STL / "01M6F7R2J52QMQ7KQT1PN73P7E.stl").exists(), reason="user drone STL not present")
def test_aileron_is_not_a_gear():
    from meshsolid import gear
    from meshsolid.meshio import load_mesh
    m, _ = load_mesh(str(USER_STL / "01M6F7R2J52QMQ7KQT1PN73P7E.stl"))
    g, info = gear.analyse(m, 0.01)
    assert g is None, info


def _box_with_fan_on_xmax(x0):
    """Unit box at x0..x0+1 as raw triangles; its +x face is a 4-triangle fan about the face centre."""
    b = trimesh.creation.box((1, 1, 1))
    b.apply_translation((x0 + 0.5, 0.5, 0.5))
    V, F = b.vertices.copy(), b.faces.copy()
    xmax = np.isclose(b.triangles_center[:, 0], x0 + 1)
    F = F[~xmax]
    c = len(V); V = np.vstack([V, [[x0 + 1, 0.5, 0.5]]])
    q = [i for i in range(8) if np.isclose(V[i, 0], x0 + 1)]
    ang = np.arctan2(V[q, 2] - 0.5, V[q, 1] - 0.5); q = [q[i] for i in np.argsort(ang)]
    fan = [[q[i], q[(i + 1) % 4], c] for i in range(4)]
    return V, np.vstack([F, fan])


def test_glued_halves_with_differently_triangulated_caps_become_one_body():
    from meshsolid.meshio import repair_mesh
    VA, FA_ = _box_with_fan_on_xmax(0.0)                       # cap at x=1 as a fan
    B = trimesh.creation.box((1, 1, 1)); B.apply_translation((1.5, 0.5, 0.5))   # cap at x=1 as 2 triangles
    V = np.vstack([VA, B.vertices]); F = np.vstack([FA_, B.faces + len(VA)])
    m, rep = repair_mesh(V, F)
    assert rep.removed_internal_walls == 6 and m.is_watertight
    assert len(m.split(only_watertight=False)) == 1 and abs(m.volume - 2.0) < 1e-9


def test_glued_halves_with_identical_caps_become_one_body():
    from meshsolid.meshio import repair_mesh
    A = trimesh.creation.box((1, 1, 1)); A.apply_translation((0.5, 0.5, 0.5))
    B = A.copy(); B.apply_transform(trimesh.transformations.reflection_matrix((1, 0, 0), (1, 0, 0)))
    V = np.vstack([A.vertices, B.vertices]); F = np.vstack([A.faces, B.faces + len(A.vertices)])
    m, rep = repair_mesh(V, F)
    assert rep.removed_internal_walls == 4 and m.is_watertight and abs(m.volume - 2.0) < 1e-9


def test_setback_patch_hang_is_cut_off(monkeypatch):
    """A filling patch whose build hangs is killed after the per-patch limit and counted, not waited on."""
    import time as _t
    from meshsolid import build_local as BL
    from meshsolid.occ_child import _use_fork
    if not _use_fork():
        import pytest
        pytest.skip("fork-based patch limit is POSIX only")

    class Hang:
        def Build(self):
            _t.sleep(60)

        def IsDone(self):
            return False

    monkeypatch.setattr(BL, "SETBACK_PATCH_TIMEOUT", 1.0)
    res = BL.LocalBuildResult()
    t0 = _t.time()
    assert BL._filling_face(Hang(), None, res) is None
    assert _t.time() - t0 < 10 and res.setback_patch_timeouts == 1


def test_segmentation_stops_at_run_budget():
    """Region growing honours the run deadline: it stops, reports it, and leaves the rest unassigned."""
    import time as _t
    import trimesh as _tm
    from meshsolid import budget
    from meshsolid.segment import segment, SegParams
    m = _tm.creation.icosphere(subdivisions=4, radius=10)
    m.vertices += np.random.default_rng(0).normal(0, 0.05, m.vertices.shape)
    budget.set_deadline(_t.time() - 1)          # already out of time
    try:
        regions, labels, info = segment(m, SegParams(tol=0.01))
    finally:
        budget.set_deadline(None)
    assert info["budget_hit"] and (labels < 0).all()
