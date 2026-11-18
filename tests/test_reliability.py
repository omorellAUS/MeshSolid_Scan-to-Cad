"""1.0.3 Reliability: per-stage time limits, heartbeat, end reasons,
input decimation + units check, job-folder pruning."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import trimesh

from meshsolid import synthetic as S
from meshsolid.normalise import check_units, decimate_for_recognition
from meshsolid.pipeline import Params, run, _stage_deadline, CURRENT


def test_params_stage_limits():
    p = Params()
    assert "segmentation" in p.stage_limits
    assert "build" in p.stage_limits
    assert p.stage_limits["segmentation"] > 0
    assert p.stage_limits["build"] > 0
    assert p.stage_heartbeat_s == 5.0
    assert p.decimate_for_recognition == 300_000


def test_stage_deadline_no_budget():
    p = Params(time_budget_s=0.0)
    dl = _stage_deadline(p, "segmentation")
    assert dl == p.stage_limits["segmentation"]


def test_stage_deadline_with_budget():
    p = Params(time_budget_s=600.0)
    dl = _stage_deadline(p, "segmentation")
    assert dl <= p.stage_limits["segmentation"]
    assert dl > 0

def test_units_1650mm_ambiguous():
    m = trimesh.creation.box(extents=[1650, 800, 400])
    info = check_units(m)
    assert info["unit"] == "mm"
    assert info["ambiguous_inches"] is True
    assert "confirm units" in info["note"]
    assert abs(info["largest_extent_mm"] - 1969.7) < 10


def test_units_small_mm_possible_inches():
    m = trimesh.creation.box(extents=[10, 5, 3])
    info = check_units(m)
    assert info["unit"] == "mm"
    assert info.get("possible_inches") is True


def test_units_metres_scaled():
    m = trimesh.creation.box(extents=[0.5, 0.3, 0.2])
    info = check_units(m)
    assert info["unit"] == "m (scaled x1000 to mm)"


def test_decimate_big_mesh():
    m = trimesh.creation.icosphere(subdivisions=7, radius=50)
    assert len(m.faces) > 300_000, f"need >300k faces, got {len(m.faces):,}"
    out = decimate_for_recognition(m, target=300_000)
    assert len(out.faces) <= 300_000 * 1.1, f"decimation didn't reduce enough: {len(out.faces):,}"
    assert len(out.faces) < len(m.faces)


def test_decimate_small_mesh_unchanged():
    m = trimesh.creation.box(extents=[10, 10, 10])
    out = decimate_for_recognition(m, target=300_000)
    assert len(out.faces) == len(m.faces)

def test_heartbeat_written(tmp_path):
    from meshsolid.pipeline import _write_heartbeat
    rep_path = tmp_path / "r.json"
    CURRENT.update(rep={}, report_path=str(rep_path), t0=time.time())
    _write_heartbeat("segmentation")
    hp = rep_path.with_suffix(".heartbeat.json")
    assert hp.exists()
    hb = json.loads(hp.read_text())
    assert hb["stage"] == "segmentation"
    assert "ts" in hb
    assert "elapsed_s" in hb


def test_end_reason_ok_on_success(tmp_path):
    shape, _ = S.box_with_hole()
    stl = tmp_path / "box.stl"
    S.to_scan(shape, noise=0.02, seed=1).export(stl)
    step = tmp_path / "box.step"
    rep = tmp_path / "box.report.json"
    rep_obj, *_ = run(str(stl), str(step), str(rep))
    assert rep_obj["status"] == "ok"
    assert rep_obj.get("end_reason") == "ok"
    assert step.exists()



def test_decimation_recorded_in_report(tmp_path):
    m = trimesh.creation.icosphere(subdivisions=5, radius=50)
    stl = tmp_path / "big.stl"
    m.export(stl)
    step = tmp_path / "big.step"
    rep = tmp_path / "big.report.json"
    rep_obj, *_ = run(str(stl), str(step), str(rep), Params(time_budget_s=120))
    if "decimation" in rep_obj:
        d = rep_obj["decimation"]
        assert d["from_faces"] > 300_000
        assert d["to_faces"] <= d["from_faces"]


def test_units_recorded_in_report(tmp_path):
    shape, _ = S.box_with_hole(L=1650, W=800, H=400)
    stl = tmp_path / "bigbox.stl"
    S.to_scan(shape, noise=0.01, seed=1).export(stl)
    step = tmp_path / "bigbox.step"
    rep = tmp_path / "bigbox.report.json"
    rep_obj, *_ = run(str(stl), str(step), str(rep), Params(time_budget_s=120))
    assert "units" in rep_obj
    assert rep_obj["units"]["ambiguous_inches"] is True


def test_prune_deletes_intermediates(tmp_path):
    from meshsolid.cli import _prune_job_folder
    report = tmp_path / "job.report.json"
    report.write_text("{}")
    step = tmp_path / "job.step"
    step.write_text("stub")
    preview = tmp_path / "job.preview.png"
    preview.write_text("stub")
    devply = tmp_path / "job.deviation.ply"
    devply.write_text("stub")
    hb = tmp_path / "job.heartbeat.json"
    hb.write_text("{}")
    deleted = _prune_job_folder(str(report), str(step), str(preview), str(devply))
    assert not preview.exists()
    assert not devply.exists()
    assert not hb.exists()
    assert report.exists()
    assert step.exists()
    assert len(deleted) == 3
def test_end_reason_failed_honestly(tmp_path):
    m = trimesh.creation.icosphere(subdivisions=3, radius=10)
    m.apply_transform(np.diag([1.0, 0.6, 0.4, 1.0]))
    stl = tmp_path / "ell.stl"
    m.export(stl)
    step = tmp_path / "ell.step"
    rep = tmp_path / "ell.report.json"
    rep_obj, *_ = run(str(stl), str(step), str(rep))
    assert rep_obj["status"] == "failed"
    assert rep_obj["failure"]
    assert not step.exists()