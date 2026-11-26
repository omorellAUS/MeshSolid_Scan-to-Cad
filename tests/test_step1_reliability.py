"""MeshSolid 2.0.0 Step 1: reliability, traceability, and the seven known bugs."""
import json
import time
from pathlib import Path

import numpy as np
import pytest
import trimesh

from meshsolid.acceptance import annotate_acceptance, describe_gate
from meshsolid.cli import format_build_line, prune_job_folder
from meshsolid.impeller_run import impeller_build_stage
from meshsolid.metrology_pdf import format_shift_mm, not_inspected_pct, pose_table_rows
from meshsolid.names import safe_part_id
from meshsolid.normalise import check_units, recognition_meshes
from meshsolid.pipeline import END_REASONS, Params, _stage_deadline


def test_end_reasons_are_the_five_names():
    assert set(END_REASONS) == {"ok", "crashed", "cancelled", "timed_out", "rejected_by_gate"}


def test_stage_limits_exist_and_shrink_with_a_budget():
    p = Params()
    assert p.stage_limits["segmentation"] == 300.0
    assert p.stage_limits["build"] == 600.0
    assert _stage_deadline(p, "segmentation") == 300.0
    tight = Params(time_budget_s=0.0)
    assert tight.decimate_for_recognition == 300_000


def test_units_check_does_not_raise_and_flags_x25_4_and_x1000():
    base = trimesh.creation.box(extents=[40, 30, 20])
    plain = check_units(base)
    assert plain["mis_scaled"] is False
    inch = base.copy()
    inch.apply_scale(25.4)
    info = check_units(inch)
    assert info["mis_scaled"] is True and info["scale_factor"] == 25.4
    assert "25.4" in info["note"]
    metres = base.copy()
    metres.apply_scale(1000)
    info_m = check_units(metres)
    assert info_m["mis_scaled"] is True and info_m["scale_factor"] == 1000
    assert "1000" in info_m["note"]


def test_recognition_reduction_keeps_full_mesh_for_deviation():
    full = trimesh.creation.icosphere(subdivisions=2)
    original_faces = len(full.faces)

    def evil(mesh, target):
        # Edits the object it was given. The deviation mesh must already be a copy.
        mesh.update_faces(np.zeros(len(mesh.faces), dtype=bool))
        return trimesh.creation.box(extents=[1, 1, 1])

    rec, dev, info = recognition_meshes(full, target=10, decimate=evil)
    assert info["full_resolution_kept_for_deviation"] is True
    assert info["deviation_faces"] == original_faces
    assert len(dev.faces) == original_faces
    assert len(rec.faces) < original_faces
    assert dev is not rec
    # under the cap, recognition and deviation are the same mesh (1.0.2a path)
    same, dev2, info2 = recognition_meshes(trimesh.creation.box(), target=300_000)
    assert info2 is None and same is dev2


def test_heartbeat_updates_while_the_main_thread_holds_the_gil(tmp_path):
    from meshsolid.heartbeat import heartbeat_path, set_stage, start, stop
    report = tmp_path / "job.report.json"
    report.write_text("{}")
    proc = start(str(report), interval=0.05, t0=time.time())
    try:
        set_stage(str(report), "segmentation")
        deadline = time.time() + 2.0
        first = None
        while time.time() < deadline:
            hp = Path(heartbeat_path(str(report)))
            if hp.is_file():
                first = json.loads(hp.read_text())
                if first.get("stage") == "segmentation":
                    break
            time.sleep(0.02)
        assert first and first["stage"] == "segmentation"
        # Pure Python holds the interpreter. A thread heartbeat cannot tick here.
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < 0.45:
            pass
        second = json.loads(Path(heartbeat_path(str(report))).read_text())
        assert second["stage"] == "segmentation"
        assert second["ts"] > first["ts"] + 0.05
    finally:
        stop(proc)


def test_prune_keeps_a_report_that_is_not_named_report(tmp_path):
    report = tmp_path / "o.json"
    report.write_text("{}")
    (tmp_path / "o.step").write_text("ISO")
    (tmp_path / "mesh.stl").write_text("x")
    prune_job_folder(str(report))
    assert report.is_file() and (tmp_path / "o.step").is_file()
    assert not (tmp_path / "mesh.stl").exists()


def test_prune_job_folder_keeps_step_reports_and_logs(tmp_path):
    report = tmp_path / "mesh_body.report.json"
    report.write_text("{}")
    step = tmp_path / "mesh_body.step"
    step.write_text("ISO")
    log = tmp_path / "mesh_body.engine.log"
    log.write_text("log")
    for name in ("mesh_body.stl", "mesh_body.preview.png", "mesh_body.deviation.ply",
                 "mesh_body.report.heartbeat.json", "mesh_body.report.stage"):
        (tmp_path / name).write_text("x")
    deleted = prune_job_folder(str(report), str(tmp_path / "mesh_body.preview.png"),
                               str(tmp_path / "mesh_body.deviation.ply"))
    assert report.is_file() and step.is_file() and log.is_file()
    # The add-in review opens the preview the moment the CLI returns, so it is kept.
    assert (tmp_path / "mesh_body.preview.png").is_file()
    assert not (tmp_path / "mesh_body.stl").exists()
    assert not (tmp_path / "mesh_body.deviation.ply").exists()
    assert not (tmp_path / "mesh_body.report.heartbeat.json").exists()
    assert deleted


def test_gate_text_names_blade_scatter_rule_and_class_ratio_without_flipping_accept():
    text = describe_gate(True, 0.0504, 0.009, "1.5 x the blade-to-blade scatter of the folded side fits")
    assert "1.5x blade-scatter" in text
    assert "5.6x over 0.009 mm" in text
    acc = {"passed": True}
    annotate_acceptance(acc, accepted=True, rms=0.0504, p95=0.10, class_rms=0.009, class_p95=0.018,
                        rule="1.5 x the blade-to-blade scatter of the folded side fits")
    assert acc["passed"] is True  # STEP is still written
    assert acc["class_limit_passed"] is False
    assert "5.6x over 0.009 mm" in acc["summary"]


def test_impeller_build_stage_has_solid_count_and_the_log_line_uses_it():
    stage = impeller_build_stage({"valid": True, "n_solids": 1, "n_faces": 42, "volume": 10.0, "seconds": 1.2})
    assert stage["n_solids"] == 1 and stage["brepcheck_valid"] is True
    rep = {"regions": [], "mesh": {"volume": 9.0},
           "stages": {"build": {"method": "impeller", "ok": True, "volume": 10.0}},
           "impeller": {"build": {"valid": True, "n_solids": 1, "n_faces": 42}}}
    line = format_build_line(rep)
    assert "solids=1" in line and "valid=True" in line
    assert "solids=None" not in line


def test_pdf_shift_is_millimetres_on_its_own_short_row():
    rows = pose_table_rows({
        "rotation_about_axis_deg": -11.999,
        "rotation_bound_deg": 12.0,
        "tilt_deg": 0.4821,
        "translation_mm": 0.0124,
        "warnings": ["alignment moved the part"],
    })
    shift = [v for lab, v in rows if lab.startswith("alignment shift")]
    assert shift == ["0.0124 mm"]
    assert all(len(v) <= 56 for _, v in rows)
    joined = " ".join(v for _, v in rows)
    assert "0.0124 m" not in joined.replace("0.0124 mm", "")
    # a translation vector is still millimetres, not metres
    assert format_shift_mm([0.0124, 0.0, 0.0]) == "0.0124 mm"


def test_pdf_beyond_cap_percent_matches_json_not_the_recompute():
    stats = {"n": 2135, "n_excluded": 7989, "pct_beyond_cap": 79.89}
    entry = {"reason": "beyond outlier cap", "n": 7989, "n_new": 7989, "pct_of_scan": 79.89,
             "detail": "beyond-cap dropped from the statistics"}
    recomputed = 100.0 * 7989 / (2135 + 7989)
    assert round(recomputed, 2) == 78.91
    shown = not_inspected_pct(entry, stats)
    assert shown == pytest.approx(79.89)
    assert round(shown, 2) != 78.91


def test_safe_ampersand_id_matches_in_addin_and_batch(tmp_path):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fusion_addin" / "MeshSolid"))
    from mslib.names import safe_part_id as addin_id
    from meshsolid.batch import _name
    from meshsolid.names import safe_part_id as engine_id
    assert engine_id("KK&K Stage 1") == addin_id("KK&K Stage 1") == "KKandK_Stage_1"
    assert "&" not in engine_id("KK&K Stage 1")
    folder = tmp_path / "in"
    folder.mkdir()
    mesh = folder / "KK&K Stage 1.stl"
    trimesh.creation.box().export(mesh)
    assert _name(mesh, folder) == "KKandK_Stage_1"
    from mslib.engine_client import EngineJob
    job = EngineJob(str(mesh), str(tmp_path / "out"))
    names = [Path(p).name for p in job.paths().values()]
    assert names
    assert all("&" not in n for n in names)
    assert any(n.startswith("KKandK_Stage_1") for n in names)
    cmd = job.command(type("S", (), {"python": "python", "extra_args": [], "timeout_s": 60})())
    assert isinstance(cmd, list)
    step = cmd[cmd.index("-o") + 1]
    assert "&" not in Path(step).name


def test_watchdog_marks_timed_out(tmp_path):
    from meshsolid.watchdog import mark_timeout
    report = tmp_path / "job.report.json"
    report.write_text(json.dumps({"status": "incomplete", "last_stage": "segmentation"}))
    assert mark_timeout(str(report), 30) is True
    rep = json.loads(report.read_text())
    assert rep["end_reason"] == "timed_out"
    assert rep["status"] == "timeout"
