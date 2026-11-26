"""MeshSolid 2.0.1: shared exclusions, spec v3, source cross-check, valid_solid, fuse profile."""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from meshsolid import __version__
from meshsolid.exclusions import (
    apply_policy, engine_summary, hand_entered_volume_mm3, mask_on_cylinders,
    metrology_summary, publish, volume_check, volume_error_pct,
)
from meshsolid.fuse_profile import BENCH_VARIANTS, FuseProfile, bench_enabled
from meshsolid.impeller_run import impeller_build_stage
from meshsolid.mesh_source import apply_cross_check, lookup_parts_csv, reconcile
from meshsolid.metrology_spec import SPEC, SPEC_V2, SPEC_V3, SPECS, spec_hash
from meshsolid.remeasure import compare_metrics, flat_metrics
from meshsolid.solid_check import valid_solid_record
import scoreboard


def test_version_is_2_0_1():
    assert __version__ == "2.0.1"
    bat = (ROOT / "INSTALL_MeshSolid.bat").read_bytes()
    assert b"2.0.1" in bat and b"2.0.0" not in bat
    assert b"\r\n" in bat and bat.count(b"\n") == bat.count(b"\r\n")
    assert "2.0.1" in (ROOT / "package.xml").read_text()
    assert "2.0.1" in (ROOT / "freecad_addin/MeshSolid/package.xml").read_text()
    assert "2.0.1" in (ROOT / "fusion_addin/MeshSolid/MeshSolid.manifest").read_text()


def test_v1_and_v2_specs_are_unchanged_and_v3_is_tagged():
    assert spec_hash(SPEC) == "05d7048b9f9d"
    assert spec_hash(SPEC_V2) == "42ceb29d6b7b"
    assert SPECS["v3"] is SPEC_V3 and SPEC_V3["spec_version"] == "v3"
    assert SPEC_V3["exclusions"]["balance_cuts"]["counted_in_general_cap"] is False
    assert spec_hash(SPEC_V3) != spec_hash(SPEC_V2)


def test_engine_and_metrology_share_one_exclusion_tally():
    rep = {
        "mesh": {"vertices": 1000},
        "face_features": {"exclusions": [
            {"kind": "balance_cut", "exclude": True, "n_points": 80, "area_mm2": 40, "depth_mm": 0.4, "label": "cut"},
            {"kind": "drill_mark", "exclude": True, "n_points": 40, "area_mm2": 10, "depth_mm": 0.2, "label": "drill"},
        ]},
        # a second detector must not be added on top of the face-classifier balance cuts
        "balance_cuts": {"by_plane": {"back": {"cuts": [
            {"n_points": 999, "area_mm2": 9000, "depth_max_mm": 9.0, "angle_deg": 10}]}}},
    }
    eng = engine_summary(rep, 1000)
    assert eng["balance_cuts"]["count"] == 1
    assert eng["balance_cuts"]["n_points"] == 80
    assert eng["pct_excluded"] == 4.0
    assert "not counted in the 5 %" in eng["balance_cuts"]["line"]
    rep["exclusions"] = eng
    met = metrology_summary(rep, 1000)
    assert met["pct_excluded"] == eng["pct_excluded"]
    assert met["balance_cuts"]["count"] == eng["balance_cuts"]["count"]
    assert met["balance_cuts"]["area_mm2"] == eng["balance_cuts"]["area_mm2"]
    publish(rep, 1000)
    assert rep["spec_version"] == "v3"
    assert rep["balance_cuts"]["policy"]["count"] == 1


def test_balance_cuts_use_their_own_caps():
    items = [
        {"id": "a", "category": "balance_cut", "n": 10, "area_mm2": 60, "depth_mm": 0.3},
        {"id": "b", "category": "balance_cut", "n": 10, "area_mm2": 50, "depth_mm": 0.3},
        {"id": "deep", "category": "balance_cut", "n": 500, "area_mm2": 5, "depth_mm": 8.0},
        {"id": "gen", "category": "general", "n": 60, "area_mm2": 1, "depth_mm": 0.1},
    ]
    out = apply_policy(items, n_scan=1000, caps={"depth_mm": 5.0, "area_mm2": 100.0})
    # smallest area that fits is 50 mm²; 60 mm² would exceed the total, and the 8 mm cut is too deep
    assert out["accepted_ids"] == ["b"]
    assert out["balance_cuts"]["count"] == 1
    assert out["balance_cuts"]["flagged"] == 2
    assert out["balance_cuts"]["over_cap"] is True
    assert out["pct_excluded"] == 6.0
    assert out["n_excluded_general"] == 60
    assert out["over_general_cap"] is True


def test_hand_entered_bore_is_subtracted_from_volume_error():
    bore = hand_entered_volume_mm3([{"kind": "bore", "d_mm": 10.0}], length_mm=10.0)
    assert bore == np.pi * 25.0 * 10.0
    raw = volume_error_pct(900.0, 1000.0, 0.0)
    adjusted = volume_error_pct(900.0, 1000.0, 100.0)
    assert raw == -10.0
    assert adjusted == 0.0
    check = volume_check(900.0, 1000.0, 100.0)
    assert check["mesh_adjusted_mm3"] == 900.0 and check["error_pct"] == 0.0


def test_support_samples_on_a_hand_entered_bore_are_ignored():
    pts = np.array([[0.0, 0.0, 1.0], [5.0, 0.0, 1.0], [0.0, 3.0, 1.0]])
    zones = [{"type": "cyl", "point": [0, 0, 0], "dir": [0, 0, 1], "r": 2.0,
              "reason": "hand-entered feature", "kind": "bore"}]
    mask = mask_on_cylinders(pts, zones)
    assert mask.tolist() == [True, False, False]


def test_valid_solid_is_one_closed_brepcheck_solid_and_count_is_logged():
    ok = valid_solid_record(1, True, True)
    assert ok["valid_solid"] is True and ok["n_solids"] == 1
    assert valid_solid_record(2, True, True)["valid_solid"] is False
    assert valid_solid_record(1, True, False)["valid_solid"] is False
    assert valid_solid_record(1, False, True)["valid_solid"] is False
    missing = valid_solid_record(None, False, False)
    assert missing["n_solids"] == 0 and missing["valid_solid"] is False
    stage = impeller_build_stage({"valid": True, "n_solids": None, "closed": False, "valid_solid": False})
    assert stage["n_solids"] == 0
    good = impeller_build_stage({"valid": True, "n_solids": 1, "closed": True, "valid_solid": True,
                                 "n_faces": 4, "volume": 1.0, "seconds": 1.0})
    assert good["valid_solid"] is True and good["n_solids"] == 1 and good["ok"] is True


def test_declared_source_is_authoritative_and_classifier_cannot_move_the_gate():
    gates = {"rms_limit": 0.05, "p95_limit": 0.10}
    out, rec = apply_cross_check({"class": "lowres_tessellation"}, gates, "scan")
    assert out["rms_limit"] == 0.05 and out["p95_limit"] == 0.10
    assert rec["gates"] == "unchanged" and rec["agree"] is False and "does not set or tighten" in rec["warning"]
    same, rec2 = apply_cross_check({"class": "scan"}, gates, "scan")
    assert same == gates and rec2["agree"] is True and rec2["warning"] is None
    text = (ROOT / "meshsolid/pipeline.py").read_text()
    assert "apply_cross_check" in text and "lowres_gates" not in text


def test_parts_csv_marks_real_scans_and_lookup_uses_source():
    parts = ROOT / "parts.csv"
    rows = list(csv.DictReader(parts.open(encoding="utf-8")))
    by_id = {}
    for row in rows:
        by_id.setdefault(row["part_id"], row["source"])
    for part_id in ("s3", "x2b1941", "kk-and-k-s1", "kk-and-k-s1-nohirth", "x2b1872-pinion"):
        assert by_id[part_id] == "scan"
    sha = rows[0]["input_mesh_sha256"]
    assert lookup_parts_csv(parts, sha) == "scan"
    assert reconcile("cad_export", "scan")["agree"] is False


def test_fuse_profile_is_timing_only_and_bench_is_off_by_default():
    prof = FuseProfile()
    assert prof.call("fuse_base_blades", lambda: "solid") == "solid"
    table = prof.table()
    assert table["ops"][0]["op"] == "fuse_base_blades" and table["ops"][0]["seconds"] >= 0
    assert "default serial fuse" in table["note"]
    assert bench_enabled(False) is False
    assert bench_enabled(True) is True
    assert bench_enabled(False, env="0") is False
    assert {row["name"] for row in BENCH_VARIANTS} == {
        "serial_glue_off", "parallel_glue_off", "serial_glue_shift", "parallel_glue_shift"}
    src = (ROOT / "meshsolid/impeller.py").read_text()
    assert "BRepAlgoAPI_Fuse(base, blades)" in src
    assert "bench_enabled(fuse_bench)" in src
    assert "used_for_solid" not in src or "default solid is unchanged" in (ROOT / "meshsolid/fuse_profile.py").read_text()


def test_remeasure_lists_every_metric_that_moves_between_v2_and_v3():
    v2 = {"rms_uncapped": 0.04, "pct_excluded": 8.0, "spec_version": "v2", "balance_cut_count": None}
    v3 = {"rms_uncapped": 0.04, "pct_excluded": 3.0, "spec_version": "v3", "balance_cut_count": 2,
          "volume_error_pct": 0.2}
    diff = compare_metrics(v2, v3)
    moved = {row["metric"] for row in diff["changes"]}
    assert "rms_uncapped" not in moved
    assert moved == {"pct_excluded", "spec_version", "balance_cut_count", "volume_error_pct"}
    report = {"spec_version": "v3", "stats": {"spec_metrics": {"pct_excluded": 3.0},
                                              "balance_cuts": {"count": 2, "pct_of_scan": 1.5, "area_mm2": 40},
                                              "volume_check": {"error_pct": 0.2},
                                              "solid_to_scan_gate": {"unsupported_pct": 0.1, "result": "PASS"}}}
    flat = flat_metrics(report)
    assert flat["spec_version"] == "v3" and flat["balance_cut_count"] == 2 and flat["volume_error_pct"] == 0.2


def test_scoreboard_row_is_tagged_spec_v3(tmp_path):
    job = tmp_path / "run_v3"
    job.mkdir()
    mesh = job / "mesh_body.stl"
    mesh.write_bytes(b"solid mesh")
    (job / "mesh_body.step").write_bytes(b"ISO-10303-21;")
    (job / "mesh_body.report.json").write_text(json.dumps({
        "version": "2.0.1", "spec_version": "v3", "status": "ok", "end_reason": "ok",
        "mesh": {"faces": 4, "volume": 1.0},
    }))
    parts = tmp_path / "parts.csv"
    parts.write_text("part_id,input_mesh_sha256,source\ns3," + scoreboard.sha256(mesh) + ",scan\n")
    out = tmp_path / "scoreboard.csv"
    rc = scoreboard.main(["--jobs", str(job), "--parts", str(parts), "-o", str(out),
                          "--engine", "2.0.1", "--solid-version", "2.0.1"])
    assert rc == 0
    rows = list(csv.DictReader(out.open()))
    assert rows[0]["spec"] == "v3"
    assert rows[0]["engine_version"] == "2.0.1"
    assert rows[0]["part_id"] == "s3"
