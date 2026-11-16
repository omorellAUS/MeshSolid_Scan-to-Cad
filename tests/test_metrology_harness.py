"""Measurement spec v1 + regression harness on synthetic data (no user part files)."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.test_metrology import _box_step  # noqa: E402

# Pinned: editing SPEC without bumping SPEC_VERSION (and this pin) fails here on purpose.



def test_spec_is_frozen_and_versioned():
    from meshsolid.metrology_spec import SPEC, SPEC_VERSION, spec_hash
    assert SPEC["spec_version"] == SPEC_VERSION == "v1"
    assert SPEC["units"] == "mm" and SPEC["outlier_cap"]["cap_mm"] == 0.2 and SPEC["pass"]["tol_mm"] == 0.05
    assert SPEC["exclusions"]["default_zone_tags"] == []          # no 'tip' / backface / bore by default
    assert spec_hash() == "05d7048b9f9d", "spec changed: bump SPEC_VERSION and re-pin the hash"


def test_metrics_count_capped_points_as_failures():
    from meshsolid.metrology_spec import metrics
    dv = np.array([0.01, 0.03, -0.04, 0.1, 0.3, -0.5, 0.02, 1.0])
    excl = np.zeros(8, bool); excl[7] = True                       # named exclusion
    cap = (np.abs(dv) > 0.2) & ~excl                                # idx 4, 5
    m = metrics(dv, excl, cap, tol=0.05)
    assert m["n_scan"] == 8 and m["n_inspected"] == 7 and m["n_excluded"] == 1 and m["capped_count"] == 2
    assert m["pass_pct_v1"] == pytest.approx(100 * 4 / 7, abs=1e-3)       # capped = failures
    assert m["pass_pct_legacy"] == pytest.approx(100 * 4 / 5, abs=1e-3)   # 1.0.1: capped dropped
    d7 = dv[:7]
    assert m["rms_uncapped"] == pytest.approx(np.sqrt(np.mean(d7 ** 2)), abs=1e-5)
    keep = np.array([0, 1, 2, 3, 6])
    assert m["rms_capped"] == pytest.approx(np.sqrt(np.mean(dv[keep] ** 2)), abs=1e-5)
    assert m["max_plus"] == 0.3 and m["max_minus"] == -0.5                  # true extremes, excl. named only
    assert m["pct_excluded"] == pytest.approx(12.5)
    assert m["p99"] >= m["p95"] and not m["target_met_v1"]


def test_pose_of_recovers_rotation_and_translation():
    from meshsolid.metrology import pose_of
    ang = np.radians(0.1)
    T = np.eye(4)
    T[:3, :3] = trimesh.transformations.rotation_matrix(ang, [0, 0, 1])[:3, :3]
    T[:3, 3] = [0.03, -0.02, 0.01]
    p = pose_of(T)
    assert p["rotation_deg"] == pytest.approx(0.1, abs=1e-6)
    assert np.allclose(p["rotation_axis"], [0, 0, 1], atol=1e-6)
    assert np.allclose(p["translation_mm"], [0.03, -0.02, 0.01])
    assert pose_of(None)["rotation_deg"] == 0.0
    T180 = trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])
    assert pose_of(T180)["rotation_deg"] == pytest.approx(180.0)


def _raised_scan(tmp_path, raise_mm=0.5, extents=(20.0, 20.0, 20.0)):
    step = _box_step(tmp_path / "box.step", extents)
    scan = trimesh.creation.box(extents=extents).subdivide().subdivide()
    top = scan.vertices[:, 2] > extents[2] / 2 - 0.01
    scan.vertices[top, 2] += raise_mm
    scan.export(tmp_path / "scan.stl")
    return step, tmp_path / "scan.stl", top


def _rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_harness_appends_rows_with_both_pass_forms(tmp_path):
    from scripts import metrology_harness as H
    step, scan, top = _raised_scan(tmp_path)
    out_csv = tmp_path / "harness.csv"
    parts = [{"part": "box_raised", "scan": str(scan), "solid": str(step), "align": False}]
    rows = H.run_parts(parts, out_csv, tmp_path / "runs", device="cpu", label="t1")
    r = rows[0]
    assert r["status"] == "ok" and r["spec_version"] == "v1" and r["deviations"] == "align=off"
    assert r["capped_count"] == int(top.sum())                       # every raised vertex beyond 0.2
    assert r["pass_pct_v1"] < r["pass_pct_legacy"] == pytest.approx(100.0, abs=0.01)
    assert r["pass_pct_v1"] == pytest.approx(100.0 * (len(top) - top.sum()) / len(top), abs=0.01)
    assert r["rms_uncapped"] > 0.1 > r["rms_capped"]
    assert r["max_plus"] == pytest.approx(0.5, abs=0.01) and r["pct_excluded"] == 0.0
    # second run appends a row under the same header
    H.run_parts(parts, out_csv, tmp_path / "runs", device="cpu", label="t2")
    got = _rows(out_csv)
    assert len(got) == 2 and list(got[0]) == H.COLUMNS
    assert got[0]["label"] == "t1" and got[1]["label"] == "t2"
    assert got[0]["rms_uncapped"] == got[1]["rms_uncapped"]          # reproducible
    assert got[0]["spec_hash"] and got[0]["git_commit"]


def test_harness_named_mask_exclusion_and_breakdown(tmp_path):
    from scripts import metrology_harness as H
    from meshsolid.metrology import load_scan
    step, scan, _top = _raised_scan(tmp_path)
    top = np.asarray(load_scan(scan).vertices)[:, 2] > 10.2      # mask in the order Metrology loads the scan
    np.save(tmp_path / "mask.npy", top)
    parts = [{"part": "masked", "scan": str(scan), "solid": str(step), "align": False,
              "exclude_mask_npy": str(tmp_path / "mask.npy"), "exclude_mask_reason": "raised lid (test)"}]
    r = H.run_parts(parts, tmp_path / "h.csv", tmp_path / "runs", device="cpu")[0]
    assert r["capped_count"] == 0 and r["n_excluded"] == int(top.sum())
    assert "raised lid (test)=" in r["excluded_by_reason"]
    assert r["pass_pct_v1"] == pytest.approx(100.0, abs=0.01)
    with pytest.raises(ValueError):
        H.measure({"part": "x", "scan": str(scan), "solid": str(step), "exclude_mask_npy": str(tmp_path / "mask.npy")},
                  tmp_path / "x", device="cpu")


def test_harness_alignment_pose_logged(tmp_path):
    from scripts import metrology_harness as H
    ext = (20.0, 30.0, 44.0)
    step = _box_step(tmp_path / "box.step", ext)
    scan = trimesh.creation.box(extents=ext).subdivide().subdivide()
    T = trimesh.transformations.rotation_matrix(np.radians(3.0), [0, 0, 1])
    T[:3, 3] = [0.4, -0.3, 0.2]
    scan.apply_transform(T)
    scan.export(tmp_path / "moved.stl")
    r = H.run_parts([{"part": "moved", "scan": str(tmp_path / "moved.stl"), "solid": str(step)}],
                    tmp_path / "h.csv", tmp_path / "runs", device="cpu")[0]
    assert r["align"] is True and r["deviations"] == "none"
    # a box has 180-degree symmetric copies; spec-v1 ICP may land on one (known, logged): the pose is either
    # the 3-degree inverse or that composed with a half turn
    assert min(abs(r["rot_deg"] - 3.0), abs(r["rot_deg"] - 177.0), abs(r["rot_deg"] - 180.0)) < 3.1
    assert r["rot_axis"] and r["icp_median_pre_mm"] is not None
    assert r["rms_uncapped"] < 0.02 and r["pass_pct_v1"] > 99.0


def test_harness_error_is_a_row_and_header_guard(tmp_path):
    from scripts import metrology_harness as H
    out_csv = tmp_path / "h.csv"
    rows = H.run_parts([{"part": "missing", "scan": str(tmp_path / "nope.stl"), "solid": str(tmp_path / "nope.step")}],
                       out_csv, tmp_path / "runs", device="cpu")
    assert rows[0]["status"] == "error" and "nope" in rows[0]["error"]
    assert _rows(out_csv)[0]["status"] == "error"
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b\n1,2\n")
    with pytest.raises(SystemExit):
        H.append_rows(bad, [{"part": "x"}])


def test_spec_v2_is_frozen_and_only_alignment_differs():
    from meshsolid.metrology_spec import SPEC_V1, SPEC_V2, SPECS, spec_hash
    assert SPECS["v1"] is SPEC_V1 and SPECS["v2"] is SPEC_V2 and SPEC_V2["spec_version"] == "v2"
    assert {k for k in SPEC_V1 if SPEC_V1[k] != SPEC_V2.get(k)} == {"spec_version", "alignment"}
    a = SPEC_V2["alignment"]
    assert a["min_points"] == 500000 and a["icp_trim_pct"] == 5.0
    assert a["converge_deg"] == 0.0005 and a["converge_mm"] == 0.001
    assert a["warn_refine_deg"] == 0.01 and a["warn_refine_mm"] == 0.01
    assert spec_hash(SPEC_V2) == "42ceb29d6b7b", "spec v2 changed: bump the version and re-pin the hash"


def test_honest_verdict_and_engine_limit_side_by_side():
    from meshsolid.metrology_spec import metrics, verdict
    dv = np.r_[np.full(95, 0.01), np.full(5, 0.5)]                 # 5 % far outside
    excl = np.zeros(100, bool)
    cap = np.abs(dv) > 0.2
    m = metrics(dv, excl, cap)
    v = verdict(m, legacy_within_pct=100.0, accept_pct=95.0)
    # 1.0.1 dropped the 5 capped points and called it PASS; the target verdict fails on rms and keeps them as failures
    assert v["engine_limit"]["result"] == "PASS" and "0.0 % excluded" in v["engine_limit"]["label"]
    assert v["target"]["result"] == "FAIL" and "rms_uncapped" in v["target"]["failed"]
    ex = np.zeros(100, bool); ex[:6] = True                         # 6 % excluded by a named reason
    m2 = metrics(np.full(100, 0.01), ex, np.zeros(100, bool))
    v2 = verdict(m2, legacy_within_pct=100.0, accept_pct=95.0)
    assert v2["target"]["result"] == "FAIL" and v2["target"]["failed"] == ["pct_excluded"]
    assert v2["engine_limit"]["result"] == "REFUSED" and "6.0 % excluded" in v2["engine_limit"]["label"]
    m3 = metrics(np.full(100, 0.01), np.zeros(100, bool), np.zeros(100, bool))
    v3 = verdict(m3, 100.0, 95.0)
    assert v3["target"]["result"] == "PASS" and v3["target"]["label"] == "PASS · 0.0 % excluded"
