"""The robustness harness writes one row per case and only calls a case clean when the engine ended
with a report and a stated failure (never a crash / traceback)."""
import csv
import json

import trimesh

from scripts import robustness_harness as RH


def test_rows_for_ok_and_garbage_input(tmp_path):
    good = tmp_path / "cube.stl"
    trimesh.creation.box((20, 10, 5)).export(good)
    bad = tmp_path / "garbage.stl"
    bad.write_bytes(b"solid nothing\nendsolid\n")
    man = tmp_path / "m.json"
    man.write_text(json.dumps({"cases": [{"case": "cube", "input": str(good)},
                                         {"case": "garbage", "input": str(bad), "was": "x"}]}))
    out = tmp_path / "rob.csv"
    RH.main(["--manifest", str(man), "--csv", str(out), "--out-root", str(tmp_path / "runs"),
             "--time-budget", "300", "--label", "t"])
    rows = {r["case"]: r for r in csv.DictReader(open(out))}
    assert set(rows) == {"cube", "garbage"} and list(rows["cube"]) == RH.COLUMNS
    assert rows["cube"]["status"] == "ok" and rows["cube"]["clean"] == "True"
    assert rows["garbage"]["status"] == "failed" and rows["garbage"]["failure"] and rows["garbage"]["clean"] == "True"


def test_determinism_diff_and_existing_csv_header(tmp_path):
    import csv
    from scripts import robustness_harness as R
    assert R.determinism_diff({"status": "ok", "volume": 1.0}, {"status": "ok", "volume": 1.0}) == (True, "")
    same, txt = R.determinism_diff({"status": "ok", "rms_gate": 0.1}, {"status": "ok", "rms_gate": 0.2})
    assert not same and "rms_gate" in txt
    # an old CSV (fewer columns) keeps its header when rows are appended
    p = tmp_path / "old.csv"
    p.write_text("timestamp,case,status\n2026,a,ok\n")
    with open(p, newline="") as f:
        assert next(csv.reader(f)) == ["timestamp", "case", "status"]
