"""Scoreboard from Step 0: Path is imported, and a job row round-trips."""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scoreboard  # noqa: E402


def test_path_is_imported():
    assert scoreboard.Path is Path


def test_scoreboard_row_for_a_job(tmp_path):
    job = tmp_path / "run_20261126"
    job.mkdir()
    mesh = job / "mesh_body.stl"
    mesh.write_bytes(b"solid mesh")
    (job / "mesh_body.step").write_bytes(b"ISO-10303-21;")
    (job / "mesh_body.report.json").write_text(json.dumps({
        "version": "1.0.2a", "status": "ok", "end_reason": "ok", "seconds": 3.5,
        "mesh": {"faces": 12, "volume": 1.0},
        "deviation": {"acceptance": {"passed": True},
                      "mesh_vertices_to_solid_excl": {"rms": 0.01, "p95": 0.02}},
    }))
    parts = tmp_path / "parts.csv"
    parts.write_text("part_id,input_mesh_sha256\ns3," + scoreboard.sha256(mesh) + "\n")
    out = tmp_path / "scoreboard.csv"
    rc = scoreboard.main(["--jobs", str(job), "--parts", str(parts), "-o", str(out),
                          "--engine", "1.0.2a", "--commit", "b4f0e607"])
    assert rc == 0
    rows = list(csv.DictReader(out.open()))
    assert len(rows) == 1
    assert rows[0]["part_id"] == "s3"
    assert rows[0]["end_reason"] == "ok"
    assert rows[0]["has_step"] == "yes"
    assert rows[0]["commit"] == "b4f0e607"
    assert rows[0]["input_mesh_sha256"] == scoreboard.sha256(mesh)
