"""Unusable input meshes fail cleanly: status failed, a plain input_error in the report, exit code 2,
no traceback, no STEP."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh

ROOT = Path(__file__).resolve().parents[1]


def _cases(tmp):
    c = {}
    c["garbage"] = tmp / "garbage.stl"; c["garbage"].write_bytes(b"not a mesh at all\x00\x01")
    c["empty"] = tmp / "empty.stl"; c["empty"].write_bytes(b"")
    c["missing"] = tmp / "missing.stl"
    c["flat"] = tmp / "flat.stl"
    trimesh.Trimesh([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], [[0, 1, 2], [1, 3, 2], [0, 2, 1], [1, 2, 3]],
                    process=False).export(c["flat"])
    c["nan"] = tmp / "nan.obj"
    c["nan"].write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nv nan 0 1\nf 1 2 3\nf 1 2 4\nf 2 3 4\nf 1 3 4\n")
    c["folder"] = tmp
    return c


@pytest.mark.parametrize("kind", ["garbage", "empty", "missing", "flat", "nan", "folder"])
def test_bad_input_fails_cleanly(tmp_path, kind):
    path = _cases(tmp_path)[kind]
    rp = tmp_path / f"{kind}.report.json"; step = tmp_path / f"{kind}.step"
    p = subprocess.run([sys.executable, "-m", "meshsolid", str(path), "-o", str(step), "--report", str(rp)],
                       cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert p.returncode == 2, p.stderr[-2000:]
    assert "Traceback" not in p.stderr
    rep = json.loads(rp.read_text())
    assert rep["status"] == "failed" and rep["input_error"] and rep["failure"].startswith("input mesh:")
    assert not step.exists()


def test_outcome_bad_input():
    sys.path.insert(0, str(ROOT / "fusion_addin" / "MeshSolid"))
    from mslib.outcome import explain
    o = explain(2, {"status": "failed", "input_error": "x.stl is empty (0 bytes)", "failure": "input mesh: ..."}, "/none")
    assert o["kind"] == "bad_input" and "empty" in o["message"] and not o["importable"]
