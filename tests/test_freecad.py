"""FreeCAD workbench (freecad_addin/MeshSolid) under a real FreeCAD 1.0: non-GUI core through
freecadcmd (engine run, STEP import as one Part solid, ms_ aliases in the spreadsheet) and the GUI
flow (run panel, progress, review, import, failure, cancel) under xvfb.  Skipped without FreeCAD."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FC = shutil.which("freecadcmd") or shutil.which("FreeCADCmd")
pytestmark = pytest.mark.skipif(FC is None, reason="FreeCAD (freecadcmd) not installed")


def _engine_mesh(tmp_path):
    from meshsolid import synthetic as S
    p = tmp_path / "block.stl"; S.to_scan(S.chamfered_block()[0]).export(p)
    return p


def _settings(tmp_path):
    p = tmp_path / "settings.json"
    p.write_text(json.dumps({"python": sys.executable, "engine_dir": str(ROOT), "timeout_s": 600, "extra_args": []}))
    return p


def _home(tmp_path):
    h = tmp_path / "fchome"; (h / "Mod").mkdir(parents=True)
    os.symlink(ROOT / "freecad_addin" / "MeshSolid", h / "Mod" / "MeshSolid")
    return h


def _env(tmp_path, **kw):
    env = dict(os.environ, MESHSOLID_SETTINGS=str(_settings(tmp_path)), MS_FC_REPO=str(ROOT),
               FREECAD_USER_HOME=str(_home(tmp_path)), LOCALAPPDATA=str(tmp_path / "local"))
    env.update({k: str(v) for k, v in kw.items()})
    return env


def test_core_engine_run_and_import(tmp_path):
    mesh = _engine_mesh(tmp_path)
    sys.path.insert(0, str(ROOT / "fusion_addin" / "MeshSolid"))
    from mslib.engine_client import EngineSettings, EngineJob, EngineRun, read_result
    job = EngineJob(str(mesh), str(tmp_path / "job"))
    run = EngineRun(job, EngineSettings(python=sys.executable, engine_dir=str(ROOT), timeout_s=600))
    import time
    while run.poll() is None:
        time.sleep(0.3)
    res = read_result(job, run.poll())
    assert res["outcome"]["importable"], res["outcome"]
    out = tmp_path / "r.json"
    subprocess.run([FC, "-c", f"exec(open({str(ROOT / 'tests/freecad/import_check.py')!r}).read())"],
                   env=_env(tmp_path, MS_FC_STEP=res["step"], MS_FC_REPORT=res["report_path"], MS_FC_RESULT=out),
                   cwd=ROOT, timeout=300, capture_output=True)
    r = json.loads(out.read_text())
    assert "error" not in r, r.get("error")
    assert r["solid"]["n_solids"] == 1 and r["solid"]["valid"]
    assert abs(r["solid"]["volume"] - res["report"]["stages"]["build"]["volume"]) < 1e-3 * r["solid"]["volume"]
    assert r["aliases"] and all(k.startswith("ms_") for k in r["aliases"])


def test_stg1_step_import(tmp_path):
    rep = ROOT / "out/user/stg1/pipe4/report.json"; step = ROOT / "out/user/stg1/pipe4/stg1.step"
    if not step.exists():
        pytest.skip("user part not on this machine")
    out = tmp_path / "r.json"
    subprocess.run([FC, "-c", f"exec(open({str(ROOT / 'tests/freecad/import_check.py')!r}).read())"],
                   env=_env(tmp_path, MS_FC_STEP=step, MS_FC_REPORT=rep, MS_FC_RESULT=out), cwd=ROOT, timeout=600,
                   capture_output=True)
    r = json.loads(out.read_text())
    assert "error" not in r, r.get("error")
    assert r["solid"]["n_solids"] == 1 and r["solid"]["valid"] and r["solid"]["faces"] == 276
    assert r["aliases"]["ms_imp_blades"] == "15" and r["aliases"]["ms_imp_bore_d"] == "14.29 mm"


@pytest.mark.skipif(shutil.which("xvfb-run") is None or shutil.which("freecad") is None, reason="no xvfb-run/freecad GUI")
def test_gui_flow(tmp_path):
    mesh = _engine_mesh(tmp_path)
    out = tmp_path / "gui.json"
    subprocess.run(["xvfb-run", "-a", "freecad", str(ROOT / "tests/freecad/gui_flow.FCMacro")],
                   env=_env(tmp_path, MS_FC_RESULT=out, MS_FC_MESH=mesh), cwd=ROOT, timeout=400, capture_output=True)
    r = json.loads(out.read_text())
    assert "error" not in r, r.get("error")
    assert r["steps"] == ["run started", "imported", "failure reported", "cancel reported"]
    assert r["solids"] and r["solids"][0][1:3] == [1, True]
    assert r["alias_value"].endswith("mm") and "does not drive" in r["sheet_rows"][1][4]
    assert r["failure_msg"][0] == "Mesh could not be used" and "no triangles" in r["failure_msg"][1]
    assert r["cancel_msg"][0] == "Cancelled" and not r["dialog_open_at_end"]
