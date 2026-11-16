"""End-to-end tests on synthetic noisy scans with known dimensions."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh

from meshsolid import synthetic as S
from meshsolid.occ_util import check, read_step, volume
from meshsolid.pipeline import Params, run

NOISE = 0.02   # mm, gaussian along normals
ROOT = Path(__file__).resolve().parents[1]
_COUPLING_CANDIDATES = [
    Path(os.environ["COUPLING_STL"]) if os.environ.get("COUPLING_STL") else None,
    ROOT / "tests" / "user_scans" / "mechanical" / "Six-Lobed Coupling Element (Mechanical Component).stl",
    Path("/workspace/AccuReverse-ScanToSolid/tests/Six-Lobed Coupling Element (Mechanical Component).stl"),
]
COUPLING = next((p for p in _COUPLING_CANDIDATES if p is not None and p.exists()), _COUPLING_CANDIDATES[-1])


def _scan(tmp, name, shape, rotate=None, seed=1):
    m = S.to_scan(shape, noise=NOISE, seed=seed, rotate=rotate)
    p = tmp / f"{name}.stl"
    m.export(p)
    return p


def _dims(rep, kind, key="value"):
    return sorted(d[key] for d in rep["dimensions"] if d["type"] == kind)


def _common_checks(rep, shape_out, step, truth_volume):
    assert rep["status"] == "ok", rep.get("failure")
    b = rep["stages"]["build"]
    assert b["brepcheck_valid"] and b["n_solids"] == 1
    assert abs(b["volume"] - truth_volume) / truth_volume < 2e-3
    dev = rep["deviation"]["mesh_to_solid"]
    assert dev["rms"] < 1.5 * NOISE and dev["max"] < 6 * NOISE
    assert rep["deviation"]["solid_to_mesh"]["p95"] < 4 * NOISE
    s2 = read_step(step)              # STEP round-trip
    # 1e-5 relative: STEP stores coordinates/knots as decimal text and the reader rebuilds
    # pcurves/tolerances, so the reloaded volume differs by float round-trip noise (seen up to
    # 1.2e-6 on plate_with_fillets, varying run to run); a real geometry change is orders larger
    assert check(s2) and abs(volume(s2) - b["volume"]) < 1e-5 * b["volume"]
    for r in rep["regions"]:
        assert "params" in r and "deviation" in r and "confidence" in r


def test_box_with_hole_round_dims(tmp_path):
    shape, t = S.box_with_hole()
    stl = _scan(tmp_path, "box", shape)
    step = tmp_path / "box.step"
    rep, *_ = run(stl, step, tmp_path / "r.json")
    _common_checks(rep, None, step, t["volume"])
    assert _dims(rep, "diameter") == pytest.approx([10.0], abs=1e-9)
    assert _dims(rep, "distance") == pytest.approx([20.0, 30.0, 40.0], abs=1e-9)
    assert np.allclose(_dims(rep, "distance", "fit_value"), [20, 30, 40], atol=0.02)


def test_box_nonround_rotated_no_rounding(tmp_path):
    shape, t = S.box_with_hole(L=37.3, W=22.6, H=14.2, D=7.4, cx=12.1, cy=9.7)
    R = trimesh.transformations.euler_matrix(0.3, -0.5, 1.1)
    R[:3, 3] = [5, -7, 12]
    stl = _scan(tmp_path, "boxr", shape, rotate=R)
    step = tmp_path / "boxr.step"
    rep, *_ = run(stl, step, None, Params(do_round=False))
    _common_checks(rep, None, step, t["volume"])
    assert np.allclose(_dims(rep, "diameter"), [7.4], atol=0.02)
    assert np.allclose(_dims(rep, "distance"), [14.2, 22.6, 37.3], atol=0.02)


def test_stepped_shaft(tmp_path):
    shape, t = S.stepped_shaft()
    stl = _scan(tmp_path, "shaft", shape)
    step = tmp_path / "shaft.step"
    rep, *_ = run(stl, step, None)
    _common_checks(rep, None, step, t["volume"])
    assert _dims(rep, "diameter") == pytest.approx(sorted(t["diameters"]), abs=1e-9)
    assert np.allclose(_dims(rep, "diameter", "fit_value"), sorted(t["diameters"]), atol=0.02)
    assert _dims(rep, "distance") == pytest.approx(sorted(t["lengths"]), abs=1e-9)
    assert _dims(rep, "span") == pytest.approx([70.0], abs=1e-9)


def test_plate_with_fillets(tmp_path):
    shape, t = S.plate_with_fillets()
    stl = _scan(tmp_path, "plate", shape)
    step = tmp_path / "plate.step"
    rep, *_ = run(stl, step, None)
    _common_checks(rep, None, step, t["volume"])
    assert _dims(rep, "diameter") == pytest.approx([8.0, 10.0, 10.0, 10.0, 10.0], abs=1e-9)
    assert _dims(rep, "distance") == pytest.approx([8.0, 40.0, 60.0], abs=1e-9)
    fit = _dims(rep, "diameter", "fit_value")
    assert np.allclose(fit[1:], 10.0, atol=0.2)  # raw fillet fits are weaker: ~0.1 mm on R5 strips
    tang = [l for l in rep["stages"]["snapping"]["log"] if l["op"] == "tangent"]
    assert len(tang) == 4


def test_full_torus_builds(tmp_path):
    m = trimesh.creation.torus(major_radius=20, minor_radius=6, major_sections=96, minor_sections=48)
    stl = tmp_path / "torus.stl"
    m.export(stl)
    step = tmp_path / "torus.step"
    rep, shape, *_ = run(stl, step, None)
    assert rep["status"] == "ok"
    assert step.exists()
    vol = rep["stages"]["build"]["volume"]
    assert vol == pytest.approx(2 * np.pi ** 2 * 20 * 36, rel=0.02)


def test_unsupported_surface_fails_honestly(tmp_path):
    # an ellipsoid is not one of the fitted analytic kinds: must fail, no STEP
    m = trimesh.creation.icosphere(subdivisions=3, radius=10)
    m.apply_transform(np.diag([1.0, 0.6, 0.4, 1.0]))
    stl = tmp_path / "ell.stl"
    m.export(stl)
    step = tmp_path / "ell.step"
    rep, shape, *_ = run(stl, step, tmp_path / "ell.json")
    assert rep["status"] == "failed"
    assert rep["failure"]
    assert not step.exists()
    assert json.loads((tmp_path / "ell.json").read_text())["status"] == "failed"


def test_cli(tmp_path):
    shape, _ = S.box_with_hole()
    stl = _scan(tmp_path, "cli", shape)
    out = subprocess.run([sys.executable, "-m", "meshsolid", str(stl), "-o", str(tmp_path / "o.step"),
                          "--report", str(tmp_path / "o.json"), "--measure", "D4=40.02"],
                         cwd=ROOT, capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    rep = json.loads((tmp_path / "o.json").read_text())
    assert rep["status"] == "ok" and (tmp_path / "o.step").exists()
    mc = rep["measurement_checks"][0]
    assert mc["model"] == 40.0 and abs(mc["difference"] + 0.02) < 1e-9


@pytest.mark.slow
@pytest.mark.skipif(not COUPLING.exists(), reason="coupling STL not available")
def test_coupling_is_honest(tmp_path):
    step = tmp_path / "c.step"
    rep, shape, *_ = run(COUPLING, step, None, Params(max_tools=200))
    # Either a validated solid within tolerance, or an explicit failure with no STEP.
    if rep["status"] == "ok":
        assert step.exists() and rep["deviation"]["acceptance"]["passed"]
    else:
        assert rep["failure"] and not step.exists()
