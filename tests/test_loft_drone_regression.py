"""Fixed floors for curved-skin loft on three drone parts — catch silent regressions.

User STLs are not in git; tests skip when absent. Floors are from round 2i/2j evidence:
- WINGTIP L 2: rms <= 0.059 (1d96ac0 baseline; dense_at=0.5 regressed to 0.104)
- AILERON: vol ratio within 5% and 1 solid (tip-apex false positive ballooned to ~10^4×)
- WING 2 L: corner-count breaks fire ( >=3 levels ) and build returns under budget without OCC hang
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from meshsolid import loft  # noqa: E402
from meshsolid.facets import detect, tolerance as facet_tol  # noqa: E402
from meshsolid.meshio import load_mesh  # noqa: E402

MANIFEST = ROOT / "results" / "drone_stl_step_pairing" / "manifest.json"
SCAN_ROOT = ROOT / "tests" / "user_scans" / "drone"

pytestmark = pytest.mark.skipif(
    not MANIFEST.exists() or not SCAN_ROOT.exists(),
    reason="drone pairing manifest / user_scans not present",
)


def _part(stl_name: str):
    man = json.loads(MANIFEST.read_text())
    p = next(x for x in man["matched"] if x["stl_name"] == stl_name)
    path = next(SCAN_ROOT.rglob(p["stl"]), None)
    if path is None:
        pytest.skip(f"STL for {stl_name} not found under {SCAN_ROOT}")
    mesh, _ = load_mesh(path)
    fac, st = detect(mesh, 0.0, True)
    diag = float(np.linalg.norm(np.ptp(mesh.vertices, axis=0)))
    tol = facet_tol(st, diag) if fac else max(0.05, 5e-5 * diag)
    return mesh, tol


def test_wingtip_l2_rms_floor():
    """Must not regress past the 1d96ac0 / 2i wingtip loft (rms 0.059)."""
    mesh, tol = _part("WINGTIP L 2")
    t0 = time.time()
    res = loft.build_loft(mesh, tol=tol, corner_deg=0.0, max_sections=80, self_check=True, time_budget=55.0)
    assert time.time() - t0 < 60.0, "wingtip loft overrun"
    assert res.n_solids == 1 and res.volume and abs(res.volume / res.mesh_volume - 1.0) < 0.02
    sc = res.self_check or {}
    assert sc.get("rms") is not None, res.reason
    assert sc["rms"] <= 0.059, f"WINGTIP L 2 rms {sc['rms']} > 0.059 (reason={res.reason})"


def test_aileron_volume_not_ballooned():
    """Tip-apex false positive ballooned aileron to ~10^4× mesh; vol must stay near 1."""
    mesh, tol = _part("AILERON 1R VTOL")
    t0 = time.time()
    res = loft.build_loft(mesh, tol=tol, corner_deg=0.0, max_sections=33, self_check=False, time_budget=40.0)
    assert time.time() - t0 < 50.0
    assert res.n_solids == 1, res.reason
    assert res.volume and abs(res.volume / res.mesh_volume - 1.0) < 0.05, (
        f"aileron vol ratio {res.volume / res.mesh_volume} (reason={res.reason})"
    )


def test_wing2l_corner_breaks_fire_under_budget():
    """Corner-count slab breaks must fire; build must return (no silent OCC hang)."""
    mesh, tol = _part("WING 2 L")
    t0 = time.time()
    res = loft.build_loft(mesh, tol=tol, corner_deg=0.0, max_sections=36, self_check=False, time_budget=90.0)
    elapsed = time.time() - t0
    assert elapsed < 100.0, f"WING 2 L hung/overran ({elapsed:.1f}s)"
    assert res.levels is not None and len(res.levels) >= 3, (
        f"expected corner-count breaks (>=3 levels), got {res.levels}"
    )
    # Soft: either fused to 1 solid or left an explicit reason (not an empty crash)
    assert res.reason or res.n_solids == 1 or res.slabs
