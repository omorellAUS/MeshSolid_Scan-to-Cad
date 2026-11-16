"""Per-blade pose/thickness fit recovers known per-blade scatter (values and sign) on a synthetic 15-fold part."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from meshsolid import synthetic_nfold as S  # noqa: E402
from meshsolid.metrology import run  # noqa: E402
import per_blade_test as PB  # noqa: E402


def test_per_blade_fit_recovers_scatter(tmp_path):
    solid, owner = S.part(15)
    br = S.balanced_scatter(15, 0.05, seed=3)
    bt = S.balanced_scatter(15, 0.02, seed=4)
    scan, _ = S.scan_of(solid, owner, noise=0.013, blade_rot_deg=br, blade_thick_mm=bt, T=S.known_motion())
    solid.export(tmp_path / "s.stl")
    scan.export(tmp_path / "c.stl")
    rep = run(tmp_path / "c.stl", tmp_path / "s.stl", tmp_path / "o", align=True, align_method="v2", n_fold=15,
              device="cpu", do_map=False, do_zones=False, do_histogram=False, do_section=False, pdf=False,
              max_dist=0.2, keep_arrays=True)
    A = rep.pop("_arrays")
    insp = ~(A["excluded"] & ~A["beyond_cap"])
    r = PB.blade_fit(A["dv"], A["points"], A["normals_solid"], A["sector"], 15, A["axis"], insp, A["beyond_cap"])
    got = np.array([p["rot_deg"] for p in r["per_blade"]])
    th = np.array([p["thickness_change_mm"] for p in r["per_blade"]])
    # sector numbering starts at an arbitrary blade: compare under the best cyclic shift (same sign required)
    c = [np.corrcoef(np.roll(got, k), br)[0, 1] for k in range(15)]
    k = int(np.argmax(c))
    assert c[k] > 0.99
    assert np.max(np.abs(np.roll(got, k) - br)) < 0.01
    assert np.corrcoef(np.roll(th, k), 2 * bt)[0, 1] > 0.99
    assert np.max(np.abs(np.roll(th, k) - 2 * bt)) < 0.005
    assert r["per_blade_spread_std"]["axial_mm"] < 0.01      # unobservable axial term stays bounded
    assert r["blade_rms"]["after_per_blade_fit"] < 0.015     # ~ the 0.013 noise floor
