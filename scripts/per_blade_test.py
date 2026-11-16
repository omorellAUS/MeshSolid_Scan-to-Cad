"""Per-blade pose / thickness test (plan step 6 preview): how much rms would per-blade modelling remove?

For every blade (sector about the axis, following the blade wrap) fit, on the exact signed deviations of the
blade-surface vertices (solid normal tangential, |n . e_t| > 0.5), after the spec-v2 alignment:
    d_i = a * n_i.(u x (p_i - c))   (rotation of the blade about the axis, rad)
        + b * n_i.u                 (axial shift, mm)
        + g                         (uniform normal offset = half the thickness change, mm)
First one common set (a, b, g) for all blades (what an N-fold model can still fix), then one set per blade
(what only per-blade modelling can fix). Points beyond a robust 6-sigma MAD gate (unmodelled features) are
left out of the fits but kept in the before/after statistics.

    python scripts/per_blade_test.py --manifest m.json --parts stg1_impeller,stage3_impeller --out DIR
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from meshsolid import metrology as M  # noqa: E402
from meshsolid.align_v2 import _gate  # noqa: E402
from meshsolid.metrology_spec import SPEC_V2, metrics  # noqa: E402


def _rms(x):
    return float(np.sqrt(np.mean(x * x))) if len(x) else float("nan")


AXIAL_MIN_RMS = 0.05   # rms(n.u) on a blade below this -> axial shift unobservable, column dropped


def _lsq(G, d):
    """Least squares on physically scaled columns (mm per deg, mm per mm) with a truncated SVD (rcond 1e-3).
    Columns whose rms is below AXIAL_MIN_RMS are dropped (reported 0 and flagged unconstrained) rather than
    rescaled, because rescaling a near-zero column only amplifies noise."""
    keep = np.sqrt(np.mean(G * G, axis=0)) >= AXIAL_MIN_RMS
    x = np.zeros(G.shape[1])
    if keep.any():
        x[keep], *_ = np.linalg.lstsq(G[:, keep], d, rcond=1e-3)
    return x


def blade_fit(dv, P, Ns, sector, n_fold, axis, insp, beyond_cap):
    c, u = np.asarray(axis[0], float), np.asarray(axis[1], float)
    u = u / np.linalg.norm(u)
    V = P - c
    rad = V - np.outer(V @ u, u)
    rad /= np.maximum(np.linalg.norm(rad, axis=1), 1e-12)[:, None]
    et = np.cross(u, rad)
    blade = insp & (np.abs(np.einsum("ij,ij->i", Ns, et)) > 0.5) & (sector >= 0)
    G = np.stack([np.einsum("ij,ij->i", Ns, np.cross(u, V)), Ns @ u, np.ones(len(P))], 1)
    # gate computed on blade points only
    g_ok = np.zeros(len(P), bool)
    g_ok[blade] = _gate(dv[blade])
    ok = blade & g_ok
    # common (N-fold) fit
    x0 = _lsq(G[ok], dv[ok])
    d1 = dv.copy()
    d1[blade] = dv[blade] - G[blade] @ x0
    # per-blade fit on top
    d2 = d1.copy()
    per = []
    for b in range(int(n_fold)):
        m = ok & (sector == b)
        mb = blade & (sector == b)
        if m.sum() < 50:
            per.append({"blade": b, "n": int(m.sum()), "skipped": True})
            continue
        xb = _lsq(G[m], d1[m])
        d2[mb] = d1[mb] - G[mb] @ xb
        per.append({"blade": b, "n": int(mb.sum()), "rot_deg": round(float(np.degrees(xb[0])), 5),
                    "axial_mm": round(float(xb[1]), 5),
                    "axial_observable": bool(_rms(G[m, 1]) >= AXIAL_MIN_RMS), "normal_offset_mm": round(float(xb[2]), 5),
                    "thickness_change_mm": round(float(2 * xb[2]), 5),
                    "rms_before": round(_rms(d1[mb]), 5), "rms_after": round(_rms(d2[mb]), 5)})
    excl = ~insp
    tol = SPEC_V2["pass"]["tol_mm"]
    cap = SPEC_V2["outlier_cap"]["cap_mm"]
    def whole(d):
        return metrics(d, excl, (np.abs(d) > cap) & insp, tol=tol)
    vals = [p for p in per if not p.get("skipped")]
    spread = {k: round(float(np.std([p[k] for p in vals])), 5) for k in ("rot_deg", "axial_mm", "thickness_change_mm")}
    r_tip = float(np.percentile(np.linalg.norm(rad * 0 + (V - np.outer(V @ u, u)), axis=1)[blade], 95))
    return {
        "n_blade_points": int(blade.sum()), "n_fit_points": int(ok.sum()),
        "blade_rms": {"aligned_v2": round(_rms(dv[blade]), 5), "after_common_fit": round(_rms(d1[blade]), 5),
                      "after_per_blade_fit": round(_rms(d2[blade]), 5)},
        "blade_rms_gated_points": {"aligned_v2": round(_rms(dv[ok]), 5), "after_common_fit": round(_rms(d1[ok]), 5),
                                   "after_per_blade_fit": round(_rms(d2[ok]), 5)},
        "common_fit": {"rot_deg": round(float(np.degrees(x0[0])), 5), "axial_mm": round(float(x0[1]), 5),
                       "thickness_change_mm": round(float(2 * x0[2]), 5)},
        "per_blade_spread_std": spread | {"rot_at_tip_mm": round(float(np.radians(spread["rot_deg"]) * r_tip), 5),
                                          "tip_radius_mm": round(r_tip, 2)},
        "whole_part": {"aligned_v2": whole(dv), "after_common_fit": whole(d1), "after_per_blade_fit": whole(d2)},
        "per_blade": per,
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--parts", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args(argv)
    parts = json.loads(Path(a.manifest).read_text(encoding="utf-8-sig"))["parts"]
    want = set(filter(None, a.parts.split(",")))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = {}
    for p in parts:
        if want and p["part"] not in want:
            continue
        rep = M.run(p["scan"], p["solid"], out / p["part"], pass_tol=0.05, max_dist=0.2, align=True,
                    align_method="v2", n_fold=p.get("n_fold"), device=a.device, do_map=False, do_zones=False,
                    do_histogram=False, do_section=False, pdf=False, keep_arrays=True)
        A = rep.pop("_arrays")
        insp = ~(A["excluded"] & ~A["beyond_cap"])
        r = blade_fit(A["dv"], A["points"], A["normals_solid"], A["sector"], A["n_fold"], A["axis"], insp,
                      A["beyond_cap"])
        r["alignment"] = {k: rep["alignment"].get(k) for k in ("iterations", "converged", "bound_hit", "refinement",
                                                               "direct_check")}
        res[p["part"]] = r
        print(p["part"], json.dumps({k: r[k] for k in ("blade_rms", "blade_rms_gated_points", "common_fit",
                                                     "per_blade_spread_std")}), flush=True)
        for k in ("aligned_v2", "after_common_fit", "after_per_blade_fit"):
            w = r["whole_part"][k]
            print("   whole", k, {q: w[q] for q in ("rms_uncapped", "rms_capped", "p95", "pass_pct_v1", "pass_pct_legacy")},
                  flush=True)
    (out / "per_blade.json").write_text(json.dumps(res, indent=2, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
