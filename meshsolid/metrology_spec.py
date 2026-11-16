"""Measurement spec v1: the frozen definition every harness row is measured with.

Any change to a value in SPEC (or to the metric definitions in `metrics`) MUST bump SPEC_VERSION, so a
change in rms between two harness rows can never come from a quietly changed exclusion, cap, alignment
or counting rule. The human-readable version is docs/measurement_spec_v1.md.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

SPEC_VERSION = "v1"

SPEC = {
    "spec_version": SPEC_VERSION,
    "units": "mm",
    "distance": {
        "direction": "one-sided: every scan vertex -> nearest point on the solid's tessellation",
        "sign": "positive = scan outside the solid (normal of the closest solid triangle)",
        "solid_tessellation": {"linear_deflection_mm": 0.01, "angular_deflection_rad": 0.1},
        "population": "scan vertices (not area-weighted samples)",
    },
    "alignment": {
        "method": "metrology.align_scan (1.0.1): trimmed point-to-point ICP, 4 proper PCA starts, "
                  "25,000 area-uniform samples on scan and solid, seed 0, 40 iterations, 90 % trim, "
                  "best start by median residual",
        "samples": 25000, "seed": 0, "iterations": 40, "trim_pct": 90,
        "default": "on (align=True), the pose is logged as rotation angle (deg) + translation (mm)",
        "note": "frozen as-is for v1, even though it is known to land on another symmetric copy "
                "and stop ~0.05-0.1 deg short; the alignment fix will be spec v2",
    },
    "exclusions": {
        "rule": "only named, listed exclusions: (1) hand-entered features from the conversion report "
                "(1.0.1 auto-exclude), (2) per-part exclusions declared in the harness manifest with a "
                "reason. No zone tags by default (no 'tip', 'backface' or 'bore' zones): they hid real "
                "defects in 1.0.1 (the 'tip' gear zone removed an impeller's rim and blade tips).",
        "default_zone_tags": [],
        "pct_excluded_basis": "excluded vertices / all scan vertices x 100, beyond-cap points NOT counted as excluded",
    },
    "outlier_cap": {
        "cap_mm": 0.2,
        "treatment": "the cap never removes points from the v1 numbers. Points with |d| > cap are counted as "
                     "FAILURES in pass_pct_v1 and included in rms_uncapped / p95 / p99 / max. The cap only "
                     "defines the legacy (1.0.1) numbers rms_capped and pass_pct_legacy, and capped_count.",
    },
    "pass": {
        "tol_mm": 0.05,
        "pass_pct_v1": "100 x #(inspected, not beyond cap, |d| <= tol) / #(inspected), inspected = scan "
                       "vertices not excluded by a named reason, INCLUDING beyond-cap points",
        "pass_pct_legacy": "100 x #(inspected, not beyond cap, |d| <= tol) / #(inspected, not beyond cap) "
                           "(1.0.1 counting: capped points dropped from numerator and denominator)",
        "target": {"rms_mm": 0.03, "pass_pct_v1": 90.0, "max_pct_excluded": 5.0},
        "accept_pct_display": 95.0,
    },
}


SPEC_V1 = SPEC

# ---- spec v2 (2026-11-16): ONLY the alignment changes; distance, exclusions, cap and pass definitions are v1's
import copy as _copy  # noqa: E402

SPEC_V2 = _copy.deepcopy(SPEC_V1)
SPEC_V2["spec_version"] = "v2"
SPEC_V2["alignment"] = {
    "method": "metrology align_method='v2' (meshsolid/align_v2.py): point-to-plane ICP on >= 500,000 points "
              "(scan vertices topped up with area-uniform scan samples, seed 0), 5 % trim of the largest |residual| "
              "and solid points within 0.25 mm of a sharp (>20 deg) edge left out, iterated until a step moves "
              "< 0.0005 deg and < 0.001 mm (max 100). Start: identity (MeshSolid builds the solid in the scan "
              "frame) or a given transform. N-fold parts: rotation about the part axis bounded to +/- half a pitch "
              "from the start. Then a rigid refinement on the signed deviation of the inspected vertices (named "
              "exclusions removed, robust 6-sigma MAD gate, same edge band) and a direct re-measurement at +/- "
              "0.005 deg / 0.005 mm in all 6 degrees of freedom",
    "min_points": 500000, "icp_trim_pct": 5.0, "edge_band_mm": 0.25, "sharp_deg": 20.0,
    "converge_deg": 0.0005, "converge_mm": 0.001, "refine_gate_sigma": 6.0,
    "warn_refine_deg": 0.01, "warn_refine_mm": 0.01, "seed": 0,
    "n_fold_source": "manifest n_fold, else the conversion report (impeller blades / gear teeth), else estimated",
    "default": "on (align=True), the pose is logged as rotation angle (deg) + translation (mm)",
    "note": "a once-per-revolution per-blade pattern is indistinguishable from a rigid pose error and is absorbed "
            "by the pose by design; only the remaining blade-to-blade scatter is left in the deviation",
}

SPECS = {"v1": SPEC_V1, "v2": SPEC_V2}


def spec_hash(spec=None):
    """Short hash of the spec dict (logged in every row so an edited-but-not-bumped spec shows up)."""
    blob = json.dumps(spec or SPEC, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def _q(a, p):
    return float(np.percentile(a, p)) if a.size else float("nan")


def metrics(dv, excluded, beyond_cap, tol=None):
    """Spec-v1 numbers from per-vertex signed deviations.

    dv: signed deviation per scan vertex (mm). excluded: bool mask of vertices removed by a NAMED reason
    (not the cap). beyond_cap: bool mask of |d| > cap among the inspected. Returns a flat dict."""
    tol = float(SPEC["pass"]["tol_mm"] if tol is None else tol)
    dv = np.asarray(dv, float)
    excluded = np.asarray(excluded, bool)
    beyond_cap = np.asarray(beyond_cap, bool) & ~excluded
    insp = ~excluded
    d = dv[insp]
    a = np.abs(d)
    within = insp & ~beyond_cap & (np.abs(dv) <= tol)
    n_scan, n_insp = int(dv.size), int(insp.sum())
    n_cap = int(beyond_cap.sum())
    kept = insp & ~beyond_cap
    dk = dv[kept]
    out = {
        "n_scan": n_scan, "n_inspected": n_insp, "n_excluded": int(excluded.sum()),
        "pct_excluded": round(100.0 * float(excluded.sum()) / max(n_scan, 1), 4),
        "capped_count": n_cap, "pct_capped": round(100.0 * n_cap / max(n_insp, 1), 4),
        "rms_uncapped": round(float(np.sqrt(np.mean(d * d))), 5) if d.size else None,
        "mean_signed": round(float(np.mean(d)), 5) if d.size else None,
        "p95": round(_q(a, 95), 5) if d.size else None,
        "p99": round(_q(a, 99), 5) if d.size else None,
        "max_plus": round(float(d.max()), 5) if d.size else None,
        "max_minus": round(float(d.min()), 5) if d.size else None,
        "rms_capped": round(float(np.sqrt(np.mean(dk * dk))), 5) if dk.size else None,
        "p95_capped": round(_q(np.abs(dk), 95), 5) if dk.size else None,
        "pass_pct_v1": round(100.0 * float(within.sum()) / max(n_insp, 1), 4),
        "pass_pct_legacy": round(100.0 * float(within.sum()) / max(int(kept.sum()), 1), 4),
    }
    t = SPEC["pass"]["target"]
    out["target_met_v1"] = bool(out["rms_uncapped"] is not None and out["rms_uncapped"] <= t["rms_mm"]
                                and out["pass_pct_v1"] >= t["pass_pct_v1"]
                                and out["pct_excluded"] <= t["max_pct_excluded"])
    return out


def verdict(m, legacy_within_pct=None, accept_pct=95.0, spec=None):
    """Honest verdict from spec-v1/v2 metrics `m` (see metrics), next to the 1.0.1 engine-limit result.

    target:       PASS only when rms_uncapped <= 0.03 mm AND pass_pct_v1 >= 90 % (beyond-cap points count as
                  failures) AND pct_excluded <= 5 %.
    engine_limit: the old rule, >= accept_pct % of the inspected vertices within the band with beyond-cap points
                  dropped (legacy_within_pct). A PASS is REFUSED when more than 5 % of the scan was excluded.
    Every PASS label carries the exclusion %, so a PASS that rests on excluded data is visible."""
    spec = spec or SPEC
    t = spec["pass"]["target"]
    tol = spec["pass"]["tol_mm"]
    ex = float(m.get("pct_excluded") or 0.0)
    rms = m.get("rms_uncapped")
    pv1 = m.get("pass_pct_v1")
    checks = [
        {"name": "rms_uncapped", "value": rms, "limit": t["rms_mm"], "op": "<=",
         "ok": rms is not None and rms <= t["rms_mm"], "text": f"rms (uncapped) <= {t['rms_mm']:g} mm"},
        {"name": "pass_pct_v1", "value": pv1, "limit": t["pass_pct_v1"], "op": ">=",
         "ok": pv1 is not None and pv1 >= t["pass_pct_v1"],
         "text": f">= {t['pass_pct_v1']:g} % within ±{tol:g} mm (beyond-cap points count as failures)"},
        {"name": "pct_excluded", "value": ex, "limit": t["max_pct_excluded"], "op": "<=",
         "ok": ex <= t["max_pct_excluded"], "text": f"<= {t['max_pct_excluded']:g} % of the scan excluded"},
    ]
    ok = all(c["ok"] for c in checks)
    failed = [c["name"] for c in checks if not c["ok"]]
    tgt = {"result": "PASS" if ok else "FAIL", "pct_excluded": round(ex, 3), "checks": checks, "failed": failed,
           "rule": f"rms <= {t['rms_mm']:g} mm, >= {t['pass_pct_v1']:g} % within ±{tol:g} mm with beyond-cap points "
                   f"as failures, <= {t['max_pct_excluded']:g} % excluded",
           "spec_version": spec["spec_version"]}
    tgt["label"] = f"{tgt['result']} · {ex:.1f} % excluded"
    out = {"target": tgt}
    if legacy_within_pct is not None:
        lp = float(legacy_within_pct) >= float(accept_pct)
        refused = lp and ex > t["max_pct_excluded"]
        res = "REFUSED" if refused else ("PASS" if lp else "FAIL")
        eng = {"result": res, "within_pct": round(float(legacy_within_pct), 3), "accept_pct": float(accept_pct),
               "pct_excluded": round(ex, 3),
               "rule": f">= {float(accept_pct):g} % of inspected vertices within ±{tol:g} mm, beyond-cap points dropped "
                       "(1.0.1 engine-limit rule)"}
        eng["label"] = (f"PASS refused · {ex:.1f} % excluded > {t['max_pct_excluded']:g} %" if refused
                        else f"{res} · {ex:.1f} % excluded")
        out["engine_limit"] = eng
    return out


def engine_verdict(dv, excluded_vertices, accepted, gate_basis="", spec=None):
    """Conversion report: the honest target verdict on the scan-vertex -> solid deviations next to the engine's
    own acceptance gate (engine-limit result). excluded_vertices: vertices the gate left out (threads, scan-only
    stubs, balance cuts, ...): they are named exclusions, so their % is shown and a PASS above 5 % is refused."""
    spec = spec or SPEC
    dv = np.asarray(dv, float)
    ex = np.zeros(len(dv), bool) if excluded_vertices is None else np.asarray(excluded_vertices, bool)
    cap = (~ex) & (np.abs(dv) > spec["outlier_cap"]["cap_mm"])
    m = metrics(dv, ex, cap)
    out = verdict(m, spec=spec)
    exp = m["pct_excluded"]
    lim = spec["pass"]["target"]["max_pct_excluded"]
    refused = bool(accepted) and exp > lim
    res = "REFUSED" if refused else ("PASS" if accepted else "FAIL")
    out["engine_limit"] = {"result": res, "pct_excluded": round(exp, 3), "rule": gate_basis or "engine acceptance gate",
                           "label": (f"PASS refused · {exp:.1f} % excluded > {lim:g} %" if refused
                                     else f"{res} · {exp:.1f} % excluded")}
    out["metrics"] = {k: m[k] for k in ("rms_uncapped", "pass_pct_v1", "pct_excluded", "p95", "p99", "rms_capped",
                                        "capped_count", "n_scan")}
    return out
