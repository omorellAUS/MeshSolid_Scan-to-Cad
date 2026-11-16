"""Input-class detector (pre-recognition): scan vs CAD tessellation vs low-res CAD tessellation vs
decimated mesh, from the facet size / edge-length distribution, dihedral flatness and vertex valence.

Why: the quadric noise estimator reads facet size as noise on a coarse CAD tessellation (Cummins
wheels, 13.6k / 47k faces: "noise" 0.32-0.43 mm vs 0.002-0.008 mm on real scans).  Each class gets its
own noise / tolerance policy, and the choice is logged in report["input_class"].

  scan                    dense, regular (valence ~6), every neighbourhood noisy: unchanged policy
  decimated               a scan reduced by edge collapse: irregular valence and edge lengths, still noisy;
                          tolerance unchanged (the noise estimate is real), class logged
  cad_tessellation        exact vertices, flat/coplanar facet populations, fine: sag-based tolerance
                          (meshsolid.facets), unchanged
  lowres_tessellation     a CAD tessellation whose facets are coarse against the part (or few faces):
                          the effective noise is the chord sag, not the quadric residual, and the
                          acceptance gates are loosened to the facet size (user 2026-11-16: the Cummins
                          meshes are very rough; looser tolerances) unless an accuracy class or explicit
                          limits were given
"""
from __future__ import annotations

import numpy as np

LOWRES_FACET_FRAC = 0.004      # median facet width >= 0.4 % of the bounding diagonal ...
LOWRES_MAX_FACES = 100_000     # ... or a CAD tessellation with fewer faces than this
LOWRES_RMS_FRAC = 0.10         # gates on a low-res tessellation: rms <= 10 % of the median facet width
LOWRES_P95_FRAC = 0.25         #                                   p95 <= 25 %


def mesh_metrics(mesh):
    V = np.asarray(mesh.vertices, float); E = np.asarray(mesh.edges_unique)
    L = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1)
    diag = float(np.linalg.norm(np.ptp(V, axis=0)))
    val = np.bincount(E.ravel(), minlength=len(V))
    val = val[val > 0]
    med = float(np.median(L))
    return {"faces": int(len(mesh.faces)), "vertices": int(len(V)), "diag_mm": round(diag, 3),
            "edge_len_median_mm": round(med, 5), "edge_len_p10_mm": round(float(np.percentile(L, 10)), 5),
            "edge_len_p90_mm": round(float(np.percentile(L, 90)), 5),
            "edge_len_cv": round(float(np.std(L) / max(np.mean(L), 1e-12)), 3),
            "edge_len_median_over_diag": round(med / max(diag, 1e-12), 6),
            "valence6_frac": round(float(np.mean(val == 6)), 3),
            "valence_ge9_frac": round(float(np.mean(val >= 9)), 3)}


def classify(mesh, sigma=None, exact=None, fst=None):
    """Returns the class record (class, reason, metrics and the noise/tolerance policy applied)."""
    m = mesh_metrics(mesh)
    fst = fst or {}
    diag = m["diag_mm"]
    cad = bool(fst.get("cad_tessellation")) or bool(exact) or bool(fst.get("fine_exact_tessellation"))
    w = float(fst.get("facet_width_median") or 0.0)
    m.update({"noise_sigma_quadric_mm": None if sigma is None else round(float(sigma), 5),
              "exact_mesh_detected": exact, "coplanar_len_frac": fst.get("coplanar_len_frac"),
              "facet_width_median_mm": w or None, "sag_p90_mm": fst.get("sag_p90")})
    if cad:
        coarse = (w >= LOWRES_FACET_FRAC * diag) or (m["faces"] < LOWRES_MAX_FACES)
        if coarse and fst.get("cad_tessellation") and not fst.get("fine_exact_tessellation"):
            cls = "lowres_tessellation"
            reason = (f"CAD tessellation (exact/coplanar facets), coarse: facet width {w:.3f} mm = "
                      f"{100 * w / max(diag, 1e-9):.2f} % of the part, {m['faces']:,} faces")
        else:
            cls = "cad_tessellation"
            reason = "CAD tessellation (exact vertices / coplanar facet populations), fine"
    elif m["valence6_frac"] < 0.35 and m["edge_len_cv"] > 0.6:
        cls = "decimated"
        reason = (f"irregular triangulation (valence-6 {m['valence6_frac']:.2f}, edge-length cv "
                  f"{m['edge_len_cv']:.2f}) with noise in every neighbourhood")
    else:
        cls = "scan"
        reason = f"dense noisy triangulation (valence-6 {m['valence6_frac']:.2f}, edge cv {m['edge_len_cv']:.2f})"
    return {"class": cls, "reason": reason, "metrics": m}


def effective_noise(ic, sigma):
    """Noise to report/use: on a tessellation the quadric residual is facet size, not noise."""
    if ic["class"] in ("lowres_tessellation", "cad_tessellation"):
        s = ic["metrics"].get("sag_p90_mm")
        return (float(s) if s is not None else 0.0), "facet chord sag p90 (the quadric residual reads facet size as noise)"
    return (None if sigma is None else float(sigma)), "quadric residual (scan noise)"


def lowres_gates(ic, gates, tol):
    """Loosened acceptance limits for a low-res tessellation (only when the caller decided it applies)."""
    w = float(ic["metrics"].get("facet_width_median_mm") or 0.0)
    rms = max(gates["rms_limit"], LOWRES_RMS_FRAC * w); p95 = max(gates["p95_limit"], LOWRES_P95_FRAC * w)
    out = dict(gates)
    out.update({"class": "lowres_tessellation", "rms_limit": float(rms), "p95_limit": float(p95),
                "basis": (f"lowres_tessellation: vertex→solid rms≤max(1.5·tol, {LOWRES_RMS_FRAC:g}·facet width), "
                          f"p95≤max(3·tol, {LOWRES_P95_FRAC:g}·facet width); facet width {w:.3f} mm, tol {tol:.4f} "
                          "(input class; user 2026-11-16: very rough CAD meshes, looser tolerances)"),
                "strict_limits": {"rms_limit": gates["rms_limit"], "p95_limit": gates["p95_limit"]}})
    return out
