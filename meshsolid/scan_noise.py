"""Scan noise estimate from small patches (no rescan needed).

Each ~10 x 10 mm patch (disc of equal area, r = 5.64 mm) gets its own local fit, so form error over the part is
not counted as noise: a plane and a degree-4 polynomial height field in the patch's PCA frame (it also removes
the curvature of cylinders and blades). The noise figure ("rms_quadric", kept as the column name) is that
polynomial's residual rms, corrected for the fitted degrees of freedom. Patches that straddle an edge / feature
are rejected (heavy residual tails, or rms > 3 x the median patch). A whiteness index (~1 = white noise, << 1 =
spatially smooth residual) flags scanner smoothing.

This is an ESTIMATE biased both ways: scanner-side smoothing or decimation hides noise (too low), residual
micro-form within a patch adds to it (too high). Only a repositioned rescan measures the real scan-to-scan
repeatability, including registration.
"""
from __future__ import annotations

import numpy as np

PATCH_R = 5.64           # mm: disc with the area of a 10 x 10 mm square


def _frame(P):
    c = P.mean(0)
    w, V = np.linalg.eigh(np.cov((P - c).T))
    return c, V[:, 2], V[:, 1], V[:, 0]           # centre, e1, e2 (in-plane), normal (smallest variance)


def _poly(x, y, deg):
    return np.stack([x ** i * y ** (d - i) for d in range(deg + 1) for i in range(d + 1)], 1)


def fit_patch(P, deg=4):
    """(plane rms, surface rms, curvature 1/mm, whiteness, ok, why) for the points P of one patch.

    surface rms: residual of a degree-`deg` polynomial height field in the patch's PCA frame, corrected for the
    fitted degrees of freedom (x sqrt(n / (n - p))); a quadric leaves ~0.007 mm of form on an r = 11 mm journal,
    a quartic does not. whiteness: rms(r_i - r_nearest) / (sqrt 2 rms): ~1 for white noise, well below 1 when the
    residual is spatially smooth (scanner smoothing, or form the fit could not follow)."""
    from scipy.spatial import cKDTree
    c, e1, e2, n = _frame(P)
    Q = P - c
    x, y, z = Q @ e1, Q @ e2, Q @ n
    rp = z - z.mean()
    A2 = _poly(x, y, 2)
    c2, *_ = np.linalg.lstsq(A2, z, rcond=None)
    k = float(max(abs(2 * c2[3]), abs(2 * c2[5]))) if len(c2) == 6 else 0.0   # x^2, y^2 terms
    A = _poly(x, y, deg)
    coef, *_ = np.linalg.lstsq(A, z, rcond=None)
    rq = z - A @ coef
    dof = max(len(z) - A.shape[1], 1)
    rms_q = float(np.sqrt(np.sum(rq ** 2) / dof))
    _d, j = cKDTree(np.stack([x, y], 1)).query(np.stack([x, y], 1), k=2)
    white = float(np.sqrt(np.mean((rq - rq[j[:, 1]]) ** 2)) / (np.sqrt(2) * np.sqrt(np.mean(rq ** 2)) + 1e-15))
    mad = 1.4826 * float(np.median(np.abs(rq - np.median(rq)))) + 1e-12
    ok, why = True, ""
    if float(np.mean(np.abs(rq) > 5 * mad)) > 0.02:
        ok, why = False, "heavy residual tails (step / feature)"
    return float(np.sqrt(np.mean(rp ** 2))), rms_q, k, white, ok, why


def patch_centres(V, spacing=10.0):
    """One scan vertex per spacing^3 voxel (the vertex nearest the voxel's centroid)."""
    key = np.floor(V / spacing).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    order = np.argsort(inv)
    bounds = np.flatnonzero(np.diff(inv[order])) + 1
    out = []
    for grp in np.split(order, bounds):
        g = V[grp]
        out.append(grp[np.argmin(np.linalg.norm(g - g.mean(0), axis=1))])
    return np.asarray(out, int)


def estimate(scan, region_mask=None, spacing=10.0, radius=PATCH_R, min_pts=40, flat_k=0.004):
    """Patch-noise estimate over the scan (or over region_mask). Returns a dict with per-patch rows and the
    median / p90 of the quadric rms for all accepted patches, planar ones (curvature < flat_k 1/mm) and curved ones."""
    from scipy.spatial import cKDTree
    V = np.asarray(scan.vertices, float)
    N = np.asarray(scan.vertex_normals, float)
    idx_all = np.arange(len(V)) if region_mask is None else np.flatnonzero(region_mask)
    tree = cKDTree(V[idx_all])
    cent = idx_all[patch_centres(V[idx_all], spacing)]
    rows = []
    for ci in cent:
        nb = idx_all[tree.query_ball_point(V[ci], radius)]
        if len(nb) < min_pts:
            continue
        # keep the sheet the centre is on (a thin blade's far side is within the radius too)
        same = (N[nb] @ N[ci]) > 0.0
        nb = nb[same]
        if len(nb) < min_pts:
            continue
        rp, rq, k, wh, ok, why = fit_patch(V[nb])
        rows.append({"centre": V[ci].round(3).tolist(), "n": int(len(nb)), "rms_plane": round(rp, 6),
                     "rms_quadric": round(rq, 6), "curvature": round(k, 5), "whiteness": round(wh, 3),
                     "ok": ok, "why": why})
    # patches straddling an edge / step / feature: far above the typical patch
    if rows:
        med = float(np.median([r["rms_quadric"] for r in rows if r["ok"]] or [r["rms_quadric"] for r in rows]))
        for r in rows:
            if r["ok"] and r["rms_quadric"] > 3.0 * med:
                r["ok"], r["why"] = False, "rms > 3 x the median patch (edge / step / feature)"

    def summ(sel):
        q = np.array([r["rms_quadric"] for r in sel])
        p = np.array([r["rms_plane"] for r in sel])
        if not len(q):
            return None
        return {"patches": int(len(q)), "median_rms_quadric": round(float(np.median(q)), 5),
                "p90_rms_quadric": round(float(np.percentile(q, 90)), 5),
                "p10_rms_quadric": round(float(np.percentile(q, 10)), 5),
                "median_rms_plane": round(float(np.median(p)), 5),
                "median_points_per_patch": int(np.median([r["n"] for r in sel])),
                "median_whiteness": round(float(np.median([r["whiteness"] for r in sel])), 3)}
    acc = [r for r in rows if r["ok"]]
    el = scan.edges_unique_length
    return {"patch_radius_mm": radius, "spacing_mm": spacing, "n_candidates": len(rows), "n_rejected": len(rows) - len(acc),
            "median_edge_mm": round(float(np.median(el)), 4),
            "all": summ(acc), "planar": summ([r for r in acc if r["curvature"] < flat_k]),
            "curved": summ([r for r in acc if r["curvature"] >= flat_k]), "rows": rows}


def end_face_mask(scan, axis, end="low", cos_min=0.98, band_mm=0.5):
    """Scan vertices on the flat end face at the low / high axial end (normal along the axis, within band_mm
    of the extreme plane cluster)."""
    c, u = axis
    V = np.asarray(scan.vertices, float)
    N = np.asarray(scan.vertex_normals, float)
    z = (V - c) @ u
    nz = N @ u
    cand = np.abs(nz) > cos_min
    if not cand.any():
        return cand
    zc = z[cand]
    # the most populated axial plane among the candidates near that end (histogram at 0.1 mm)
    lo, hi = float(zc.min()), float(zc.max())
    h, e = np.histogram(zc, bins=max(int((hi - lo) / 0.1), 1))
    third = max(len(h) // 3, 1)
    if end == "low":
        j = int(np.argmax(h[:third]))
    else:
        j = len(h) - third + int(np.argmax(h[-third:]))
    z0 = 0.5 * (e[j] + e[j + 1])
    return cand & (np.abs(z - z0) < band_mm)


def flatness(scan, mask):
    """Global plane fit of a face: rms and peak-to-valley of the residual (form error + noise)."""
    P = np.asarray(scan.vertices, float)[mask]
    if len(P) < 10:
        return None
    c, e1, e2, n = _frame(P)
    r = (P - c) @ n
    return {"n": int(len(P)), "rms": round(float(np.sqrt(np.mean(r ** 2))), 5),
            "pv": round(float(r.max() - r.min()), 5), "p99_abs": round(float(np.percentile(np.abs(r), 99)), 5)}
