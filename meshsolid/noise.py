"""Estimate scan noise (sigma) from local quadric fits, to derive the fit tolerance automatically."""
import numpy as np
from scipy.spatial import cKDTree


def estimate_noise(mesh, n_patches=600, k=24, seed=0, pct=10, return_exact=False):
    """10th percentile of local quadric-fit rms, scaled by 1.2 (chi-dist correction for ~18 dof):
    robust to patches straddling sharp edges, which inflate the upper percentiles."""
    V = mesh.vertices
    if len(V) < k + 1:                   # a handful of vertices: only an exact (CAD) mesh is that coarse
        return (0.0, True) if return_exact else 0.0
    tree = cKDTree(V)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(V), min(n_patches, len(V)), replace=False)
    _, nb = tree.query(V[idx], k=k)
    res = []
    for ids in nb:
        P = V[ids] - V[ids].mean(0)
        _, _, vt = np.linalg.svd(P, full_matrices=False)
        x, y, z = P @ vt[0], P @ vt[1], P @ vt[2]
        A = np.c_[x * x, x * y, y * y, x, y, np.ones_like(x)]
        c, *_ = np.linalg.lstsq(A, z, rcond=None)
        r = z - A @ c
        res.append(np.sqrt(np.mean(r * r) * k / (k - 6)))
    if return_exact:
        p5, p10 = np.percentile(res, [5, pct])
        # an exact (CAD-tessellated) mesh has a population of patches with ~zero residual;
        # a scan has noise in every patch, so its low percentiles stay close together
        return float(1.2 * p10), bool(p5 < 0.1 * p10)
    return float(1.2 * np.percentile(res, pct))


def auto_tol_detail(mesh, sigma=None):
    """Fit-tolerance breakdown.

    ``tol`` / ``sigma`` / ``exact`` match ``auto_tol`` (an exact mesh still returns sigma 0 and the
    CAD floor). ``measured_sigma`` is the quadric estimate before that zeroing, and ``scan_tol`` is
    ``max(4.5·measured_sigma, 5e-5·diag, 0.005)`` — the tolerance a declared scan keeps when the
    exact-mesh floor would tighten the gate.
    """
    exact = False
    if sigma is None:
        measured, exact = estimate_noise(mesh, return_exact=True)
    else:
        measured = float(sigma)
    diag = float(np.linalg.norm(np.ptp(mesh.vertices, axis=0)))
    scan_tol = max(4.5 * float(measured), 5e-5 * diag, 0.005)
    if exact:
        tol, reported = max(5e-5 * diag, 0.002), 0.0
    else:
        tol, reported = scan_tol, float(measured)
    return {"tol": float(tol), "sigma": float(reported), "exact": bool(exact),
            "measured_sigma": float(measured), "scan_tol": float(scan_tol), "diag": float(diag)}


def auto_tol(mesh, sigma=None):
    """Returns (tol, sigma, exact_mesh)."""
    detail = auto_tol_detail(mesh, sigma)
    return detail["tol"], detail["sigma"], detail["exact"]
