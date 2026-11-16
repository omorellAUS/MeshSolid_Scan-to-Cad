"""Least-squares primitive fitting: plane, cylinder, sphere, cone."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def _perp_basis(a):
    a = _unit(a)
    t = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
    u = _unit(np.cross(a, t))
    return u, np.cross(a, u)


def _dir_from_angles(th, ph):
    return np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])


def _angles_from_dir(d):
    d = _unit(d)
    return np.arccos(np.clip(d[2], -1, 1)), np.arctan2(d[1], d[0])


@dataclass
class Primitive:
    kind: str                      # plane | cylinder | sphere | cone
    params: dict = field(default_factory=dict)
    rms: float = np.inf
    max_err: float = np.inf
    normal_err_deg: float = np.inf  # RMS angle between mesh normals and surface normals

    def distance(self, P):
        return signed_distance(self, P)

    def to_json(self):
        return {"kind": self.kind,
                "params": {k: (np.asarray(v).tolist() if isinstance(v, np.ndarray) else float(v))
                           for k, v in self.params.items()},
                "rms": float(self.rms), "max_err": float(self.max_err),
                "normal_err_deg": float(self.normal_err_deg)}


def signed_distance(p: Primitive, P):
    P = np.atleast_2d(P)
    k, q = p.kind, p.params
    if k == "plane":
        return (P - q["point"]) @ q["normal"]
    if k == "cylinder":
        d = P - q["point"]
        r = d - np.outer(d @ q["axis"], q["axis"])
        return np.linalg.norm(r, axis=1) - q["radius"]
    if k == "sphere":
        return np.linalg.norm(P - q["center"], axis=1) - q["radius"]
    if k == "cone":
        d = P - q["apex"]
        h = d @ q["axis"]
        r = np.linalg.norm(d - np.outer(h, q["axis"]), axis=1)
        a = q["half_angle"]
        return r * np.cos(a) - h * np.sin(a)
    if k == "torus":
        return _torus_dist(P, q["center"], q["axis"], q["major_radius"], q["minor_radius"])
    if k == "pipe":
        from .blend import pipe_distance
        return pipe_distance(q, P)[0]
    if k == "bspline":
        from .freeform import sampler_of
        return sampler_of(p).distance(P)
    raise ValueError(k)


def surface_normal(p: Primitive, P):
    """Unit normal of the primitive's surface at (the projection of) points P (unsigned direction)."""
    P = np.atleast_2d(P)
    k, q = p.kind, p.params
    if k == "plane":
        return np.tile(q["normal"], (len(P), 1))
    if k == "cylinder":
        d = P - q["point"]
        r = d - np.outer(d @ q["axis"], q["axis"])
        return r / np.maximum(np.linalg.norm(r, axis=1, keepdims=True), 1e-12)
    if k == "sphere":
        d = P - q["center"]
        return d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
    if k == "cone":
        d = P - q["apex"]
        h = d @ q["axis"]
        r = d - np.outer(h, q["axis"])
        rn = r / np.maximum(np.linalg.norm(r, axis=1, keepdims=True), 1e-12)
        a = q["half_angle"]
        n = rn * np.cos(a) - np.outer(np.ones(len(P)), q["axis"]) * np.sin(a)
        return n
    if k == "torus":
        h = P - q["center"]
        ha = h @ q["axis"]
        rv = h - np.outer(ha, q["axis"])
        rn = rv / np.maximum(np.linalg.norm(rv, axis=1, keepdims=True), 1e-12)
        spine = q["center"] + rn * q["major_radius"]
        d = P - spine
        return d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
    if k == "pipe":
        from .blend import pipe_normal
        return pipe_normal(q, P)
    if k == "bspline":
        from .freeform import sampler_of
        return sampler_of(p).normal(P)
    raise ValueError(k)


def _finish(p: Primitive, P, N, W=None):
    d = signed_distance(p, P)
    w = np.ones(len(P)) if W is None else W / W.sum() * len(W)
    p.rms = float(np.sqrt(np.sum(w * d * d) / len(d)))
    p.max_err = float(np.max(np.abs(d)))
    if N is not None:
        sn = surface_normal(p, P)
        c = np.abs(np.sum(sn * N, axis=1)).clip(0, 1)
        p.normal_err_deg = float(np.degrees(np.sqrt(np.mean(np.arccos(c) ** 2))))
    return p


def fit_plane(P, N=None, W=None):
    w = np.ones(len(P)) if W is None else W
    c = (P * w[:, None]).sum(0) / w.sum()
    X = (P - c) * np.sqrt(w)[:, None]
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    n = vt[2]
    if N is not None and np.sum(N @ n) < 0:
        n = -n
    return _finish(Primitive("plane", {"point": c, "normal": n}), P, N, W)


def _circle_fit_2d(xy):
    A = np.c_[2 * xy, np.ones(len(xy))]
    b = (xy ** 2).sum(1)
    s, *_ = np.linalg.lstsq(A, b, rcond=None)
    c = s[:2]
    r = np.sqrt(max(s[2] + c @ c, 1e-12))
    return c, r


def fit_cylinder(P, N=None, W=None, axis0=None):
    if axis0 is None:
        if N is None:
            raise ValueError("cylinder init needs normals or axis0")
        C = N.T @ (N * (1 if W is None else W[:, None]))
        axis0 = np.linalg.eigh(C)[1][:, 0]
    axis0 = _unit(axis0)
    u, v = _perp_basis(axis0)
    c0 = P.mean(0)
    xy = np.c_[(P - c0) @ u, (P - c0) @ v]
    cc, r0 = _circle_fit_2d(xy)
    pt0 = c0 + cc[0] * u + cc[1] * v
    th0, ph0 = _angles_from_dir(axis0)
    sw = None if W is None else np.sqrt(W)

    def res(x):
        a = _dir_from_angles(x[0], x[1])
        uu, vv = _perp_basis(axis0)
        pt = pt0 + x[2] * uu + x[3] * vv
        d = P - pt
        rr = np.linalg.norm(d - np.outer(d @ a, a), axis=1) - x[4]
        return rr if sw is None else rr * sw

    sol = least_squares(res, [th0, ph0, 0, 0, r0], method="lm", max_nfev=200)
    a = _dir_from_angles(sol.x[0], sol.x[1])
    pt = pt0 + sol.x[2] * u + sol.x[3] * v
    # canonical point: closest to the data centroid along axis
    pt = pt + ((c0 - pt) @ a) * a
    return _finish(Primitive("cylinder", {"point": pt, "axis": a, "radius": abs(sol.x[4])}), P, N, W)


def fit_sphere(P, N=None, W=None):
    A = np.c_[2 * P, np.ones(len(P))]
    b = (P ** 2).sum(1)
    s, *_ = np.linalg.lstsq(A, b, rcond=None)
    c = s[:3]
    r = np.sqrt(max(s[3] + c @ c, 1e-12))
    sw = None if W is None else np.sqrt(W)

    def res(x):
        rr = np.linalg.norm(P - x[:3], axis=1) - x[3]
        return rr if sw is None else rr * sw

    sol = least_squares(res, [*c, r], method="lm", max_nfev=200)
    return _finish(Primitive("sphere", {"center": sol.x[:3], "radius": abs(sol.x[3])}), P, N, W)


def fit_cone(P, N, W=None):
    """Cone init from normals: normals of a cone make constant angle with the axis."""
    # axis: direction a minimizing var(N @ a) -> smallest eigenvector of covariance of N
    Nc = N - N.mean(0)
    a0 = np.linalg.eigh(Nc.T @ Nc)[1][:, 0]
    if np.mean(N @ a0) > 0:
        a0 = -a0
    sa = float(np.clip(-np.mean(N @ a0), -0.99, 0.99))  # sin(half_angle) with outward normals
    half0 = np.arcsin(sa)
    if abs(half0) < np.radians(2):
        return None
    # apex: least squares point such that (p - apex) . n = 0 for all (tangent planes pass through apex)
    apex0, *_ = np.linalg.lstsq(N, (P * N).sum(1), rcond=None)
    th0, ph0 = _angles_from_dir(a0)
    sw = None if W is None else np.sqrt(W)

    def res(x):
        a = _dir_from_angles(x[0], x[1])
        d = P - x[2:5]
        h = d @ a
        r = np.linalg.norm(d - np.outer(h, a), axis=1)
        rr = r * np.cos(x[5]) - h * np.sin(x[5])
        return rr if sw is None else rr * sw

    sol = least_squares(res, [th0, ph0, *apex0, half0], method="lm", max_nfev=400)
    a = _dir_from_angles(sol.x[0], sol.x[1])
    # the residual r*cos(t) - h*sin(t) has the same zero set for t + k*pi, and for (-t, -axis):
    # wrap to (-pi/2, pi/2], then flip the axis for a negative angle
    half = float(sol.x[5])
    half = half - np.pi * np.round(half / np.pi)
    if half < 0:
        a, half = -a, -half
    return _finish(Primitive("cone", {"apex": sol.x[2:5], "axis": a, "half_angle": half}), P, N, W)


def fit_best(P, N, W=None, tol=0.05, kinds=("plane", "cylinder", "sphere", "cone")):
    """Fit all candidate kinds, choose the simplest that is within tolerance (else lowest rms)."""
    cands = []
    for k in kinds:
        try:
            if k == "plane":
                p = fit_plane(P, N, W)
            elif k == "cylinder":
                p = fit_cylinder(P, N, W)
            elif k == "sphere":
                p = fit_sphere(P, N, W)
            else:
                p = fit_cone(P, N, W)
        except Exception:
            p = None
        if p is not None and np.isfinite(p.rms):
            if p.kind == "cylinder" and p.params["radius"] > 1e4:
                continue
            if p.kind == "sphere" and p.params["radius"] > 1e4:
                continue
            cands.append(p)
    if not cands:
        return None, []
    ok = [c for c in cands if c.rms <= tol and c.normal_err_deg < 10]
    if ok:
        return ok[0], cands   # kinds are ordered simplest-first
    return min(cands, key=lambda c: c.rms), cands


# ---------------------------------------------------------------- torus
def _torus_dist(P, c, a, R, r):
    h = P - c
    ha = h @ a
    rho = np.linalg.norm(h - np.outer(ha, a), axis=1)
    return np.sqrt((rho - R) ** 2 + ha ** 2) - r


def _circle3d(Q, W=None):
    c0 = Q.mean(0)
    _, s, vt = np.linalg.svd(Q - c0, full_matrices=False)
    a = vt[2]
    u, v = vt[0], vt[1]
    xy = np.c_[(Q - c0) @ u, (Q - c0) @ v]
    cc, R = _circle_fit_2d(xy)
    res = np.linalg.norm(xy - cc, axis=1) - R
    off = (Q - c0) @ a
    err = np.sqrt(np.mean(res ** 2) + np.mean(off ** 2))
    return c0 + cc[0] * u + cc[1] * v, a, R, err


def fit_torus(P, N, W=None, axis0=None, init=None):
    """Torus: the tube-centre points q = p - s*n lie on a circle (the spine).
    1-D search over signed tube radius s, then LM refinement of (centre, axis, R, r).
    `init` (a torus Primitive) skips the search (warm start while region growing)."""
    ext = float(np.linalg.norm(np.ptp(P, axis=0)))
    if ext <= 0 or len(P) < 10:
        return None
    if init is not None:
        q = init.params
        th0, ph0 = _angles_from_dir(q["axis"])
        x0 = [*q["center"], th0, ph0, q["major_radius"], q["minor_radius"]]
        sol = least_squares(lambda x: _torus_dist(P, x[:3], _dir_from_angles(x[3], x[4]), x[5], x[6]),
                            x0, method="lm", max_nfev=60)
        x = sol.x
        if x[5] <= 0 or x[6] <= 0:
            return None
        return _finish(Primitive("torus", {"center": x[:3], "axis": _dir_from_angles(x[3], x[4]),
                                           "major_radius": x[5], "minor_radius": abs(x[6])}), P, N, W)
    best = None
    grid = ext * np.geomspace(0.01, 1.5, 32)
    for s in np.r_[grid, -grid]:
        c, a, R, err = _circle3d(P - s * N)
        if best is None or err < best[0]:
            best = (err, s, c, a, R)
    err, s, c, a, R = best
    # local 1-D refinement around best s
    lo, hi = s * 0.8, s * 1.25
    for _ in range(18):
        m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
        e1 = _circle3d(P - m1 * N)[3]
        e2 = _circle3d(P - m2 * N)[3]
        if e1 < e2:
            hi = m2
        else:
            lo = m1
    s = (lo + hi) / 2
    c, a, R, err = _circle3d(P - s * N)
    if axis0 is not None:
        a = _unit(axis0) * np.sign(_unit(axis0) @ a or 1)
    th0, ph0 = _angles_from_dir(a)
    sw = None if W is None else np.sqrt(W)

    def res(x):
        rr = _torus_dist(P, x[:3], _dir_from_angles(x[3], x[4]), x[5], x[6])
        return rr if sw is None else rr * sw

    sol = least_squares(res, [*c, th0, ph0, R, abs(s)], method="lm", max_nfev=150)
    x = sol.x
    a = _dir_from_angles(x[3], x[4])
    if x[5] <= 0 or x[6] <= 0:
        return None
    p = Primitive("torus", {"center": x[:3], "axis": a, "major_radius": x[5], "minor_radius": abs(x[6])})
    return _finish(p, P, N, W)
