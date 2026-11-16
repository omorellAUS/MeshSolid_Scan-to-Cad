"""Spec-v2 alignment and N-fold sectors for Metrology.

align_v2: point-to-plane ICP on >= 500k scan points (vertices, topped up with area-uniform samples on the scan),
5 % trim of the largest |residual|, iterated until a step moves < 0.0005 deg and < 0.001 mm. For an N-fold part
it starts from the identity (MeshSolid builds the solid in the scan's frame) or from a given transform, and the
rotation about the part axis is bounded to half a pitch, so the pose can never jump to another blade/tooth copy.
refine_on_deviation: the final rigid refinement on the signed deviation of the inspected scan vertices.
direct_check: re-measures the trimmed rms at the final pose and at +/- small perturbations of every degree of
freedom, so the sign of the correction is verified by measurement, not by a fitted normal convention.

Every transform here maps the SCAN onto the solid (it is applied to a copy of the scan), like metrology.align_scan.
"""
from __future__ import annotations

import numpy as np

MIN_POINTS = 500_000
TRIM_PCT = 5.0
CONV_DEG = 0.0005
CONV_MM = 0.001
WARN_DEG = 0.01
WARN_MM = 0.01


EDGE_BAND_MM = 0.25
SHARP_DEG = 20.0


def sharp_edge_flags(solid_tm, sharp_deg=SHARP_DEG):
    """(F, 3) bool: local edge j (v_j -> v_j+1) of each solid triangle is a sharp or open edge."""
    F = np.asarray(solid_tm.faces, np.int64)
    nV = len(solid_tm.vertices)
    key = lambda a, b: np.minimum(a, b) * nV + np.maximum(a, b)   # noqa: E731
    ek = np.stack([key(F[:, j], F[:, (j + 1) % 3]) for j in range(3)], 1)
    adj_e = np.asarray(solid_tm.face_adjacency_edges, np.int64)
    ang = np.degrees(np.asarray(solid_tm.face_adjacency_angles, float))
    smooth = set(key(adj_e[ang <= sharp_deg, 0], adj_e[ang <= sharp_deg, 1]).tolist())
    flat = ek.ravel()
    return ~np.fromiter((x in smooth for x in flat), bool, len(flat)).reshape(F.shape)


def _seg_dist(P, A, B):
    AB = B - A
    t = np.clip(np.einsum("ij,ij->i", P - A, AB) / np.maximum(np.einsum("ij,ij->i", AB, AB), 1e-300), 0, 1)
    return np.linalg.norm(P - (A + t[:, None] * AB), axis=1)


def edge_band(solid_tm, cp, tri, band=EDGE_BAND_MM, flags=None):
    """True where the closest solid point lies within `band` mm of a sharp/open edge. Point-to-plane is
    ill-posed there (the closest face flips with the sign of a tiny motion, and scans round sharp edges)."""
    if flags is None:
        flags = sharp_edge_flags(solid_tm)
    V = np.asarray(solid_tm.vertices, float)
    F = np.asarray(solid_tm.faces, np.int64)[tri]
    fl = flags[tri]
    out = np.zeros(len(cp), bool)
    for j in range(3):
        m = fl[:, j]
        if m.any():
            out[m] |= _seg_dist(cp[m], V[F[m, j]], V[F[m, (j + 1) % 3]]) < band
    return out


def _closest(solid_tm, P):
    """Signed point-to-plane residual of P on the solid (positive = outside), closest points, normals, tri."""
    from .deviation import _on_surface
    cp, _d, tri = _on_surface(solid_tm, P)
    tri = np.asarray(tri, int)
    n = np.asarray(solid_tm.face_normals, float)[tri]
    r = np.einsum("ij,ij->i", np.asarray(P, float) - cp, n)
    return r, cp, n, tri


def rot_about(axis, ang):
    axis = np.asarray(axis, float)
    axis = axis / (np.linalg.norm(axis) + 1e-300)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)


def rigid(omega, t, c):
    """4x4 of the small motion p -> R(omega)(p - c) + c + t (omega = rotation vector, rad)."""
    a = float(np.linalg.norm(omega))
    R = rot_about(omega, a) if a > 0 else np.eye(3)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = c - R @ c + t
    return T


def apply(T, P):
    return np.asarray(P, float) @ T[:3, :3].T + T[:3, 3]


def axis_angle(T, u):
    """Signed rotation angle (rad) of T about unit axis u (the axial component of its rotation vector)."""
    R = T[:3, :3]
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    a = np.arccos(c)
    if a < 1e-12:
        return 0.0
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(a))
    return float(a * (w @ u))


def _solve(P, n, r, c, w=None):
    """Least squares point-to-plane step: minimise sum (r + J x)^2, J = [(p - c) x n, n]."""
    J = np.hstack([np.cross(P - c, n), n])
    if w is not None:
        J = J * w[:, None]
        r = r * w
    x, *_ = np.linalg.lstsq(J, -r, rcond=None)
    return x[:3], x[3:]


def _trim(r, pct=TRIM_PCT):
    a = np.abs(r)
    return a <= np.percentile(a, 100.0 - pct)


def sample_points(scan, min_points=MIN_POINTS, seed=0):
    """Scan vertices, topped up with area-uniform samples on the scan triangles to at least min_points."""
    import trimesh
    V = np.asarray(scan.vertices, float)
    need = int(min_points) - len(V)
    if need <= 0 or len(scan.faces) == 0:
        return V
    S = np.asarray(trimesh.sample.sample_surface(scan, need, seed=seed)[0], float)
    return np.vstack([V, S])


def align_v2(scan, solid_tm, n_fold=None, axis=None, T0=None, min_points=MIN_POINTS, max_iter=100, seed=0):
    """Point-to-plane ICP. Returns (T 4x4, info dict). T maps the scan onto the solid."""
    P0 = sample_points(scan, min_points, seed)
    c = P0.mean(0)
    T = np.eye(4) if T0 is None else np.asarray(T0, float).copy()
    u = None
    pitch = None
    if axis is not None:
        u = np.asarray(axis[1], float)
        u = u / np.linalg.norm(u)
    if n_fold and int(n_fold) >= 2 and u is not None:
        pitch = 2 * np.pi / int(n_fold)
    info = {"method": "point-to-plane ICP (spec v2)", "n_points": int(len(P0)), "trim_pct": TRIM_PCT,
            "start": "identity" if T0 is None else "given transform", "n_fold": int(n_fold) if n_fold else None,
            "rotation_bound_deg": None if pitch is None else round(float(np.degrees(pitch / 2)), 6),
            "bound_hit": False, "iterations": 0, "converged": False}
    hist = []
    flags = sharp_edge_flags(solid_tm)
    info["edge_band_mm"] = EDGE_BAND_MM
    for it in range(int(max_iter)):
        P = apply(T, P0)
        r, cp, n, tri = _closest(solid_tm, P)
        k = _trim(r) & ~edge_band(solid_tm, cp, tri, flags=flags)
        om, t = _solve(P[k], n[k], r[k], apply(T, c[None])[0])
        dT = rigid(om, t, apply(T, c[None])[0])
        Tn = dT @ T
        if pitch is not None:
            a = axis_angle(Tn @ np.linalg.inv(np.eye(4) if T0 is None else np.asarray(T0, float)), u)
            if abs(a) > pitch / 2:
                info["bound_hit"] = True
                excess = a - np.sign(a) * pitch / 2
                Tn = rigid(-excess * u, np.zeros(3), apply(Tn, c[None])[0]) @ Tn
        T = Tn
        step_deg = float(np.degrees(np.linalg.norm(om)))
        step_mm = float(np.linalg.norm(t))
        hist.append((round(step_deg, 7), round(step_mm, 7), round(float(np.sqrt(np.mean(r[k] ** 2))), 6)))
        info["iterations"] = it + 1
        if step_deg < CONV_DEG and step_mm < CONV_MM:
            info["converged"] = True
            break
    r, *_ = _closest(solid_tm, apply(T, P0))
    k = _trim(r)
    info["trimmed_rms_mm"] = round(float(np.sqrt(np.mean(r[k] ** 2))), 6)
    info["history_last"] = hist[-3:]
    return T, info


REFINE_GATE_SIGMA = 6.0


def _gate(r, k=REFINE_GATE_SIGMA):
    """Robust gate: |r - median| <= k x 1.4826 x MAD (k = 6). Unlike a fixed 5 % trim it does not cut the tails
    of a blade that is genuinely rotated or thickened (the 5 % trim biased the pose by ~0.009 deg in the
    self-test, a 4-sigma gate by 0.0016 deg, 6 sigma by 0.0003 deg); it only drops gross outliers (missing
    features such as an unmodelled slot, unscanned junk)."""
    m = float(np.median(r))
    s = 1.4826 * float(np.median(np.abs(r - m))) + 1e-9
    return np.abs(r - m) <= k * s


def refine_on_deviation(points, solid_tm, mask=None, max_iter=30):
    """Final rigid refinement on the signed deviation of the inspected points (named exclusions removed,
    robust 6-sigma MAD gate, see _gate). Returns (dT 4x4 to compose AFTER the ICP pose, info)."""
    P0 = np.asarray(points, float)
    if mask is not None:
        P0 = P0[np.asarray(mask, bool)]
    c = P0.mean(0)
    dT = np.eye(4)
    it = 0
    flags = sharp_edge_flags(solid_tm)
    for it in range(int(max_iter)):
        P = apply(dT, P0)
        r, cp, n, tri = _closest(solid_tm, P)
        k = _gate(r) & ~edge_band(solid_tm, cp, tri, flags=flags)
        om, t = _solve(P[k], n[k], r[k], c)
        dT = rigid(om, t, c) @ dT
        if np.degrees(np.linalg.norm(om)) < CONV_DEG / 10 and np.linalg.norm(t) < CONV_MM / 10:
            break
    from .metrology import pose_of
    p = pose_of(dT)
    tmm = float(np.linalg.norm(p["translation_mm"]))
    # translation of the centroid (rotation about the inspected centroid is reported separately)
    t_c = float(np.linalg.norm(apply(dT, c[None])[0] - c))
    info = {"rotation_deg": p["rotation_deg"], "rotation_axis": p["rotation_axis"],
            "centroid_shift_mm": round(t_c, 6), "translation_mm": p["translation_mm"],
            "iterations": it + 1, "n_points": int(len(P0))}
    info["warning"] = None
    if p["rotation_deg"] > WARN_DEG or t_c > WARN_MM:
        info["warning"] = (f"final refinement moved {p['rotation_deg']:.4f} deg / {t_c:.4f} mm "
                           f"(warn above {WARN_DEG} deg / {WARN_MM} mm): the ICP pose was not at the optimum")
    del tmm
    return dT, info


def direct_check(points, solid_tm, mask=None, d_deg=0.005, d_mm=0.005, axis=None):
    """Trimmed rms of the inspected points at the final pose and at +/- perturbations of all 6 degrees of
    freedom (rotations about the centroid). ok = no perturbation lowers it (sign verified by measurement)."""
    P0 = np.asarray(points, float)
    if mask is not None:
        P0 = P0[np.asarray(mask, bool)]
    c = P0.mean(0)
    r0, cp0, _n0, tri0 = _closest(solid_tm, P0)
    k0 = _gate(r0) & ~edge_band(solid_tm, cp0, tri0)
    base = float(np.sqrt(np.mean(r0[k0] ** 2)))
    axes = [("rx", np.array([1.0, 0, 0])), ("ry", np.array([0, 1.0, 0])), ("rz", np.array([0, 0, 1.0]))]
    if axis is not None:
        axes.append(("r_axis", np.asarray(axis, float) / np.linalg.norm(axis)))
    rows = []
    for name, a in axes:
        for s in (1, -1):
            T = rigid(s * np.radians(d_deg) * a, np.zeros(3), c)
            r, *_ = _closest(solid_tm, apply(T, P0))
            rows.append((f"{name}{'+' if s > 0 else '-'}{d_deg}deg", float(np.sqrt(np.mean(r[k0] ** 2)))))
    for name, a in (("tx", [1.0, 0, 0]), ("ty", [0, 1.0, 0]), ("tz", [0, 0, 1.0])):
        for s in (1, -1):
            T = rigid(np.zeros(3), s * d_mm * np.asarray(a), c)
            r, *_ = _closest(solid_tm, apply(T, P0))
            rows.append((f"{name}{'+' if s > 0 else '-'}{d_mm}mm", float(np.sqrt(np.mean(r[k0] ** 2)))))
    worst = min(v for _n, v in rows)
    return {"rms_at_pose_mm": round(base, 7), "perturbed": {n: round(v, 7) for n, v in rows},
            "ok": bool(worst >= base - 1e-7),
            "note": "same gated point set (6-sigma MAD) at every pose; ok = the pose is a measured minimum"}


# ---------------------------------------------------------------- N-fold sectors (blade / tooth index)
def _frame(u):
    u = np.asarray(u, float) / np.linalg.norm(u)
    e1 = np.cross(u, [1.0, 0, 0])
    if np.linalg.norm(e1) < 0.1:
        e1 = np.cross(u, [0, 1.0, 0])
    e1 /= np.linalg.norm(e1)
    return u, e1, np.cross(u, e1)


def _tangential_weights(solid_tm, c, u, n_samples=300000, seed=0):
    """(theta, r, z, weight) of area samples on the solid in the blade band; weight = |tangential normal|."""
    import trimesh
    P, fi = trimesh.sample.sample_surface(solid_tm, n_samples, seed=seed)
    n = np.asarray(solid_tm.face_normals, float)[fi]
    u, e1, e2 = _frame(u)
    V = np.asarray(P) - c
    z = V @ u
    x, y = V @ e1, V @ e2
    r = np.hypot(x, y)
    th = np.arctan2(y, x)
    et = np.stack([-np.sin(th), np.cos(th)], 1)
    n2 = np.stack([n @ e1, n @ e2], 1)
    wt = np.abs(np.einsum("ij,ij->i", n2, et))
    rmax = float(np.percentile(r, 99.5))
    band = (r > 0.3 * rmax) & (r < 0.995 * rmax)
    return th[band], r[band], z[band], wt[band]


def estimate_n_fold(solid_tm, axis, kmin=3, kmax=80):
    """Blade / tooth count from the angular spectrum of tangential-normal surface area, computed per radial
    ring (wrapped blades smear a whole-part spectrum). Returns (N, relative strength)."""
    c, u = axis
    th, r, _z, w = _tangential_weights(solid_tm, c, u)
    if len(th) < 100:
        return None, 0.0
    ks = np.arange(1, kmax + 1)
    edges = np.quantile(r, np.linspace(0, 1, 9))
    amp = np.zeros(len(ks))
    for a, b in zip(edges[:-1], edges[1:]):
        m = (r >= a) & (r <= b)
        if m.sum() < 50:
            continue
        E = np.exp(-1j * np.outer(ks, th[m]))
        amp += np.abs(E @ w[m]) / max(float(w[m].sum()), 1e-12)
    top = amp[kmin - 1:].max()
    for k in range(kmin, kmax + 1):
        # a true N-fold part also shows 2N, 3N ...: the smallest k near the maximum
        if amp[k - 1] >= 0.6 * top:
            return int(k), float(amp[k - 1] / max(amp.sum(), 1e-12))
    return None, 0.0


class SectorField:
    """Blade/tooth centre angle as a function of (r, z): the N-th harmonic phase of the tangential-normal area
    in (r, z) bins, unwrapped so that one blade is followed continuously through its wrap."""

    def __init__(self, solid_tm, axis, n_fold, nr=16, nz=8):
        c, u = axis
        self.c, self.u, self.n = np.asarray(c, float), np.asarray(u, float) / np.linalg.norm(u), int(n_fold)
        th, r, z, w = _tangential_weights(solid_tm, self.c, self.u)
        self.re = np.quantile(r, np.linspace(0, 1, nr + 1))
        self.ze = np.quantile(z, np.linspace(0, 1, nz + 1))
        ir = np.clip(np.searchsorted(self.re, r) - 1, 0, nr - 1)
        iz = np.clip(np.searchsorted(self.ze, z) - 1, 0, nz - 1)
        H = np.zeros((nr, nz), complex)
        np.add.at(H, (ir, iz), w * np.exp(1j * self.n * th))
        cnt = np.zeros((nr, nz))
        np.add.at(cnt, (ir, iz), w)
        ok = np.abs(H) > 0.05 * np.maximum(cnt, 1e-12)
        ang = np.angle(H)
        # fill weak bins from the strongest bin in their row (or the global), then unwrap r, then z
        g = np.angle(H.sum())
        for i in range(nr):
            for k in range(nz):
                if not ok[i, k]:
                    row = ok[i]
                    ang[i, k] = ang[i, np.argmax(np.abs(H[i]))] if row.any() else g
        ang = np.unwrap(ang, axis=0)                     # follow the blade outwards in r, per z column
        for k in range(1, nz):                           # then keep every column on the same blade
            d = float(np.mean(ang[:, k] - ang[:, k - 1]))
            ang[:, k] -= 2 * np.pi * np.round(d / (2 * np.pi))
        self.phase = ang / self.n                                                   # rad, blade-centre angle
        self.nr, self.nz = nr, nz

    def __call__(self, points):
        u, e1, e2 = _frame(self.u)
        V = np.asarray(points, float) - self.c
        z = V @ u
        x, y = V @ e1, V @ e2
        r = np.hypot(x, y)
        th = np.arctan2(y, x)
        ir = np.clip(np.searchsorted(self.re, r) - 1, 0, self.nr - 1)
        iz = np.clip(np.searchsorted(self.ze, z) - 1, 0, self.nz - 1)
        pitch = 2 * np.pi / self.n
        return (np.floor(np.mod(th - self.phase[ir, iz] + pitch / 2, 2 * np.pi) / pitch).astype(int)) % self.n
