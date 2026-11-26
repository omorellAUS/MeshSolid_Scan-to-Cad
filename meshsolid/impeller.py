"""M5: open centrifugal impeller from a scan, built the way a designer would: one analytic/revolved hub
fitted on all sectors, ONE blade fitted on all N sectors folded into one, patterned N times and fused with
the hub into one solid.

Frame: u = rotation axis pointing from the back face towards the eye; z = distance along u from the back
plane (z = 0 at the back plane, z < 0 is impossible, the eye is at z > 0); theta about u.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


# ------------------------------------------------------------------ frame and symmetry
@dataclass
class Frame:
    a0: np.ndarray            # point on the axis, on the back plane
    u: np.ndarray             # axis, back -> eye
    e1: np.ndarray
    e2: np.ndarray

    def local(self, P):
        d = np.asarray(P, float) - self.a0
        x, y, z = d @ self.e1, d @ self.e2, d @ self.u
        return np.hypot(x, y), np.arctan2(y, x), z

    def local_n(self, P, N):
        """normal components (n_r, n_theta, n_z) at P"""
        d = np.asarray(P, float) - self.a0
        x, y = d @ self.e1, d @ self.e2
        r = np.maximum(np.hypot(x, y), 1e-9)
        er = (np.outer(x / r, self.e1) + np.outer(y / r, self.e2))
        et = (np.outer(-y / r, self.e1) + np.outer(x / r, self.e2))
        return np.einsum("ij,ij->i", N, er), np.einsum("ij,ij->i", N, et), N @ self.u

    def world(self, r, th, z):
        r, th, z = map(np.asarray, (r, th, z))
        return (self.a0 + np.multiply.outer(r * np.cos(th), self.e1) + np.multiply.outer(r * np.sin(th), self.e2)
                + np.multiply.outer(z, self.u))


def _basis(u):
    e1 = np.cross(u, [1.0, 0, 0])
    if np.linalg.norm(e1) < 0.1:
        e1 = np.cross(u, [0, 1.0, 0])
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(u, e1)


def find_frame(C, N, A):
    """Axis from the large flat back face (area-weighted plane fit of the dominant axial-normal faces at
    the axial extreme that carries the outer rim) and the rim circle; u points from the back to the eye."""
    c = np.average(C, axis=0, weights=A)
    X = C - c
    w, E = np.linalg.eigh(np.cov(X.T, aweights=A))
    u0 = E[:, 0]                               # disc: smallest spread along the axis
    r0 = np.linalg.norm(X - np.outer(X @ u0, u0), axis=1)
    R = np.percentile(r0, 99.9)
    rim = r0 > 0.85 * R
    z0 = X @ u0
    # the back face is the rim end with more axial-normal area at its extreme
    best = None
    for sg in (1.0, -1.0):
        zz = sg * z0
        k = rim & (np.abs(N @ u0) > 0.95) & (zz > np.percentile(zz[rim], 98) - 1.0)
        if best is None or A[k].sum() > best[0]:
            best = (A[k].sum(), sg, k)
    _, sg, kb = best
    # plane fit on the back-face faces (all radii at that level: back face incl. any flat cap)
    zz = sg * z0
    zb = np.median(zz[kb])
    kb = (np.abs(N @ u0) > 0.95) & (np.abs(zz - zb) < 0.5)
    for _ in range(3):
        cb = np.average(C[kb], axis=0, weights=A[kb])
        _, _, Vt = np.linalg.svd((C[kb] - cb) * np.sqrt(A[kb])[:, None], full_matrices=False)
        nb = Vt[2] * np.sign(Vt[2] @ (sg * u0))
        dist = (C - cb) @ nb
        kb = (np.abs(N @ nb) > 0.95) & (np.abs(dist) < 0.3)
    u = -nb                                     # back -> eye
    e1, e2 = _basis(u)
    # rim circle: outer cylinder faces (radial normals at the max radius), algebraic circle fit
    d = C - cb
    x, y = d @ e1, d @ e2
    rr = np.hypot(x, y)
    nr = np.abs((N @ e1) * x / np.maximum(rr, 1e-9) + (N @ e2) * y / np.maximum(rr, 1e-9))
    k = (rr > 0.97 * R) & (nr > 0.9)
    for _ in range(3):
        M = np.c_[2 * x[k], 2 * y[k], np.ones(k.sum())]
        sol = np.linalg.lstsq(M, x[k] ** 2 + y[k] ** 2, rcond=None)[0]
        cx, cy = sol[0], sol[1]
        Rr = math.sqrt(sol[2] + cx * cx + cy * cy)
        res = np.hypot(x - cx, y - cy) - Rr
        k = k & (np.abs(res) < 0.3)
    a0 = cb + cx * e1 + cy * e2
    a0 = a0 - ((a0 - cb) @ u) * u
    return Frame(a0, u, e1, e2), {"rim_radius": float(Rr), "rim_faces": int(k.sum()), "back_faces": int(kb.sum())}


def blade_count(r, th, nt, A, R, radii=(0.4, 0.55, 0.7, 0.85), nmax=40):
    """N from the angular FFT of tangentially-facing area in rings."""
    votes = []
    for f in radii:
        k = (np.abs(nt) > 0.5) & (np.abs(r - f * R) < 0.03 * R)
        if k.sum() < 200:
            continue
        h = np.histogram(th[k], bins=720, range=(-np.pi, np.pi), weights=A[k])[0]
        F = np.abs(np.fft.rfft(h - h.mean()))
        n = int(np.argmax(F[2:nmax]) + 2)
        votes.append(n)
    if not votes:
        return None, votes
    vals, cnt = np.unique(votes, return_counts=True)
    return int(vals[np.argmax(cnt)]), votes


# ------------------------------------------------------------------ meridional curves
def _theta_cells(th, ntheta=360):
    return ((th + np.pi) / (2 * np.pi) * ntheta).astype(int) % ntheta


def envelope_curve(r, z, th, pick, side, dt=0.5, ntheta=180, min_cells=20, q=None):
    """A meridional curve that is monotone in t = r - z (hub floor or blade tip line of a centrifugal
    wheel), as (t, s = r + z) samples.  In each (t, theta) cell the extreme s on the given side
    (side=-1: material side min, +1: max) is taken; the curve value is the median over the theta cells,
    so features that occupy only a few angles (blade roots, blade tips) do not move it."""
    t = r - z; s = r + z
    k = pick
    it = np.floor(t[k] / dt).astype(int); ic = _theta_cells(th[k], ntheta)
    key = it.astype(np.int64) * ntheta + ic
    o = np.lexsort((side * s[k], key))   # within each key: sorted by side*s ascending
    key_o = key[o]; s_o = s[k][o]
    last = np.r_[np.nonzero(np.diff(key_o))[0], len(key_o) - 1]
    ext = s_o[last]                      # max of side*s -> the extreme on that side
    kk = key_o[last] // ntheta
    T, S = [], []
    for b in np.unique(kk):
        v = ext[kk == b]
        if len(v) >= min_cells:
            T.append((b + 0.5) * dt); S.append(float(np.median(v) if q is None else np.percentile(v, q)))
    return np.array(T), np.array(S)


class _Ext1:
    """1D spline with linear extension beyond its data range"""

    def __init__(self, sp, t0, t1, end_window=0.0):
        self.sp, self.t0, self.t1 = sp, float(t0), float(t1)
        if end_window > 0:        # end slopes as secants over a window: a rounded corner at the data end
            w = min(end_window, 0.3 * (t1 - t0))      # (e.g. tip -> LE round) does not bend the extension
            self.d0 = float((sp(t0 + w) - sp(t0)) / w); self.d1 = float((sp(t1) - sp(t1 - w)) / w)
        else:
            self.d0, self.d1 = float(sp(t0, 1)), float(sp(t1, 1))

    def __call__(self, t, nu=0):
        t = np.asarray(t, float); tc = np.clip(t, self.t0, self.t1)
        if nu == 1:
            return np.where(t < self.t0, self.d0, np.where(t > self.t1, self.d1, self.sp(tc, 1)))
        return self.sp(tc) + np.where(t < self.t0, self.d0 * (t - self.t0), 0) + np.where(t > self.t1, self.d1 * (t - self.t1), 0)


def smooth_curve(T, S, n_knots=12, k=3, iters=4, end_window=0.0):
    """Robust least-squares cubic spline s(t); returns a callable and the inlier rms."""
    from scipy.interpolate import make_lsq_spline
    t0, t1 = T.min(), T.max()
    inner = np.linspace(t0, t1, n_knots + 2)[1:-1]
    kn = np.r_[[t0] * (k + 1), inner, [t1] * (k + 1)]
    w = np.ones(len(T))
    for _ in range(iters):
        sp = make_lsq_spline(T, S, kn, k=k, w=w)
        res = S - sp(T)
        mad = 1.4826 * np.median(np.abs(res)) + 1e-6
        w = 1.0 / (1.0 + (res / (2.5 * mad)) ** 2)
    rms = float(np.sqrt(np.average(res ** 2, weights=w)))
    return _Ext1(sp, t0, t1, end_window), rms


def boundary_curve(r, z, pick, q=99.7, dt=0.5, min_n=200):
    """Outer boundary s(t) of a point set in the meridional plane (percentile of s = r + z per t bin)."""
    t = r - z; s = r + z
    tb = np.floor(t[pick] / dt).astype(int); sv = s[pick]
    T, S = [], []
    for b in np.unique(tb):
        v = sv[tb == b]
        if len(v) >= min_n:
            T.append((b + 0.5) * dt); S.append(float(np.percentile(v, q)))
    return np.array(T), np.array(S)


def _eye_profile(r, z, th, nr, r_max, z_min, dz=0.25, ntheta=180):
    """lower radial envelope r(z) of the outward faces above z_min (median over theta cells)"""
    k = (nr > 0.3) & (z > z_min) & (r < r_max)
    if k.sum() < 200:
        return np.zeros(0), np.zeros(0)
    iz = np.floor((z[k] - z_min) / dz).astype(int); ic = _theta_cells(th[k], ntheta)
    key = iz.astype(np.int64) * ntheta + ic
    o = np.lexsort((r[k], key)); ko = key[o]; ro = r[k][o]
    first = np.r_[0, np.nonzero(np.diff(ko))[0] + 1]
    zb = ko[first] // ntheta; rmin = ro[first]
    Z, R = [], []
    for b in np.unique(zb):
        v = rmin[zb == b]
        if len(v) >= 20:
            Z.append(z_min + (b + 0.5) * dz); R.append(float(np.median(v)))
    return np.array(Z), np.array(R)


def eye_cylinder(r, z, th, nr, r_max, z_min, dz=0.25, ntheta=180, run=8, tol=0.03):
    """Hub nose at the eye: radial profile r(z) above z_min (lower envelope over theta cells, median).
    If the nose runs into a straight axial cylinder (a shaft, nut or scan fixture standing on the front
    face), the hub ends at the corner where the profile reaches the cylinder radius.  Returns
    dict(z_front, r_cyl, profile) or None when no cylinder follows the nose."""
    k = (nr > 0.3) & (z > z_min) & (r < r_max)
    if k.sum() < 200:
        return None
    iz = np.floor((z[k] - z_min) / dz).astype(int); ic = _theta_cells(th[k], ntheta)
    key = iz.astype(np.int64) * ntheta + ic
    o = np.lexsort((r[k], key)); ko = key[o]; ro = r[k][o]
    first = np.r_[0, np.nonzero(np.diff(ko))[0] + 1]
    zb = ko[first] // ntheta; rmin = ro[first]
    Z, R = [], []
    for b in np.unique(zb):
        v = rmin[zb == b]
        if len(v) >= 20:
            Z.append(z_min + (b + 0.5) * dz); R.append(float(np.median(v)))
    Z, R = np.array(Z), np.array(R)
    for j in range(len(R) - run):
        seg = R[j:j + run]
        if seg.max() - seg.min() < tol and np.all(np.diff(Z[j:j + run]) < 1.5 * dz):
            r_cyl = float(np.median(R[j:j + 3 * run]))
            if j < 2 or R[0] - r_cyl < 4 * tol:       # no nose before it: not a nose/cylinder corner
                return None
            # corner: last nose sample clearly outside the cylinder, extrapolated linearly onto it
            i = int(np.nonzero(R[:j + 1] - r_cyl > 2 * tol)[0][-1])
            i0 = max(0, i - 3)
            a = np.polyfit(R[i0:i + 1], Z[i0:i + 1], 1) if i > i0 else (0.0, Z[i])
            zf = float(np.polyval(a, r_cyl + tol)) if i > i0 else float(Z[i])
            zf = float(np.clip(zf, Z[i], Z[min(j, len(Z) - 1)] + dz))
            return {"z_front": zf, "r_cyl": r_cyl, "profile": np.c_[R[:i + 1], Z[:i + 1]]}
    return None


def eye_separation(x, y, z, r, z_lo, z_hi, prof_z, prof_r, dz=0.5, band=0.4, min_n=60):
    """Concentricity of the eye-side hub surface: a circle is fitted to the points near the hub profile
    in each axial slice.  The wheel's own surfaces are concentric with its axis (centre offset ~0.01 mm);
    a separately mounted piece (nose nut, spinner, scan fixture) is not.  Returns dict(z_sep, ecc profile)
    with z_sep where the offset departs from the wheel's baseline (back-extrapolated), or None."""
    rows = []
    for z0 in np.arange(z_lo, z_hi, dz):
        k = (z > z0) & (z < z0 + dz)
        if k.sum() < min_n:
            continue
        rp = np.interp(z0 + dz / 2, prof_z, prof_r)
        k &= np.abs(r - rp) < band
        if k.sum() < min_n:
            continue
        M = np.c_[2 * x[k], 2 * y[k], np.ones(k.sum())]
        c = np.linalg.lstsq(M, x[k] ** 2 + y[k] ** 2, rcond=None)[0]
        rows.append((z0 + dz / 2, float(np.hypot(c[0], c[1]))))
    if len(rows) < 6:
        return None
    Z, E = np.array(rows).T
    base = float(np.median(E[:3]))
    thr = max(0.04, 3 * base)
    for i in range(len(E) - 2):
        if np.all(E[i:i + 3] > thr):
            j = i
            # departure start: a shifted piece steps (onset at the slice boundary); a tilted one grows
            # linearly and is back-extrapolated, at most two slices
            if i >= 1 and E[i] > 2 * E[i - 1]:
                zs = float(Z[i] - dz / 2)
            elif i >= 1:
                a = np.polyfit(Z[i - 1:i + 3], E[i - 1:i + 3], 1)
                zs = float((base - a[1]) / a[0]) if a[0] > 1e-6 else float(Z[i - 1])
                zs = float(np.clip(zs, Z[max(i - 2, 0)], Z[i]))
            else:
                zs = float(Z[i] - dz / 2)
            return {"z_sep": zs, "baseline_ecc": round(base, 4), "ecc": [(round(a, 2), round(b, 4)) for a, b in zip(Z, E)]}
    return None


def fit_floor(spec, d, cells=(40, 50), h_max=0.6, v_cam=0.12, min_n=8):
    """Passage floor as a periodic sheet: height h(p, phi) above the revolved hub (along the mer-map v
    direction, mm), phi in [0, 1) across one passage from a blade's camber to the next.  Cell medians
    of the floor-facing faces; cells under the root blends (h > h_max) or without data are filled from the
    passage interior (constant along phi), then lightly smoothed.  The floor carries what a revolve
    cannot: machining ridges, floor/fillet steps, a floor that is not exactly axisymmetric."""
    from scipy.ndimage import gaussian_filter
    N = spec.N; mm = spec.mermap()
    r, th, z, nt = d["r"], d["th"], d["z"], d["nt"]
    k = (np.abs(nt) < 0.3) & (z > 0.3)
    p, v = mm.pv(r[k], z[k]); ok = np.isfinite(p) & (p > -0.02) & (p < 1.0) & (v < 0.25) & (v > -0.1)
    idx = np.where(k)[0][ok]; p = p[ok]; v = v[ok]
    hr, hz = spec.hub.at(p); sr, sz = spec.shroud.at(p); h = v * np.hypot(sr - hr, sz - hz)
    thc = spec.camber(np.clip(p, 0, 1), np.full_like(p, v_cam))
    phi = np.mod(th[idx] - thc, 2 * np.pi / N) * N / (2 * np.pi)
    sel = h < h_max + 0.6
    nP, nF = cells
    ip = np.clip((p[sel] * nP).astype(int), 0, nP - 1); jf = np.clip((phi[sel] * nF).astype(int), 0, nF - 1)
    key = ip * nF + jf; hv = h[sel]
    o = np.argsort(key, kind="stable"); key, hv = key[o], hv[o]
    G = np.full(cells, np.nan); C = np.zeros(cells, int)
    st = np.r_[0, np.nonzero(np.diff(key))[0] + 1]; en = np.r_[st[1:], len(key)]
    for a, b in zip(st, en):
        if b - a >= min_n:
            G[key[a] // nF, key[a] % nF] = np.median(hv[a:b]); C[key[a] // nF, key[a] % nF] = b - a
    G[G > h_max] = np.nan                       # root blends belong to the blade sheets
    n_floor = int(np.isfinite(G).sum())
    for i in range(nP):                         # fill along phi from the passage interior
        row = G[i]; good = np.isfinite(row)
        if good.sum() < 3:
            continue
        j = np.arange(nF); row[~good] = np.interp(j[~good], j[good], row[good])
    for jj in range(nF):                        # rows without data: from neighbouring p
        col = G[:, jj]; good = np.isfinite(col)
        if good.any() and (~good).any():
            col[~good] = np.interp(np.nonzero(~good)[0], np.nonzero(good)[0], col[good])
    G = np.nan_to_num(G, nan=0.0)
    # periodic in phi: smooth with wrap, light (0.7 cell)
    Gs = gaussian_filter(G, (0.7, 0.7), mode=("nearest", "wrap"))
    pc = (np.arange(nP) + 0.5) / nP; fc = (np.arange(nF) + 0.5) / nF
    return {"pc": pc, "fc": fc, "h": Gs, "cells_with_floor": n_floor, "v_cam": v_cam,
            "h_range": [round(float(Gs.min()), 3), round(float(Gs.max()), 3)]}


def edge_round(depth, half_width):
    """Radius of a round on the edge of a plate from its half-width at a set of depths below the cut face:
    w(h) = t/2 - rho + sqrt(rho^2 - (rho - h)^2) for h < rho, t/2 beyond.  Returns (rho, t/2, rms)."""
    h = np.asarray(depth, float); w = np.asarray(half_width, float)
    best = None
    for rho in np.linspace(0.05, 2.0, 196):
        for h0 in np.linspace(-0.2, 0.2, 41):            # offset of the cut plane estimate
            hh = np.clip(h - h0, 0, None)
            g = np.where(hh < rho, -rho + np.sqrt(np.clip(rho ** 2 - (rho - hh) ** 2, 0, None)), 0.0)
            t2 = float(np.mean(w - g)); e = float(np.sqrt(np.mean((w - g - t2) ** 2)))
            if best is None or e < best[3]:
                best = (rho, t2, h0, e)
    return {"rho": float(best[0]), "half_t": best[1], "plane_offset": float(best[2]), "rms": best[3]}


class MerCurve:
    """A meridional curve given as s(t) (t = r - z, s = r + z) restricted to [ta, tb], with arc-length
    parameter p in [0, 1] and linear extension beyond both ends."""

    def __init__(self, sp, ta, tb, n=2000):
        self.sp = sp
        tt = np.linspace(ta, tb, n); ss = sp(tt)
        self.R = (ss + tt) / 2; self.Z = (ss - tt) / 2
        d = np.hypot(np.diff(self.R), np.diff(self.Z))
        self.L = float(d.sum()); self.p = np.r_[0, np.cumsum(d)] / self.L

    def at(self, p):
        p = np.asarray(p, float)
        R = np.interp(p, self.p, self.R); Z = np.interp(p, self.p, self.Z)
        lo = p < 0; hi = p > 1
        if lo.any():
            dr, dz = (self.R[1] - self.R[0]) / self.p[1], (self.Z[1] - self.Z[0]) / self.p[1]
            R = np.where(lo, self.R[0] + p * dr, R); Z = np.where(lo, self.Z[0] + p * dz, Z)
        if hi.any():
            dr, dz = (self.R[-1] - self.R[-2]) / (1 - self.p[-2]), (self.Z[-1] - self.Z[-2]) / (1 - self.p[-2])
            R = np.where(hi, self.R[-1] + (p - 1) * dr, R); Z = np.where(hi, self.Z[-1] + (p - 1) * dz, Z)
        return R, Z


def _root(f, ta, tb, n=400):
    from scipy.optimize import brentq
    tt = np.linspace(ta, tb, n); ff = np.array([f(t) for t in tt])
    i = np.nonzero(np.sign(ff[:-1]) != np.sign(ff[1:]))[0]
    if len(i) == 0:
        raise ValueError("no crossing")
    return brentq(f, tt[i[0]], tt[i[0] + 1])


def t_at_z(sp, z0, ta, tb):
    return _root(lambda t: float(sp(t) - t) / 2 - z0, ta, tb)


def t_at_z_or_end(sp, z0, ta, tb, t_end):
    """t where the hub reaches the LE plane z0, or the hub's own front end t_end when it never does (an
    inclined leading edge whose blade tops rise above the hub nose, e.g. the KK&K stage 1: hub nose z 75,
    blade tops 75 -> 79.6 from hub to tip). Returns (t, reached)."""
    try:
        return t_at_z(sp, z0, ta, tb), True
    except ValueError:
        return float(t_end), False


def t_at_z_closest(sp, z0, ta, tb, max_gap=1.0):
    """t where the curve reaches z0; when it falls just short (a rounded tip-to-LE corner leaves the tip curve
    0.07 mm under the LE plane on the 165 mm 15-blade wheel), the point of closest approach if within max_gap.
    Returns (t, gap) with gap = 0 when reached; raises ValueError beyond max_gap."""
    try:
        return t_at_z(sp, z0, ta, tb), 0.0
    except ValueError:
        tt = np.linspace(ta, tb, 2000)
        zz = (sp(tt) - tt) / 2
        i = int(np.argmin(np.abs(zz - z0)))
        gap = float(abs(zz[i] - z0))
        if gap > max_gap:
            raise
        return float(tt[i]), gap


def t_at_r_wide(sp, r0, ta, tb, frac=0.15):
    """t_at_r, retried with the end slope taken as a secant over the last `frac` of the data when the short
    end window bends the extension away from r0 (a tip curve ending in a TE round / rim step, KK&K stage 1:
    tip data stops at r 110.7 with a -2.2 end slope, so the extension never reaches the 119 mm rim).
    Returns (t, curve): the curve the t belongs to (the widened one on fallback; the caller must build on it,
    the original curve evaluated at that t lies elsewhere)."""
    try:
        return t_at_r(sp, r0, ta, tb), sp
    except ValueError:
        if not isinstance(sp, _Ext1):
            raise
        w = frac * (sp.t1 - sp.t0)
        wide = _Ext1(sp.sp, sp.t0, sp.t1, end_window=w)
        return t_at_r(wide, r0, ta, tb + 5 * w), wide


def t_at_r(sp, r0, ta, tb):
    return _root(lambda t: float(sp(t) + t) / 2 - r0, ta, tb)


def _fill_cells(Zc, good):
    """Unit phasors on a (nP, nV) cell grid with bad cells replaced by nearest-good-cell values (along v
    within each p row, rows without any good cell from the nearest good row)."""
    Zu = np.where(good, Zc / np.maximum(np.abs(Zc), 1e-30), 0)
    out = Zu.copy(); rows_ok = good.any(axis=1)
    if not rows_ok.any():
        return Zc
    for i in np.where(rows_ok)[0]:
        jg = np.where(good[i])[0]
        jb = np.where(~good[i])[0]
        if len(jb):
            out[i, jb] = Zu[i, jg[np.abs(jb[:, None] - jg[None, :]).argmin(1)]]
    ig = np.where(rows_ok)[0]
    for i in np.where(~rows_ok)[0]:
        out[i] = out[ig[np.abs(ig - i).argmin()]]
    return out


class MerMap:
    """(p, v) <-> (r, z): M(p, v) = (1 - v) H(p) + v S(p) between the hub and shroud curves."""

    def __init__(self, H: MerCurve, S: MerCurve, pr=(-0.06, 1.06), vr=(-0.25, 1.25), n=(500, 160)):
        from scipy.interpolate import LinearNDInterpolator
        self.H, self.S = H, S
        P, Vv = np.meshgrid(np.linspace(*pr, n[0]), np.linspace(*vr, n[1]), indexing="ij")
        Rr, Zz = self.rz(P.ravel(), Vv.ravel())
        self._inv = LinearNDInterpolator(np.c_[Rr, Zz], np.c_[P.ravel(), Vv.ravel()])

    def rz(self, p, v):
        hr, hz = self.H.at(p); sr, sz = self.S.at(p)
        return (1 - v) * hr + v * sr, (1 - v) * hz + v * sz

    def pv(self, r, z):
        out = self._inv(np.c_[r, z])
        return out[:, 0], out[:, 1]


# ------------------------------------------------------------------ analysis
@dataclass
class Spec:
    frame: Frame
    N: int
    hub: object = None          # MerCurve hub floor LE -> TE
    shroud: object = None       # MerCurve tip line LE -> TE
    hub_spline: object = None
    shroud_spline: object = None
    hub_t: tuple = ()
    z_le: float = 0.0
    r_te: float = 0.0
    camber: object = None       # theta_c(p, v)
    sides: dict = field(default_factory=dict)    # +1 / -1 -> LSQBivariateSpline of offset from camber
    info: dict = field(default_factory=dict)

    def mermap(self):
        return MerMap(self.hub, self.shroud)


def _even_faces(mesh, trigger=0.1, rel=0.01):
    """Face-centre statistics assume small, even facets.  CAD exports carry long facets on flat, ruled and
    revolved faces (area-weighted 90th percentile of the longest facet edge > trigger * radius); those are
    subdivided to rel * radius so face centres sample the surface evenly.  Scans are returned unchanged.
    Returns (mesh, max_edge or None)."""
    import trimesh
    V = np.asarray(mesh.vertices); Fc = np.asarray(mesh.faces)
    R = 0.5 * float(np.max(mesh.extents))
    L = np.linalg.norm(V[Fc] - V[np.roll(Fc, 1, 1)], axis=2).max(1)
    o = np.argsort(L); c = np.cumsum(mesh.area_faces[o]) / mesh.area
    if L[o][min(np.searchsorted(c, 0.9), len(o) - 1)] <= trigger * R:
        return mesh, None
    lim = rel * R
    v, f = trimesh.remesh.subdivide_to_size(V, Fc, max_edge=lim, max_iter=20)
    return trimesh.Trimesh(v, f, process=False), round(lim, 4)


def _faces(mesh, F):
    C = np.asarray(mesh.triangles_center); Nn = np.asarray(mesh.face_normals); A = np.asarray(mesh.area_faces)
    r, th, z = F.local(C); nr, nt, nz = F.local_n(C, Nn)
    return dict(r=r, th=th, z=z, nr=nr, nt=nt, nz=nz, A=A)


class _Ext2:
    """bivariate spline evaluated with a linear (first-order) extension outside [0,1]^2"""

    def __init__(self, spl, lo=(0.0, 0.0), hi=(1.0, 1.0), lin=(True, True, True, True)):
        """lin: linear continuation at (p low, p high, v low, v high); False = held constant"""
        self.s, self.lo, self.hi, self.lin = spl, np.asarray(lo), np.asarray(hi), lin

    def __call__(self, p, v):
        p = np.asarray(p, float); v = np.asarray(v, float)
        pc = np.clip(p, self.lo[0], self.hi[0]); vc = np.clip(v, self.lo[1], self.hi[1])
        f = self.s.ev(pc, vc)
        dp = p - pc; dv = v - vc
        dp = np.where(dp < 0, dp * self.lin[0], dp * self.lin[1])
        dv = np.where(dv < 0, dv * self.lin[2], dv * self.lin[3])
        return f + self.s.ev(pc, vc, dx=1) * dp + self.s.ev(pc, vc, dy=1) * dv


class _CamberExt(_Ext2):
    """_Ext2 whose linear continuation past the p ends (the LE overhang above the eye plane, the TE) uses a
    slope that is smooth across the span: the spline's end derivative zigzags along v (end cells are
    polluted by edge rounds), which made the LE sheets wavy and the LE fillet fail.  Slope = secant over
    0.1 in p, fitted along v by a cubic."""

    def __init__(self, spl, lo, hi, lin=(True, True, False, True), w=0.1, zig_rel=0.25):
        super().__init__(spl, lo, hi, lin)
        vv = np.linspace(lo[1], hi[1], 40)
        self._sl = []; self.smoothed = []
        for p0, p1, pe in ((lo[0], lo[0] + w, lo[0]), (hi[0] - w, hi[0], hi[0])):
            sl = (spl.ev(np.full_like(vv, p1), vv) - spl.ev(np.full_like(vv, p0), vv)) / (p1 - p0)
            c = np.polyfit(vv, sl, 3)
            raw = spl.ev(np.full_like(vv, pe), vv, dx=1)
            # only where the end derivative zigzags (rms off its smooth fit > 25% of its rms): clean
            # fits keep the spline's own continuation
            zig = np.sqrt(np.mean((raw - np.polyval(c, vv)) ** 2)) > zig_rel * np.sqrt(np.mean(raw ** 2))
            self.smoothed.append(bool(zig))
            self._sl.append(c if zig else None)

    def __call__(self, p, v):
        p = np.asarray(p, float); v = np.asarray(v, float)
        pc = np.clip(p, self.lo[0], self.hi[0]); vc = np.clip(v, self.lo[1], self.hi[1])
        f = self.s.ev(pc, vc)
        dp = p - pc; dv = v - vc
        raw = self.s.ev(pc, vc, dx=1)
        s0 = np.polyval(self._sl[0], vc) if self._sl[0] is not None else raw
        s1 = np.polyval(self._sl[1], vc) if self._sl[1] is not None else raw
        slp = np.where(dp < 0, s0 * self.lin[0], s1 * self.lin[1])
        dv = np.where(dv < 0, dv * self.lin[2], dv * self.lin[3])
        return f + slp * dp + self.s.ev(pc, vc, dy=1) * dv


def _fit_sides(p, v, dl, side, rr, inner, cells=(60, 28), box=(0.03, 0.97, 0.0, 0.92)):
    """Each blade side as offset-from-camber(p, v): cell medians over the stacked sectors (robust to
    outliers and to the rounded edges, which are outside the box), holes filled by interpolation, then a
    lightly smoothed bicubic spline on the cell grid.  Outside the box the surface is continued linearly
    (_Ext2), so the control net stays bounded."""
    from scipy.interpolate import RectBivariateSpline, griddata
    fits, stats, fallback = {}, {}, {}
    p0, p1, v0, v1 = box
    pe = np.linspace(p0, p1, cells[0] + 1)
    ve = v0 + (v1 - v0) * np.linspace(0, 1, cells[1] + 1) ** 1.6     # dense near the hub (root blend)
    pc = 0.5 * (pe[1:] + pe[:-1]); vc = 0.5 * (ve[1:] + ve[:-1])
    for sg in (1, -1):
        m = inner & (side == sg) & (np.sign(dl) == sg)
        ip = np.digitize(p[m], pe) - 1; iv = np.digitize(v[m], ve) - 1
        ok = (ip >= 0) & (ip < cells[0]) & (iv >= 0) & (iv < cells[1])
        key = ip[ok] * cells[1] + iv[ok]; f = dl[m][ok]
        o = np.argsort(key, kind="stable"); key, f = key[o], f[o]
        G = np.full(cells, np.nan); cnt = np.zeros(cells, int)
        starts = np.r_[0, np.nonzero(np.diff(key))[0] + 1]; ends = np.r_[starts[1:], len(key)]
        for a, b in zip(starts, ends):
            if b - a >= 5:
                c = key[a]; G[c // cells[1], c % cells[1]] = np.median(f[a:b]); cnt[c // cells[1], c % cells[1]] = b - a
        PP, VV = np.meshgrid(pc, vc, indexing="ij")
        good = np.isfinite(G)
        if (~good).any():
            fill = griddata(np.c_[PP[good], VV[good]], G[good], np.c_[PP[~good], VV[~good]], method="linear")
            bad = ~np.isfinite(fill)
            if bad.any():
                fill[bad] = griddata(np.c_[PP[good], VV[good]], G[good], np.c_[PP[~good], VV[~good]][bad], method="nearest")
            G[~good] = fill
        # smoothing: residual budget ~ (0.01 mm / r)^2 per cell
        rm = float(np.median(rr[m]))
        from scipy.interpolate import RegularGridInterpolator, LSQBivariateSpline
        pd_ = np.linspace(pc[0], pc[-1], 4 * len(pc)); vd_ = np.linspace(vc[0], vc[-1], 4 * len(vc))
        lin_ = RegularGridInterpolator((pc, vc), G)(np.stack(np.meshgrid(pd_, vd_, indexing="ij"), -1))

        def _dev(sp_):
            # between the nodes too: a near-interpolating FITPACK fit can oscillate wildly there
            d_ = float(np.nanmax(np.abs(sp_(pd_, vd_) - lin_))) * rm
            return d_ if np.isfinite(d_) else 1e9

        spl = None; tried = []
        for fac in (1.0, 4.0, 16.0, 64.0, 256.0):   # smoothing budget (0.01 mm)^2 per cell, relaxed if it oscillates
            cand = RectBivariateSpline(pc, vc, G, kx=3, ky=3, s=fac * G.size * (0.01 / rm) ** 2)
            dv_ = _dev(cand); tried.append((dv_, f"s x{fac:g}", cand))
            if dv_ <= 0.15:
                spl = cand
                if fac > 1:
                    fallback[sg] = f"s x{fac:g}"
                break
        if spl is None:
            # fixed-knot least squares on the complete cell grid, knots every 2 cells; then whichever
            # candidate deviates least from the cell grid (cummins_140: s x1 dev 1.5 mm, lsq 2.5 mm)
            tx = pc[2:-2:2]; ty = vc[2:-2:2]
            PPf, VVf = np.meshgrid(pc, vc, indexing="ij")
            lsq = LSQBivariateSpline(PPf.ravel(), VVf.ravel(), G.ravel(), tx, ty, kx=3, ky=3)
            tried.append((_dev(lsq), "lsq", lsq))
            best = min(tried, key=lambda t_: t_[0])
            spl = best[2]
            fallback[sg] = f"{best[1]} (dev {best[0]:.3g} mm; " + ", ".join(f"{n_} {d_:.3g}" for d_, n_, _ in tried) + ")"
        res = (dl[m] - spl.ev(np.clip(p[m], p0, p1), np.clip(v[m], v0, v1))) * rr[m]
        # below the hub the side is held constant (it is inside the hub); elsewhere continued linearly
        fits[sg] = _Ext2(spl, (pc[0], vc[0]), (pc[-1], vc[-1]), lin=(False, False, False, True))
        fits[sg].grid = (pc, vc, G, cnt)
        stats[sg] = {"n": int(m.sum()), "rms_mm": round(float(np.sqrt(np.mean(res ** 2))), 4),
                     "p95_mm": round(float(np.percentile(np.abs(res), 95)), 4), "empty_cells": int((~good).sum()),
                     "grid_fit_fallback": fallback.get(sg)}
    return fits, stats


def analyse(mesh, n_axis_iter=6, log=print):
    from scipy.interpolate import RectBivariateSpline, interp1d
    mesh, sub = _even_faces(mesh)
    C = np.asarray(mesh.triangles_center); Nn = np.asarray(mesh.face_normals); A = np.asarray(mesh.area_faces)
    F, finfo = find_frame(C, Nn, A)
    info = {"frame": finfo, "axis_corrections": [], "subdivided_to_mm": sub}
    for it in range(n_axis_iter + 1):
        d = _faces(mesh, F)
        r, th, z, nr, nt, nz = (d[k] for k in ("r", "th", "z", "nr", "nt", "nz"))
        R = finfo["rim_radius"]
        if it == 0:
            N, votes = blade_count(r, th, nt, A, R)
            info["blade_count_votes"] = votes
        # LE plane: the axial face at the eye carrying the blade tops.  The top of the blade sides is taken
        # per angular sector (median over N sectors), so a keyway or other single feature cannot set it.
        pt = (np.abs(nt) > 0.5) & (z > 0.5)
        sec = np.floor((th[pt] + np.pi) / (2 * np.pi) * N).astype(int) % N
        tops = [np.percentile(z[pt][sec == i], 99.5) for i in range(N) if np.sum(sec == i) > 50]
        ztop = float(np.median(tops))
        if splitter_pattern(info.get("blade_count_votes") or []) is not None and len(tops) == N:
            # full + splitter: only the full blades reach the eye; the median of N mixed tops falls between
            ztop = float(max(np.median(tops[0::2]), np.median(tops[1::2])))
        k = (nz > 0.97) & (np.abs(z - ztop) < 1.0)
        z_le = float(np.median(z[k]))
        k = (np.abs(nr) > 0.9) & (r > 0.98 * R) & (np.abs(nt) < 0.3)
        r_te = float(np.median(r[k]))
        # hub floor (and its nose at the eye, up to a shaft/stub cylinder if one stands on the front face);
        # axial faces above the LE plane (a front face) are not hub floor
        notfront = ~((nz > 0.9) & (z > z_le - 0.3))
        pk = (np.abs(nt) < 0.3) & (nr + nz > 0.5) & (z > 1.0) & (r > 0.15 * R) & (r < 0.995 * R) & notfront
        T, S = envelope_curve(r, z, th, pk, -1)
        r_end, z_end = (S[0] + T[0]) / 2, (S[0] - T[0]) / 2
        eye = eye_cylinder(r, z, th, nr, r_end + 1.0, z_end - 0.5)
        if eye is not None:
            pk = (np.abs(nt) < 0.3) & (nr + nz > 0.5) & (z > 1.0) & (r > eye["r_cyl"] + 0.05) & \
                 (z < eye["z_front"]) & (r < 0.995 * R) & notfront
            T, S = envelope_curve(r, z, th, pk, -1)
        else:
            # a machined front face: axial faces above the LE plane inside the hub nose radius
            kf = (nz > 0.97) & (z > z_le + 0.3) & (r < r_end + 0.5)
            if A[kf].sum() > 0.5 * np.pi * max(r_end ** 2 - (0.5 * r_end) ** 2, 1.0):
                zf_ = float(np.median(z[kf]))
                eye = {"z_front": zf_, "r_cyl": None, "kind": "front_face"}
                pk = pk & (z < zf_)
                T, S = envelope_curve(r, z, th, pk, -1)
        # a separately mounted piece on the eye (not concentric with the wheel) ends the hub as well
        if True:  # also when a flat face was found: a separate piece can carry its own end face
            Cx = C - F.a0; xx, yy = Cx @ F.e1, Cx @ F.e2
            Rp, Zp = (S + T) / 2, (S - T) / 2
            o_ = np.argsort(Zp)
            r_lim = max(r_end, float(np.interp(z_le - 3.0, Zp[o_], Rp[o_]))) + 1.0
            pr_z, pr_r = _eye_profile(r, z, th, nr, r_lim, z_le - 3.0)
            sep = eye_separation(xx, yy, z, r, z_le - 3.0, float(pr_z.max()) if len(pr_z) else z_le, pr_z, pr_r) \
                if len(pr_z) > 6 else None
            info["eye_concentricity"] = sep
            if sep is not None and (eye is None or sep["z_sep"] < eye["z_front"] - 0.25) and sep["z_sep"] > z_le:
                eye = {"z_front": sep["z_sep"], "r_cyl": None, "kind": "separate_piece"}
                pk = (np.abs(nt) < 0.3) & (nr + nz > 0.5) & (z > 1.0) & (z < sep["z_sep"]) & (r < 0.995 * R) & notfront
                T, S = envelope_curve(r, z, th, pk, -1)
        hsp, hrms = smooth_curve(T, S, n_knots=14)
        Tb, Sb = boundary_curve(r, z, pt)
        bnd = interp1d(Tb, Sb, bounds_error=False, fill_value=np.nan)
        ds = (r + z) - bnd(r - z)
        ktip = (np.abs(nt) < 0.5) & (nr + nz > 0.3) & (ds > -1) & (ds < 1.5) & (z < z_le - 0.4) & (r < r_te - 0.3)
        # trailing-edge faces (radial normals, |nz| small) near the rim are not tip: on a coarse
        # tessellation (Cummins 140, 1.3 mm facets) their centroids fall inside r_te - 0.3 and drag the tip
        # line's end down to the hub (tip z -1.75 at r_te vs ~8.7 on the mesh -> envelope profile crosses
        # the hub, blade trim empty)
        ktip &= ~((r > r_te - 4.0) & (np.abs(nz) < 0.5))
        Tt, St = boundary_curve(r, z, ktip, q=50, min_n=30)
        ssp, srms = smooth_curve(Tt, St, n_knots=10, end_window=3.0)
        th0, reached = t_at_z_or_end(hsp, z_le, T.min() - 5, T.max(), T.min())
        info["hub_reaches_le_plane"] = bool(reached)
        th1 = t_at_r(hsp, r_te, T.min(), T.max() + 10)
        ts1, ssp_w = t_at_r_wide(ssp, r_te, Tt.min(), Tt.max() + 10)
        if ssp_w is not ssp:
            info["tip_end_secant_widened"] = True
            ssp = ssp_w
        ts0, gap_le = t_at_z_closest(ssp, z_le, Tt.min() - 20, Tt.max()); info["tip_le_gap_mm"] = round(gap_le, 4)
        H = MerCurve(hsp, th0, th1); Sh = MerCurve(ssp, ts0, ts1)
        mm = MerMap(H, Sh)
        kb = np.where((np.abs(nt) > 0.25) & (z > 0.5))[0]
        p, v = mm.pv(r[kb], z[kb]); ok = np.isfinite(p); kb, p, v = kb[ok], p[ok], v[ok]
        # faces between 0.25 and 0.5 tangential only on the root blend (within 3 mm of the hub floor)
        hr_, hz_ = H.at(p); sr_, sz_ = Sh.at(p)
        hh = v * np.hypot(sr_ - hr_, sz_ - hz_)
        keep = (np.abs(nt[kb]) > 0.5) | (hh < 3.0)
        kb, p, v = kb[keep], p[keep], v[keep]
        nP, nV = 50, 16
        ip = np.clip((p * nP).astype(int), 0, nP - 1); iv = np.clip((v * nV).astype(int), 0, nV - 1)
        Zc = np.zeros((nP, nV), complex); np.add.at(Zc, (ip, iv), np.exp(1j * N * th[kb]) * A[kb])
        Wc = np.zeros((nP, nV)); np.add.at(Wc, (ip, iv), A[kb])
        # empty or incoherent cells (no faces, or both sides of a thin blade cancelling poorly) carry no
        # phase: fill them from the nearest good cells, along v first, then along p
        good = (Wc > 0.05 * np.median(Wc[Wc > 0])) & (np.abs(Zc) > 0.3 * Wc)
        Zc = _fill_cells(Zc, good)
        ph = np.angle(Zc) / N
        U = np.zeros_like(ph); mid = nV // 2
        U[:, mid] = np.unwrap(ph[:, mid] * N) / N
        for j in list(range(mid + 1, nV)) + list(range(mid - 1, -1, -1)):
            ref = U[:, j - 1] if j > mid else U[:, j + 1]
            U[:, j] = ref + np.angle(np.exp(1j * N * (ph[:, j] - ref))) / N
        pc = (np.arange(nP) + 0.5) / nP; vc = (np.arange(nV) + 0.5) / nV
        # the camber is a smooth reference only (the side sheets carry the detail): fitted above the root
        # blend (v >= 0.12, where fillet and floor features pollute the phase) with a 0.02 mm budget, held
        # constant below
        # and away from the LE/TE end cells (partly past the blade ends: edge rounds, trailing-edge faces,
        # the exit rim), extended linearly from there
        j0 = int(np.searchsorted(vc, 0.12))
        i0, i1 = 2, nP - 2
        r_m = float(np.median(r[kb]))
        cam = RectBivariateSpline(pc[i0:i1], vc[j0:], U[i0:i1, j0:], kx=3, ky=3,
                                  s=(i1 - i0) * (nV - j0) * (0.02 / r_m) ** 2)
        camx = _CamberExt(cam, (pc[i0], vc[j0]), (pc[i1 - 1], vc[-1]), lin=(True, True, False, True))
        thc = camx(p, v)
        dl = np.angle(np.exp(1j * N * (th[kb] - thc))) / N
        side = np.sign(nt[kb])
        # root-blend faces (0.25 < |n_theta| <= 0.5) only next to the blade (<= 4 mm from the camber line):
        # hub-floor ridges between the blades are not blade sides
        nb = (np.abs(nt[kb]) > 0.5) | (np.abs(dl) * r[kb] < 4.0)
        kb, p, v, dl, side, thc = kb[nb], p[nb], v[nb], dl[nb], side[nb], thc[nb]
        inner = (p > 0.03) & (p < 0.97) & (v > 0.0) & (v < 0.92)
        fits, fstats = _fit_sides(p, v, dl, side, r[kb], inner)
        log(f"[impeller] pass {it}: N={N} hub rms {hrms:.4f} shroud rms {srms:.4f} z_le {z_le:.3f} r_te {r_te:.3f} sides {fstats}")
        if it == n_axis_iter:
            break
        # axis correction from the blades' tangential residuals (a blade pattern is exactly N-fold about
        # the true axis; an offset/tilted axis shows as a once-around tangential shift)
        rows, rhs = [], []
        for sg in (1, -1):
            m = (side == sg) & (np.sign(dl) == sg) & inner
            res = (dl[m] - fits[sg](p[m], v[m])) * r[kb][m]
            ok = np.abs(res) < 0.6
            T_ = th[kb][m]; et = np.c_[-np.sin(T_), np.cos(T_)]
            rows.append(np.c_[et, et * z[kb][m][:, None]][ok]); rhs.append(res[ok])
        X = np.vstack(rows); y = np.concatenate(rhs)
        sol = np.linalg.lstsq(X, y, rcond=None)[0]
        c0, al = 0.8 * sol[:2], 0.8 * sol[2:]     # damped: the camber map moves with the axis
        info["axis_corrections"].append({"shift_mm": [round(float(x), 4) for x in c0],
                                         "tilt_mrad": [round(1e3 * float(x), 4) for x in al]})
        u = F.u + al[0] * F.e1 + al[1] * F.e2; u /= np.linalg.norm(u)
        a0 = F.a0 + c0[0] * F.e1 + c0[1] * F.e2
        e1, e2 = _basis(u)
        F = Frame(a0, u, e1, e2)
        if np.hypot(*c0) < 0.003 and np.hypot(*al) < 2e-5:
            break
    spec = Spec(frame=F, N=N, hub=H, shroud=Sh, hub_spline=hsp, shroud_spline=ssp, hub_t=(T.min(), T.max()),
                z_le=z_le, r_te=r_te, camber=camx, sides=fits)
    # leading-edge round in the eye plane: blade half-width vs depth below the LE plane
    hs = Sh.at(0.0)[0]; hr0 = H.at(0.0)[0]
    kq = (z > z_le - 1.6) & (z < z_le + 0.4) & (r > hr0 + 0.25 * (hs - hr0)) & (r < hr0 + 0.85 * (hs - hr0))
    pq, vq = mm.pv(r[kq], np.minimum(z[kq], z_le - 0.05)); okq = np.isfinite(pq)
    dq = np.angle(np.exp(1j * N * (th[kq][okq] - camx(np.clip(pq[okq], 0, 1), np.clip(vq[okq], 0, 1))))) / N * r[kq][okq]
    zq = z[kq][okq]; dep, hw = [], []
    for z0 in np.arange(z_le - 1.6, z_le + 0.3, 0.1):
        q = (zq > z0) & (zq < z0 + 0.1) & (np.abs(dq) < 2.5)
        if q.sum() > 60:
            a, b = np.percentile(dq[q], [2, 98]); dep.append(z_le - (z0 + 0.05)); hw.append((b - a) / 2)
    le = edge_round(dep, hw) if len(dep) >= 6 else None
    info["le_round"] = le
    spec.le_round = le
    spec.eye = None if eye is None else {"z_front": eye["z_front"], "r_cyl": eye["r_cyl"], "kind": eye.get("kind", "stub_cylinder")}
    info.update({"eye": spec.eye, "hub_rms": hrms, "shroud_rms": srms, "side_fit": fstats, "rim_radius": R})
    try:
        spec.floor = fit_floor(spec, d)
        info["floor"] = {k_: spec.floor[k_] for k_ in ("cells_with_floor", "h_range")}
    except Exception as e:  # noqa: BLE001
        spec.floor = None; info["floor"] = {"error": str(e)}
    spec._pts = dict(kb=kb, p=p, v=v, dl=dl, side=side, d=d)
    try:
        fam = blade_families(spec, th[kb], r[kb])
    except Exception as e:  # noqa: BLE001
        fam = {"error": f"{type(e).__name__}: {e}"}
    spec.families = fam if fam.get("splitter") else None
    spec.sides_splitter = None
    if spec.families:
        # full blades and splitters are different blades (Cummins 140: 2.75 vs 2.15 mm thick at mid
        # span): one side fit over both leaves a +-0.3 mm bimodal residual (rms 0.34 mm).  Each family
        # gets its own side fit; the splitter's only over its own extent (beyond p_split).
        try:
            par = fam["splitter_parity"]
            thc_ = camx(np.clip(p, -0.05, 1.05), np.clip(v, -0.15, 1.1))
            k_ = np.rint((th[kb] - thc_ - dl) * N / (2 * np.pi)).astype(int) % N
            isS = (k_ % 2) == par; rr_ = r[kb]
            ff, fst = _fit_sides(p[~isS], v[~isS], dl[~isS], side[~isS], rr_[~isS], inner[~isS])
            ps0 = min(0.5, fam["p_split"] + 0.02)
            sf, sst = _fit_sides(p[isS], v[isS], dl[isS], side[isS], rr_[isS], inner[isS] & (p[isS] > ps0),
                                 box=(max(0.03, ps0), 0.97, 0.0, 0.92))
            spec.sides = ff; spec.sides_splitter = sf
            info["side_fit_mixed"] = fstats
            info["side_fit"] = fst; info["side_fit_splitter"] = sst
            fam["scatter_own_fit"] = {"full": {str(k): {"rms_mm": fst[k]["rms_mm"], "p95_mm": fst[k]["p95_mm"]} for k in fst},
                                      "splitter": {str(k): {"rms_mm": sst[k]["rms_mm"], "p95_mm": sst[k]["p95_mm"]} for k in sst}}
            log(f"[impeller] per-family side fits: full {fst} splitter {sst}")
        except Exception as e:  # noqa: BLE001
            spec.sides_splitter = None
            info["side_fit_families_error"] = f"{type(e).__name__}: {e}"
    info["families"] = fam
    spec.info = info
    return spec


def blade_le_per_sector(blade_idx, p, v, N, v_band=(0.2, 0.8), q=2.0, min_n=40):
    """Leading-edge position (meridional p, 2nd percentile of the blade's side points in the mid-span band)
    of every blade 0..N-1; nan where a blade has too few points."""
    m = (v > v_band[0]) & (v < v_band[1])
    out = np.full(N, np.nan)
    for k in range(N):
        q_ = m & (blade_idx == k)
        if q_.sum() >= min_n:
            out[k] = np.percentile(p[q_], q)
    return out


def classify_families(le, min_gap=0.03, min_sep_frac=0.5):
    """Per-sector LE test: full + splitter alternate when the even and odd blades' LE positions form two
    non-overlapping groups whose medians are at least min_gap (fraction of the meridional length) apart.  Returns a dict; 'splitter'
    is True with splitter_parity (0/1), p_split and per-family stats."""
    le = np.asarray(le, float); N = len(le)
    out = {"le_p_per_blade": [None if not np.isfinite(x) else round(float(x), 4) for x in le], "splitter": False}
    if N < 6 or N % 2 or np.isfinite(le).sum() < 0.75 * N:
        out["reason"] = "too few blades with LE data or odd N"
        return out
    ev, od = le[0::2], le[1::2]
    me, mo = float(np.nanmedian(ev)), float(np.nanmedian(od))
    se, so = float(np.nanmax(np.abs(ev - me))), float(np.nanmax(np.abs(od - mo)))
    gap = abs(mo - me)
    out.update({"le_p_even": round(me, 4), "le_p_odd": round(mo, 4), "spread_even": round(se, 4),
                "spread_odd": round(so, 4), "gap": round(gap, 4)})
    # the two groups must not overlap: the earliest LE of the later group lies clear of the latest LE of the
    # earlier group (Cummins 140: even 0.057..0.089, odd 0.012..0.021 -> separation 0.036 of a 0.057 gap)
    late, early = (od, ev) if mo > me else (ev, od)
    sep = float(np.nanmin(late) - np.nanmax(early))
    out["separation"] = round(sep, 4)
    # clear of the tight (full-blade, sharp LE) group by >= 3x its spread: the splitters' LE p is the
    # noisy one (rounded/short LE: Cummins 140 re-analysed even 0.046..0.148, odd 0.012..0.021 -> sep
    # 0.025 of a 0.068 gap, which a fraction-of-gap rule rejected)
    s_early = se if mo > me else so
    sep_min = max(0.5 * min_gap, min(min_sep_frac * gap, 3.0 * s_early))
    out["separation_required"] = round(sep_min, 4)
    if gap >= min_gap and sep >= sep_min:
        par = 1 if mo > me else 0
        out.update({"splitter": True, "splitter_parity": par, "p_split": round(max(me, mo), 4),
                    "n_full": N // 2, "n_splitter": N // 2})
    else:
        out["reason"] = "single family (LE positions do not alternate)"
    return out


def blade_families(spec, th_kb, r_kb):
    """Two blade families (full + splitter) from the analysed side points: blade index from the folded
    phase, LE per blade (classify_families), and per-family scatter of the side-fit residuals."""
    P = spec._pts; p, v, dl, side = P["p"], P["v"], P["dl"], P["side"]
    N = spec.N
    thc = spec.camber(np.clip(p, -0.05, 1.05), np.clip(v, -0.15, 1.1))
    k = np.rint((th_kb - thc - dl) * N / (2 * np.pi)).astype(int) % N
    le = blade_le_per_sector(k, p, v, N)
    fam = classify_families(le)
    inner = (p > 0.03) & (p < 0.97) & (v > 0.0) & (v < 0.92)
    res = np.full(p.shape, np.nan)
    for sg in (1, -1):
        m = (side == sg) & inner
        res[m] = (dl[m] - spec.sides[sg](p[m], v[m])) * r_kb[m]
    sc = {}
    groups = {"all": np.ones(N, bool)}
    if fam.get("splitter"):
        par = fam["splitter_parity"]
        groups = {"full": (np.arange(N) % 2) != par, "splitter": (np.arange(N) % 2) == par}
        # only the splitters' own extent counts for both (upstream of p_split only full blades exist)
        inner = inner & (p > fam["p_split"] + 0.02)
    for name, gk in groups.items():
        m = inner & gk[k] & np.isfinite(res) & (np.abs(res) < 0.6)
        if m.sum() > 50:
            sc[name] = {"n": int(m.sum()), "rms_mm": round(float(np.sqrt(np.mean(res[m] ** 2))), 4),
                        "p95_mm": round(float(np.percentile(np.abs(res[m]), 95)), 4)}
    fam["scatter"] = sc
    return fam


# ------------------------------------------------------------------ B-rep
def _P(x):
    from OCP.gp import gp_Pnt
    return gp_Pnt(*map(float, x))


def _bspline_curve(P3, tol=1e-4):
    from OCP.collections import Array1_gp_Pnt as TColgp_Array1OfPnt
    from OCP.GeomAPI import GeomAPI_PointsToBSpline
    from OCP.GeomAbs import GeomAbs_C2
    arr = TColgp_Array1OfPnt(1, len(P3))
    for i, p in enumerate(P3):
        arr.SetValue(i + 1, _P(p))
    return GeomAPI_PointsToBSpline(arr, 3, 8, GeomAbs_C2, tol).Curve()


def _bspline_surface(G, tol=2e-3):
    """G: (nu, nv, 3) grid of points -> approximating B-spline surface (degree 3..8, C2).  Approximation of
    points that lie on a smooth surface keeps the control net next to the data (bounded poles).  The
    approximation can still diverge on some grids (wild poles between the data): the surface is sampled
    and checked against the grid's bounding box, and refitted with lower degree / looser settings."""
    from OCP.collections import Array2_gp_Pnt as TColgp_Array2OfPnt
    from OCP.GeomAPI import GeomAPI_PointsToBSplineSurface
    from OCP.GeomAbs import GeomAbs_C2, GeomAbs_C1
    nu, nv, _ = G.shape
    arr = TColgp_Array2OfPnt(1, nu, 1, nv)
    for i in range(nu):
        for j in range(nv):
            arr.SetValue(i + 1, j + 1, _P(G[i, j]))
    lo, hi = G.reshape(-1, 3).min(0), G.reshape(-1, 3).max(0)
    margin = max(0.5, 0.02 * float(np.linalg.norm(hi - lo)))
    last = None
    for dmin, dmax, cont, tf in ((3, 8, GeomAbs_C2, 1.0), (3, 3, GeomAbs_C2, 1.0), (3, 5, GeomAbs_C2, 3.0),
                                 (3, 3, GeomAbs_C1, 5.0)):
        sf = GeomAPI_PointsToBSplineSurface(arr, dmin, dmax, cont, tol * tf).Surface()
        last = sf
        u1, u2, v1, v2 = sf.Bounds()
        Q = np.array([[sf.Value(u, v).X(), sf.Value(u, v).Y(), sf.Value(u, v).Z()]
                      for u in np.linspace(u1, u2, 3 * nu) for v in np.linspace(v1, v2, 3 * nv)])
        if np.all(Q.min(0) > lo - margin) and np.all(Q.max(0) < hi + margin):
            return sf
    return last


def _edge_poly(P3):
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
    return BRepBuilderAPI_MakeEdge(_bspline_curve(P3)).Edge()


def _seg(a, b):
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
    return BRepBuilderAPI_MakeEdge(_P(a), _P(b)).Edge()


def _revolve_profile(F, edges2d):
    """edges2d: list of ('line', (r,z), (r,z)) or ('curve', array (n,2) r,z) forming a closed loop in the
    meridional half plane theta = 0; revolved 360 deg about the frame axis."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeFace
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeRevol
    from OCP.gp import gp_Ax1, gp_Dir
    w = BRepBuilderAPI_MakeWire()
    for e in edges2d:
        if e[0] == "line":
            a, b = (F.world(e[1][0], 0.0, e[1][1]), F.world(e[2][0], 0.0, e[2][1]))
            w.Add(_seg(a, b))
        else:
            Q = np.array([F.world(r, 0.0, z) for r, z in e[1]])
            w.Add(_edge_poly(Q))
    if not w.IsDone():
        raise RuntimeError("profile wire not closed")
    f = BRepBuilderAPI_MakeFace(w.Wire(), True)
    if not f.IsDone():
        raise RuntimeError("profile face failed")
    return BRepPrimAPI_MakeRevol(f.Face(), gp_Ax1(_P(F.a0), gp_Dir(*map(float, F.u))), 2 * math.pi).Shape()


def _rz_poly(curve: MerCurve, p0, p1, n=200):
    p = np.linspace(p0, p1, n)
    R, Z = curve.at(p)
    return np.c_[R, Z]


def _le_round_offset(h, rho, a_max=75.0):
    """Inward offset (mm, <= 0) of each blade side that rounds the edge where the side meets the eye
    plane: h = depth below the plane.  Circle of radius rho tangent to the side at h = rho; within the
    last few % near the plane (where the circle turns horizontal) continued with the slope at a_max deg,
    so the side sheet stays a graph of h."""
    h = np.asarray(h, float)
    hc = rho * (1 - np.sin(np.radians(a_max)))
    g = np.where(h >= rho, 0.0, -rho + np.sqrt(np.clip(rho ** 2 - (rho - np.clip(h, hc, rho)) ** 2, 0, None)))
    k = np.tan(np.radians(a_max))
    return np.where(h < hc, g - k * (hc - h), g)


def _grid_folds(G):
    """Number of places where consecutive point steps along the first grid direction reverse."""
    du = np.diff(G, axis=0)
    return int((np.einsum("ijk,ijk->ij", du[1:], du[:-1]) < 0).sum())


def _tip_continuation(fn, P, V, vh=0.9, w=0.1, sigma=2.0):
    """fn(p, v) on the (P, V) grid, with v > vh replaced by a linear continuation whose slope is the
    secant over [vh - w, vh], Gaussian-smoothed along p."""
    from scipy.ndimage import gaussian_filter1d
    p = P[:, 0]
    a = fn(p, np.full_like(p, vh)); b = fn(p, np.full_like(p, vh - w))
    sl = gaussian_filter1d((a - b) / w, sigma, mode="nearest")
    out = fn(P.ravel(), V.ravel()).reshape(P.shape)
    hi = V > vh
    out = np.where(hi, a[:, None] + sl[:, None] * (V - vh), out)
    return out.ravel()


def blade_slab(spec: Spec, pr=(-0.05, 1.05), vr=(-0.15, 1.10), n=(61, 31), le_rho=None, le_apply=True):
    """Closed solid between the two fitted side surfaces, extended past LE, TE, tip and into the hub
    (trimmed later).  The four closing faces are ruled between the side surfaces' boundary curves."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_Sewing, BRepBuilderAPI_MakeSolid
    from OCP.BRepFill import BRepFill
    from OCP.TopoDS import TopoDS
    from OCP.ShapeFix import ShapeFix_Solid
    F = spec.frame; mm = spec.mermap()
    pn = np.linspace(*pr, n[0])
    if le_rho:
        # LE round built into the side sheets: extra p rows through the round (depth -0.3 .. 1.5 rho)
        pv = np.linspace(0, 1, 9)
        _, z0 = mm.rz(np.zeros(9), pv); _, z1 = mm.rz(np.full(9, 0.01), pv)
        dzdp = float(np.median((z0 - z1) / 0.01))                      # depth per unit p at the LE
        extra = np.linspace(-0.3, 1.5, 19) * le_rho / max(dzdp, 1e-6)
        pn = np.unique(np.r_[pn[np.min(np.abs(pn[:, None] - extra[None, :]), 1) > 0.3 * (extra[1] - extra[0])], extra])
    n = (len(pn), n[1])
    P, V = np.meshgrid(pn, np.linspace(*vr, n[1]), indexing="ij")
    R, Z = mm.rz(P.ravel(), V.ravel())
    thc = spec.camber(P.ravel(), V.ravel())
    tp = spec.sides[1](P.ravel(), V.ravel()); tm = spec.sides[-1](P.ravel(), V.ravel())
    if _grid_folds(F.world(R, thc + tp, Z).reshape(n[0], n[1], 3)) or _grid_folds(F.world(R, thc + tm, Z).reshape(n[0], n[1], 3)):
        # a fold (the point rows backtrack) is never real geometry: it comes from the noisy edge
        # derivative used to continue the fits past the tip; continue them from v = 0.9 with a secant
        # slope smoothed along p instead (only done when a fold is found, so clean grids are unchanged)
        thc, tp, tm = (_tip_continuation(fn, P, V) for fn in (spec.camber, spec.sides[1], spec.sides[-1]))
    if le_rho and le_apply:
        g = _le_round_offset(spec.z_le - Z, le_rho) / R                # angle, <= 0
        tp = tp + g; tm = tm - g
    # the extensions beyond the data (tip, LE/TE overhang) are linear and can make the two sides cross;
    # keep at least 30% of the in-data median thickness (those parts are trimmed away anyway)
    inside = (P.ravel() >= 0) & (P.ravel() <= 1) & (V.ravel() >= 0) & (V.ravel() <= 1)
    t_min = 0.3 * float(np.median(((tp - tm) * R)[inside]))
    short = (tp - tm) * R < t_min
    if short.any():
        mid = 0.5 * (tp + tm); half = 0.5 * t_min / R
        tp = np.where(short, mid + half, tp); tm = np.where(short, mid - half, tm)
    grids = {}
    for sg, off in ((1, tp), (-1, tm)):
        grids[sg] = F.world(R, thc + off, Z).reshape(n[0], n[1], 3)
    return _slab_from_grids(grids[1], grids[-1]), grids


def _slab_from_grids(gA, gB):
    """Closed solid between two (nu, nv, 3) point grids: two B-spline sheets and four ruled closing faces
    between their boundary iso-curves, sewn; oriented so that infinity is outside."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_Sewing, BRepBuilderAPI_MakeSolid
    from OCP.BRepFill import BRepFill
    from OCP.TopoDS import TopoDS
    from OCP.ShapeFix import ShapeFix_Solid
    grids = {1: gA, -1: gB}
    surfs = {sg: _bspline_surface(grids[sg]) for sg in grids}
    faces = [BRepBuilderAPI_MakeFace(surfs[sg], 1e-6).Face() for sg in (1, -1)]
    # closing faces: ruled between the two surfaces' own boundary iso-curves (exactly their face edges)
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
    isos = {}
    for sg in (1, -1):
        u1, u2, v1, v2 = surfs[sg].Bounds()
        isos[sg] = [surfs[sg].UIso(u1), surfs[sg].UIso(u2), surfs[sg].VIso(v1), surfs[sg].VIso(v2)]
    for c1, c2 in zip(isos[1], isos[-1]):
        faces.append(BRepFill.Face_s(BRepBuilderAPI_MakeEdge(c1).Edge(), BRepBuilderAPI_MakeEdge(c2).Edge()))
    sew = BRepBuilderAPI_Sewing(5e-3)
    for f in faces:
        sew.Add(f)
    sew.Perform()
    sh = sew.SewedShape()
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_SHELL
    ex = TopExp_Explorer(sh, TopAbs_SHELL)
    shell = TopoDS.Shell_s(ex.Current()) if hasattr(TopoDS, "Shell_s") else TopoDS.Shell(ex.Current())
    ms = BRepBuilderAPI_MakeSolid(shell)
    fx = ShapeFix_Solid(ms.Solid()); fx.Perform()
    sol = fx.Solid()
    # orientation: a solid whose shell normals point inwards is the complement of the blade (booleans
    # with it come out empty).  The signed volume is the reliable test for a closed shell: the
    # infinite-point classifier was seen to answer OUT for an inverted 8+8 splitter slab (cummins_140,
    # volume -4004 mm3 -> blade trim 0.04 mm3).  The classifier is only the tie-break for a ~zero volume.
    from OCP.BRepClass3d import BRepClass3d_SolidClassifier
    from OCP.TopAbs import TopAbs_IN
    from . import occ_util as U
    vol = U.volume(sol)
    if abs(vol) > 1e-9:
        flip = vol < 0
    else:
        cl = BRepClass3d_SolidClassifier(sol); cl.PerformInfinitePoint(1e-6)
        flip = cl.State() == TopAbs_IN
    if flip:
        sol = TopoDS.Solid_s(sol.Reversed()) if hasattr(TopoDS, "Solid_s") else TopoDS.Solid(sol.Reversed())
    return sol



def U_volume(s):
    from . import occ_util as U
    return U.volume(s)


def _floor_eval(fl):
    """h(p, phi) callable from the floor grid (periodic in phi, held constant beyond p ends)"""
    from scipy.interpolate import RectBivariateSpline
    pc, fc, H = fl["pc"], fl["fc"], fl["h"]
    w = 4
    fce = np.r_[fc[-w:] - 1, fc, fc[:w] + 1]; He = np.c_[H[:, -w:], H, H[:, :w]]
    sp = RectBivariateSpline(pc, fce, He, kx=3, ky=3, s=0)
    return lambda p, f: sp.ev(np.clip(p, pc[0], pc[-1]), np.mod(f, 1.0))


def floor_slab(spec, delta, depth=0.4, pr=(-0.03, 1.03), fr=(0.02, 0.98), n=(64, 60)):
    """One passage of the hub floor as a solid: top = fitted floor h(p, phi), bottom = delta + depth below
    the revolved hub (inside the lowered hub).  Trimmed later by the blade envelope.  The phi range stops
    short of the camber lines (inside the blade roots), so neighbouring slabs neither overlap nor touch."""
    F = spec.frame; mm = spec.mermap(); fl = spec.floor; N = spec.N
    h = _floor_eval(fl)
    P, Fi = np.meshgrid(np.linspace(*pr, n[0]), np.linspace(*fr, n[1]), indexing="ij")
    p = P.ravel(); fi = Fi.ravel()
    hr, hz = spec.hub.at(np.clip(p, 0, 1)); sr, sz = spec.shroud.at(np.clip(p, 0, 1)); L = np.hypot(sr - hr, sz - hz)
    th = spec.camber(np.clip(p, 0, 1), np.full_like(p, fl["v_cam"])) + fi * 2 * np.pi / N
    top = np.maximum(h(p, fi), -0.5 * delta)          # never below the lowered hub surface
    grids = []
    for vv in (top / L, -(delta + depth) / L):
        R, Z = mm.rz(p, vv)
        grids.append(F.world(R, th, Z).reshape(n[0], n[1], 3))
    return _slab_from_grids(grids[0], grids[1])


def _round_plane_edges(shape, F, z0, rho, log=print, min_len=3.0, timeout=120):
    """Fillet the long edges of a blade lying in the axial plane z = z0 (the LE / eye cut) with radius
    rho.  Returns (shape, info); the unfilleted shape is returned if the fillet fails or is invalid."""
    from OCP.BRepFilletAPI import BRepFilletAPI_MakeFillet
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.GCPnts import GCPnts_AbscissaPoint
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_EDGE
    from OCP.TopoDS import TopoDS
    picked = []
    ex = TopExp_Explorer(shape, TopAbs_EDGE); n = 0; seen = []
    while ex.More():
        e = TopoDS.Edge_s(ex.Current()) if hasattr(TopoDS, "Edge_s") else TopoDS.Edge(ex.Current())
        ex.Next()
        if any(e.IsSame(s) for s in seen):
            continue
        seen.append(e)
        c = BRepAdaptor_Curve(e); a, b = c.FirstParameter(), c.LastParameter()
        P = np.array([[c.Value(t).X(), c.Value(t).Y(), c.Value(t).Z()] for t in np.linspace(a, b, 9)])
        _, _, zz = F.local(P)
        L = GCPnts_AbscissaPoint.Length_s(c)
        if np.all(np.abs(zz - z0) < 1e-3) and L > min_len:
            rr, _, _ = F.local(P)
            if rr.max() - rr.min() > 0.5 * L:              # radial-running edge: a blade side meets the cut
                picked.append(e); n += 1
    info = {"rho": rho, "edges": n, "ok": False}
    if n == 0:
        return shape, info
    from .occ_child import fillet_in_child
    try:
        res = fillet_in_child(shape, [(e, rho) for e in picked], timeout)
        if res is not None and not res.IsNull() and BRepCheck_Analyzer(res).IsValid():
            info["ok"] = True
            log(f"[impeller] LE round r={rho:.3f} on {n} edges")
            return res, info
    except TimeoutError:
        info["error"] = f"timeout {timeout}s"
    except Exception as ex_:  # noqa: BLE001
        info["error"] = str(ex_)
    log(f"[impeller] LE round failed ({info.get('error', 'not done/invalid')}); kept sharp")
    return shape, info


def _common(a, b, fuzzy=(0.0, 0.01, 0.03)):
    """BRepAlgoAPI_Common, retried with a fuzzy value when the exact boolean comes back empty although the
    arguments overlap (165 mm 15-blade wheel: the slab and the blade envelope meet in near-coincident
    geometry - the tip curve ends 0.066 mm under the LE plane - and the exact Common returned nothing, so the
    wheel was built as a bare hub, 9.3 mm rms).  0.01 mm is below the scan noise."""
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Common
    from OCP.collections import List_TopoDS_Shape
    out = None
    for fz in fuzzy:
        if fz == 0.0:
            out = BRepAlgoAPI_Common(a, b).Shape()
        else:
            op = BRepAlgoAPI_Common(); la = List_TopoDS_Shape(); la.Append(a); lt = List_TopoDS_Shape(); lt.Append(b)
            op.SetArguments(la); op.SetTools(lt); op.SetFuzzyValue(fz); op.Build()
            out = op.Shape()
        if out is not None and not out.IsNull() and U_volume(out) > 1e-6:
            _COMMON_LAST["method"] = "exact" if fz == 0.0 else f"fuzzy {fz:g}"
            return out
    # last resort: a AND b = a - (box - b).  Cummins 140 (1.3 mm CAD facets): the slab and the envelope
    # overlap (slab mid-thickness points classify IN the envelope, slab x cylinder works) but every
    # Common variant came back empty; the complement cut gives 5086 of 5320 mm3 (the parts above the
    # tip line / past the LE and TE removed), matching the per-ring commons that did work
    try:
        from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
        from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
        from OCP.Bnd import Bnd_Box
        from OCP.BRepBndLib import BRepBndLib
        from OCP.BRepCheck import BRepCheck_Analyzer
        from OCP.gp import gp_Pnt
        bb = Bnd_Box(); BRepBndLib.Add_s(a, bb); BRepBndLib.Add_s(b, bb)
        mn, mx = bb.CornerMin(), bb.CornerMax(); pad = 10.0
        box = BRepPrimAPI_MakeBox(gp_Pnt(mn.X() - pad, mn.Y() - pad, mn.Z() - pad),
                                  gp_Pnt(mx.X() + pad, mx.Y() + pad, mx.Z() + pad)).Shape()
        comp = BRepAlgoAPI_Cut(box, b).Shape()
        alt = BRepAlgoAPI_Cut(a, comp).Shape()
        va = U_volume(alt)
        if (alt is not None and not alt.IsNull() and 1e-6 < va <= U_volume(a) * (1 + 1e-6) + 1e-6
                and va <= U_volume(b) * (1 + 1e-6) + 1e-6 and BRepCheck_Analyzer(alt).IsValid()):
            _COMMON_LAST["method"] = "complement cut"
            return alt
    except Exception:  # noqa: BLE001
        pass
    _COMMON_LAST["method"] = "empty"
    return out


_COMMON_LAST = {"method": None}


LE_ROUND_BUDGET_S = 300.0          # overall wall-clock cap on the LE rounding attempts per build
_LE_LAST_OK = {"method": None}     # last successful LE rounding method (tried first next time)


def _blade(spec, env, rb, le_round=True, log=print):
    """Blade solid = slab trimmed by the envelope, with the leading-edge round in the eye plane.
    The round may be put on the slab cut only by the eye plane: there the two LE edges run across the
    whole (extended) blade and end on the slab's closing faces outside the part, instead of ending in the
    root blend against the hub where constant-radius fillets fail.  Then the rounded slab is trimmed.
    Order: round after the trim (works where the edges end cleanly), then on the eye-plane-cut slab,
    then sharp.  le_info records the method and every attempt."""
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Common
    from OCP.BRepCheck import BRepCheck_Analyzer
    F = spec.frame
    slab, _ = blade_slab(spec)
    sv = {"slab_valid": bool(BRepCheck_Analyzer(slab).IsValid())}
    le = getattr(spec, "le_round", None)
    if not (le_round and le and le["rho"] > 0.1):
        return _common(slab, env), sv | {"ok": False, "method": "none",
                                                            "reason": "no LE round estimated" if le_round else "disabled"}
    rho = min(le["rho"], 0.9 * le["half_t"])
    attempts = []
    big = 2.0 * spec.r_te
    blade = _common(slab, env)
    import time as _time
    t0 = _time.time()
    budget = float(getattr(spec, "le_budget_s", LE_ROUND_BUDGET_S))

    def over():
        return _time.time() - t0 > budget

    def post_trim():
        for f_ in (1.0, 0.75, 0.5):
            if over():
                return None
            b_, inf = _round_plane_edges(blade, F, spec.z_le, f_ * rho, log, timeout=60)
            attempts.append({"stage": "post_trim"} | inf)
            if inf.get("ok"):
                return b_, sv | inf | {"method": "post_trim", "attempts": attempts}
        return None

    def sheet_round():
        # round built into the side sheets (no OCC fillet): slab rebuilt with the sides pulled in near the
        # eye plane, then trimmed; accepted if valid and its volume is a little below the sharp blade's
        from OCP.BRepCheck import BRepCheck_Analyzer as _BA
        for f_ in (1.0, 0.75, 0.5):
            if over():
                return None
            try:
                v_sharp = U_volume(_common(blade_slab(spec, le_rho=f_ * rho, le_apply=False)[0], env))
                slab_r, _ = blade_slab(spec, le_rho=f_ * rho)
                b_ = _common(slab_r, env)
                ok = (not b_.IsNull()) and _BA(slab_r).IsValid() and _BA(b_).IsValid() and 0.95 * v_sharp < U_volume(b_) < v_sharp
            except Exception as ex_:  # noqa: BLE001
                attempts.append({"stage": "sheet_round", "rho": f_ * rho, "ok": False, "error": str(ex_)})
                continue
            attempts.append({"stage": "sheet_round", "rho": f_ * rho, "ok": bool(ok),
                             "volume": U_volume(b_) if b_ is not None and not b_.IsNull() else None, "volume_sharp": v_sharp})
            if ok:
                log(f"[impeller] LE round r={f_ * rho:.3f} built into the side sheets")
                return b_, sv | {"rho": f_ * rho, "ok": True, "method": "sheet_round", "rho_estimated": rho, "attempts": attempts}
        return None

    def per_edge():
        # one edge at a time, with a radius ladder down to 0.25 x (the walker's success is per edge/radius)
        if over():
            return None
        b_, inf = _round_edges_one_by_one(blade, F, spec.z_le, [f_ * rho for f_ in (1.0, 0.5, 0.25)], log)
        attempts.append({"stage": "per_edge"} | inf)
        if inf.get("ok"):
            return b_, sv | inf | {"method": "per_edge", "rho_estimated": rho, "attempts": attempts}
        return None

    def pre_trim():
        if over():
            return None
        cut = _revolve_profile(F, [("line", (0.1, -50.0), (big, -50.0)), ("line", (big, -50.0), (big, spec.z_le)),
                                   ("line", (big, spec.z_le), (0.1, spec.z_le)), ("line", (0.1, spec.z_le), (0.1, -50.0))])
        slab_c = _common(slab, cut)
        for f_ in (1.0, 0.5):
            if over():
                return None
            s_, inf = _round_plane_edges(slab_c, F, spec.z_le, f_ * rho, log, timeout=60)
            attempts.append({"stage": "pre_trim"} | inf)
            if inf.get("ok"):
                b = _common(s_, env)
                if not b.IsNull() and BRepCheck_Analyzer(b).IsValid() and U_volume(b) > 0:
                    return b, sv | inf | {"method": "pre_trim", "attempts": attempts}
                attempts[-1]["trim_failed"] = True
        return None

    stages = {"post_trim": post_trim, "sheet_round": sheet_round, "per_edge": per_edge, "pre_trim": pre_trim}
    order = list(stages)
    if _LE_LAST_OK["method"] in stages:                 # the method that worked last time goes first
        order.remove(_LE_LAST_OK["method"]); order.insert(0, _LE_LAST_OK["method"])
    for name in order:
        res = stages[name]()
        if res is not None:
            _LE_LAST_OK["method"] = name
            res[1]["order"] = order
            res[1]["seconds"] = round(_time.time() - t0, 1)
            return res
    capped = over()
    log("[impeller] LE round: " + ("time budget used up" if capped else "all attempts failed") + "; LE kept sharp (flagged in the report)")
    return blade, sv | {"rho": rho, "ok": False, "method": "sharp_fallback", "attempts": attempts, "order": order,
                        "budget_s": budget, "budget_exceeded": bool(capped), "seconds": round(_time.time() - t0, 1)}


def _le_edges(shape, F, z0, min_len=3.0):
    """Long radial-running edges lying in the axial plane z = z0, with their midpoints."""
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.GCPnts import GCPnts_AbscissaPoint
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_EDGE
    from OCP.TopoDS import TopoDS
    out = []; ex = TopExp_Explorer(shape, TopAbs_EDGE)
    while ex.More():
        e = TopoDS.Edge(ex.Current()); ex.Next()
        if any(e.IsSame(s) for s, _ in out):
            continue
        c = BRepAdaptor_Curve(e); a, b = c.FirstParameter(), c.LastParameter()
        P = np.array([[c.Value(t).X(), c.Value(t).Y(), c.Value(t).Z()] for t in np.linspace(a, b, 9)])
        rr, _, zz = F.local(P)
        L = GCPnts_AbscissaPoint.Length_s(c)
        if np.all(np.abs(zz - z0) < 1e-3) and L > min_len and rr.max() - rr.min() > 0.5 * L:
            out.append((e, P[4]))
    return out


def _round_edges_one_by_one(shape, F, z0, rhos, log=print, timeout=60):
    """Fillet the LE edges one at a time, each with the first radius of the ladder that OCC builds
    (the fillet walker succeeds or fails per edge and per radius on freeform sides).  Edges are found
    again by midpoint after each fillet.  ok only if every edge got a round."""
    from OCP.BRepFilletAPI import BRepFilletAPI_MakeFillet
    from OCP.BRepCheck import BRepCheck_Analyzer
    mids = [m for _, m in _le_edges(shape, F, z0)]
    info = {"edges": len(mids), "rho_per_edge": [], "ok": False}
    if not mids:
        return shape, info
    cur = shape
    for m in mids:
        cand = [(e, mm) for e, mm in _le_edges(cur, F, z0)]
        if not cand:
            return shape, info
        e = min(cand, key=lambda x: np.linalg.norm(x[1] - m))
        if np.linalg.norm(e[1] - m) > 0.5:
            return shape, info
        done = None
        for r in rhos:
            from .occ_child import fillet_in_child
            try:
                res = fillet_in_child(cur, [(e[0], r)], timeout)
            except Exception:  # noqa: BLE001  (timeout or child failure: next radius)
                res = None
            if res is not None and not res.IsNull() and BRepCheck_Analyzer(res).IsValid():
                done = (res, r); break
        if done is None:
            return shape, info
        cur = done[0]; info["rho_per_edge"].append(round(float(done[1]), 4))
    info["ok"] = True
    info["rho"] = float(np.mean(info["rho_per_edge"]))
    log(f"[impeller] LE round per edge r={info['rho_per_edge']}")
    return cur, info


def _in_child(fn, timeout):
    """Run fn() (returning a TopoDS_Shape or None) in a forked child with a time limit; the shape comes
    back as a BRep file.  OCC operations such as fillets can hang on hard input."""
    import os, tempfile, time, signal
    from OCP.BRepTools import BRepTools
    from OCP.BRep import BRep_Builder
    from OCP.TopoDS import TopoDS_Shape
    fd, path = tempfile.mkstemp(suffix=".brep"); os.close(fd)
    pid = os.fork()
    if pid == 0:                                   # child
        code = 1
        try:
            sh = fn()
            if sh is not None:
                BRepTools.Write_s(sh, path); code = 0
            else:
                code = 2
        finally:
            os._exit(code)
    t0 = time.time()
    while True:
        done, status = os.waitpid(pid, os.WNOHANG)
        if done:
            break
        if time.time() - t0 > timeout:
            os.kill(pid, signal.SIGKILL); os.waitpid(pid, 0); os.unlink(path)
            raise TimeoutError
        time.sleep(0.1)
    code = os.waitstatus_to_exitcode(status)
    if code != 0:
        os.unlink(path)
        return None
    sh = TopoDS_Shape(); BRepTools.Read_s(sh, path, BRep_Builder()); os.unlink(path)
    return sh


def _pattern_indexed(ax, N, shape_for_k):
    """Compound of shape_for_k(k) rotated by 2 pi k / N about ax, k = 0..N-1."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCP.gp import gp_Trsf
    from OCP.TopoDS import TopoDS_Compound
    from OCP.BRep import BRep_Builder
    bb = BRep_Builder(); comp = TopoDS_Compound(); bb.MakeCompound(comp)
    for k in range(N):
        tr = gp_Trsf(); tr.SetRotation(ax, 2 * math.pi * k / N)
        bb.Add(comp, BRepBuilderAPI_Transform(shape_for_k(k), tr, True).Shape())
    return comp


def _region_beyond_line(F, line_rz, r_far, z_low):
    """Revolved region on the TE side of a meridional line (r, z) running hub (low v) -> shroud (high v):
    closed by a horizontal line out to r_far, down to z_low and back under the line's hub end."""
    L = np.asarray(line_rz, float)
    a, b = tuple(L[0]), tuple(L[-1])
    return _revolve_profile(F, [("curve", L), ("line", b, (r_far, b[1])), ("line", (r_far, b[1]), (r_far, z_low)),
                                ("line", (r_far, z_low), (a[0], z_low)), ("line", (a[0], z_low), a)])


def _splitter_region(spec, p_split, vr=(-0.3, 1.3), n=60):
    """Region downstream of the splitter LE line p = p_split (meridional map, extended past hub and tip)."""
    mm = spec.mermap(); vv = np.linspace(*vr, n)
    R, Z = mm.rz(np.full(n, p_split), vv)
    z_low = min(-5.0, float(Z.min()) - 5.0)
    return _region_beyond_line(spec.frame, np.c_[R, Z], spec.r_te + 20.0, z_low)


def build(spec: Spec, bore_d, z_front=None, log=print, unify=False, le_round=True, floor=True, floor_delta=0.4,
          fuse_bench=False):
    """Hub (revolved profile) + N blades (slab trimmed to the blade envelope, patterned), fused into one
    solid.  Returns (shape, info)."""
    import time
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Common, BRepAlgoAPI_Fuse
    from .fuse_profile import FuseProfile, bench_enabled, bench_fuse
    prof = FuseProfile()
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCP.gp import gp_Trsf, gp_Ax1, gp_Dir
    from OCP.TopoDS import TopoDS_Compound
    from OCP.BRep import BRep_Builder
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.TopAbs import TopAbs_SOLID, TopAbs_FACE
    t0 = time.time()
    F = spec.frame; S = spec.shroud
    rb = 0.5 * bore_d
    hsp = spec.hub_spline
    t_lo = spec.hub_t[0]                  # hub data end at the eye
    eye = getattr(spec, "eye", None)
    if z_front is None:
        z_front = eye["z_front"] if eye else float((hsp(t_lo) - t_lo) / 2)
    zf = float(z_front)
    t_f = t_at_z(hsp, zf, t_lo - 10, spec.hub_t[1])
    t_te = t_at_r(hsp, spec.r_te, spec.hub_t[0], spec.hub_t[1] + 10)
    tt = np.linspace(t_f, t_te, 300); ss = hsp(tt)
    hub_rz = np.c_[(ss + tt) / 2, (ss - tt) / 2]          # from the front face to the TE (rim)
    r_f = float(hub_rz[0, 0]); z_te = float(hub_rz[-1, 1])
    use_floor = floor and getattr(spec, "floor", None) is not None
    prof_plain = [("line", (rb, 0.0), (spec.r_te, 0.0)),
                  ("line", (spec.r_te, 0.0), (spec.r_te, z_te)),
                  ("curve", hub_rz[::-1]),
                  ("line", (r_f, zf), (rb, zf)),
                  ("line", (rb, zf), (rb, 0.0))]
    if use_floor:
        # bladed part of the hub lowered by delta (along the mer-map v direction); the per-passage floor
        # slabs put the real floor back on top.  The step at the LE plane and the rim lie in the plane
        # z = z_le and on the cylinder r = r_te, which the blade envelope trims to exactly.
        mm_ = spec.mermap(); pp = np.linspace(1.0, 0.0, 300)
        hr_, hz_ = spec.hub.at(pp); sr_, sz_ = spec.shroud.at(pp); L_ = np.hypot(sr_ - hr_, sz_ - hz_)
        lo_r, lo_z = mm_.rz(pp, -floor_delta / L_)
        lowered = np.c_[lo_r, lo_z]
        t_le = t_at_z_or_end(hsp, spec.z_le, t_lo - 10, spec.hub_t[1], t_lo)[0]
        tt2 = np.linspace(t_f, t_le, 200); ss2 = hsp(tt2)
        nose = np.c_[(ss2 + tt2) / 2, (ss2 - tt2) / 2]            # front face -> LE (not lowered)
        prof = [("line", (rb, 0.0), (spec.r_te, 0.0)),
                ("line", (spec.r_te, 0.0), tuple(lowered[0])),
                ("curve", lowered),
                ("line", tuple(lowered[-1]), tuple(nose[-1])),
                ("curve", nose[::-1]),
                ("line", (r_f, zf), (rb, zf)),
                ("line", (rb, zf), (rb, 0.0))]
    else:
        prof = [("line", (rb, 0.0), (spec.r_te, 0.0)),
                ("line", (spec.r_te, 0.0), (spec.r_te, z_te)),
                ("curve", hub_rz[::-1]),
                ("line", (r_f, zf), (rb, zf)),
                ("line", (rb, zf), (rb, 0.0))]
    hub = _revolve_profile(F, prof)
    log(f"[impeller] hub revolved ({time.time() - t0:.1f}s): front face z {zf:.3f} r {r_f:.3f}, rim z {z_te:.3f}")
    sh = _rz_poly(S, 0.0, 1.0)
    # inner trim of the blade: buried 1 mm inside the hub nose at the eye plane (the hub only grows in r
    # below it), so the blade's eye-plane top ends inside the hub where it is still full thickness
    t_le = t_at_z_or_end(hsp, spec.z_le, t_lo - 10, spec.hub_t[1], t_lo)[0]
    r_hub_le = float((hsp(t_le) + t_le) / 2)
    r_in = max(rb + 0.3, r_hub_le - 1.0)
    env = _revolve_profile(F, [("line", (r_in, 0.3), (spec.r_te, 0.3)),
                               ("line", (spec.r_te, 0.3), tuple(sh[-1])),
                               ("curve", sh[::-1]),
                               ("line", tuple(sh[0]), (r_in, spec.z_le)),
                               ("line", (r_in, spec.z_le), (r_in, 0.3))])
    _COMMON_LAST["method"] = None
    blade, le_info = _blade(spec, env, rb, le_round, log)
    trim_method = _COMMON_LAST["method"]
    slab_valid = bool(le_info.pop("slab_valid", True))
    if blade is None or blade.IsNull() or U_volume(blade) < max(1.0, 1e-4 * U_volume(hub)):
        # (Cummins 140: 0.043 mm3 from a blown-up side fit gave 33 loose solids instead of a clean failure)
        # never return a bare hub as an impeller: the gate would report a 9 mm 'fit' instead of the cause
        raise ValueError("impeller.build: blade trim (slab x envelope) came out empty, also with a fuzzy boolean")
    log(f"[impeller] blade trimmed ({time.time() - t0:.1f}s)")
    fslab = None
    if use_floor:
        fs = floor_slab(spec, floor_delta)
        fslab = prof.call("common_floor_envelope", lambda: BRepAlgoAPI_Common(fs, env).Shape())
        log(f"[impeller] floor slab ({time.time() - t0:.1f}s) vol {U_volume(fslab):.1f}")
    ax = gp_Ax1(_P(F.a0), gp_Dir(*map(float, F.u)))

    def pattern(sh):
        return _pattern_indexed(ax, spec.N, lambda k: sh)

    def one_valid(sh):
        return BRepCheck_Analyzer(sh).IsValid() and U.count(sh, TopAbs_SOLID) == 1

    from . import occ_util as U
    base = hub
    if fslab is not None:
        # staged: the floor slabs first (they do not touch each other), then the blades (one compound
        # with overlapping blades and floors does not fuse correctly)
        hf = prof.call("fuse_hub_floors", lambda: BRepAlgoAPI_Fuse(hub, pattern(fslab)).Shape())
        log(f"[impeller] hub + floors ({time.time() - t0:.1f}s) valid={one_valid(hf)}")
        if one_valid(hf):
            base = hf
        else:
            spec.info["floor_fuse_failed"] = True
            fslab = None
            hub = _revolve_profile(F, prof_plain)       # back to the plain revolved hub
            base = hub
    fam = getattr(spec, "families", None)
    split_info = None
    if fam and fam.get("splitter"):
        cut_ = _splitter_region(spec, fam["p_split"])
        sblade, s_sides = None, "full-blade side fit"
        if getattr(spec, "sides_splitter", None) is not None:
            # the splitter from its own side fit (thinner/different from the full blade); its LE is cut
            # sharp at p_split, so no LE round on its slab
            import copy as _copy
            sspec = _copy.copy(spec); sspec.sides = spec.sides_splitter
            try:
                sb_, _ = _blade(sspec, env, rb, False, log)
                if sb_ is not None and not sb_.IsNull() and U_volume(sb_) > 1.0:
                    sblade = _common(sb_, cut_); s_sides = "own family side fit"
            except Exception as e:  # noqa: BLE001
                s_sides = f"full-blade side fit (own fit failed: {type(e).__name__}: {e})"
            if sblade is None or sblade.IsNull() or U_volume(sblade) <= 1e-6:
                sblade = None
        if sblade is None:
            sblade = _common(blade, cut_)
        split_info = {"n_full": fam["n_full"], "n_splitter": fam["n_splitter"], "p_split": fam["p_split"],
                      "splitter_parity": fam["splitter_parity"], "full_blade_volume": U_volume(blade),
                      "splitter_blade_volume": U_volume(sblade), "splitter_le": "sharp (cut at the LE line)",
                      "splitter_sides": s_sides}
        if sblade is None or sblade.IsNull() or U_volume(sblade) <= 1e-6:
            raise ValueError("impeller.build: splitter blade (blade x region beyond p_split) came out empty")
        par = fam["splitter_parity"]
        blades = _pattern_indexed(ax, spec.N, lambda k: sblade if k % 2 == par else blade)
        log(f"[impeller] splitter build: {fam['n_full']} full + {fam['n_splitter']} splitter, p_split {fam['p_split']}")
    else:
        blades = pattern(blade)
    def _fuse_base_blades():
        fu = BRepAlgoAPI_Fuse(base, blades)
        fu.Build()
        return fu.Shape()
    shape = prof.call("fuse_base_blades", _fuse_base_blades)
    log(f"[impeller] fused ({time.time() - t0:.1f}s) valid={BRepCheck_Analyzer(shape).IsValid()}")
    log("[impeller] fuse profile\n" + prof.format_table())
    spec.info["fuse_valid_before_unify"] = bool(BRepCheck_Analyzer(shape).IsValid())
    if unify:
      try:
        from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
        un = ShapeUpgrade_UnifySameDomain(shape, True, True, False); un.Build(); shape = un.Shape()
      except Exception as e:  # noqa: BLE001
        log(f"[impeller] unify skipped: {e}")
    info = {"valid": bool(BRepCheck_Analyzer(shape).IsValid()), "n_solids": U.count(shape, TopAbs_SOLID),
            "n_faces": U.count(shape, TopAbs_FACE), "volume": U.volume(shape), "hub_volume": U.volume(hub),
            "blade_volume": U.volume(blade), "slab_valid": slab_valid, "blade_trim_boolean": trim_method,
            "front_face": {"z": zf, "r_outer": r_f}, "rim_hub_z": z_te, "bore_d": bore_d,
            "le_round": le_info, "splitter": split_info, "floor": ({"delta": floor_delta, "slab_volume": U.volume(fslab), "h_range": spec.floor.get("h_range")}
                                           if fslab is not None else None),
            "seconds": round(time.time() - t0, 1),
            "fuse_profile": prof.table()}
    try:
        from .solid_check import assess_shape
        rec = assess_shape(shape)
        info["n_solids"] = rec["n_solids"]
        info["closed"] = rec["closed"]
        info["valid_solid"] = rec["valid_solid"]
        info["valid"] = rec["brepcheck_valid"]
    except Exception as ex:  # noqa: BLE001
        info["n_solids"] = 0 if info.get("n_solids") is None else info["n_solids"]
        info["closed"] = bool(info.get("valid"))
        info["valid_solid"] = bool(info.get("valid") and info.get("n_solids") == 1 and info["closed"])
        info["solid_check_error"] = f"{type(ex).__name__}: {ex}"
    log(f"[impeller] solids={info['n_solids']} valid_solid={info['valid_solid']} brepcheck={info['valid']}")
    if bench_enabled(fuse_bench):
        try:
            info["fuse_bench"] = bench_fuse(base, blades)
            log("[impeller] fuse bench (discarded; default solid unchanged): "
                + ", ".join(f"{r['variant']} {r['seconds']:.3f}s" for r in info["fuse_bench"]["variants"]))
        except Exception as ex:  # noqa: BLE001
            info["fuse_bench"] = {"ran": False, "error": f"{type(ex).__name__}: {ex}"}
    else:
        info["fuse_bench"] = {"ran": False, "reason": "off by default"}
    return shape, info


# ------------------------------------------------------------------ detection and parameters
def probe(mesh, min_blades=5):
    """Is this an N-fold bladed rotor?  Blades are thin sheets spanning most of the radius, so all four
    rings (0.4 .. 0.85 R) must carry tangentially-facing area with the same N-fold period (a gear's teeth
    occupy only a thin ring at the OD; a turned body has no tangential faces).  Returns (N or None, info)."""
    try:
        m2, _ = _even_faces(mesh)
        C = np.asarray(m2.triangles_center); Nn = np.asarray(m2.face_normals); A = np.asarray(m2.area_faces)
        F, finfo = find_frame(C, Nn, A)
        d = _faces(m2, F)
        N, votes = blade_count(d["r"], d["th"], d["nt"], A, finfo["rim_radius"])
    except Exception as e:  # noqa: BLE001
        return None, {"reason": f"probe failed: {type(e).__name__}: {e}"}
    tang = float(A[np.abs(d["nt"]) > 0.5].sum() / A.sum())
    info = {"votes": votes, "tangential_area_frac": round(tang, 3), "rim_radius": round(finfo["rim_radius"], 3)}
    ok = N is not None and len(votes) == 4 and len(set(votes)) == 1 and N >= min_blades and tang > 0.15
    info["reason"] = "N-fold blades at all radii" if ok else "not a bladed rotor (votes/tangential area)"
    sp = splitter_pattern(votes, tang)
    if sp is not None and not ok and 2 * sp["n_full"] >= min_blades:
        info["splitter"] = sp
        info["reason"] = (f"splitter impeller: {sp['n_full']} full + {sp['n_splitter']} splitter blades "
                          f"(ring votes {votes}); analysed {2 * sp['n_full']}-fold, families split per sector")
        return 2 * sp["n_full"], info
    return (N if ok else None), info


def splitter_pattern(votes, tang=1.0, min_full=3):
    """Full + splitter blades alternating: the inner ring(s) (inside the splitters' leading edge) see N_full,
    the outer rings 2 x N_full (Cummins 140 mm: ring votes 8/16/16/16 at 0.40/0.55/0.70/0.85 R).  Returns
    {n_full, n_splitter, splitter_le_ring_index, votes} or None.  Rings run inner -> outer."""
    if len(votes) < 3 or tang <= 0.15:
        return None
    n_out = votes[-1]
    if n_out % 2 or n_out // 2 < min_full:
        return None
    n_full = n_out // 2
    k = 0
    while k < len(votes) and votes[k] == n_full:
        k += 1
    if k == 0 or k == len(votes) or any(v != n_out for v in votes[k:]):
        return None
    return {"n_full": n_full, "n_splitter": n_full, "splitter_le_ring_index": k, "votes": list(votes)}


def scan_bore(spec, mesh):
    """Bore diameter if the scan has one: inward-facing faces inside the hub nose radius."""
    d = _faces(mesh, spec.frame)
    r_le = float(spec.hub.at(np.array([0.0]))[0][0])
    k = (d["nr"] < -0.9) & (d["r"] < 0.9 * r_le) & (d["z"] > 0.5) & (d["z"] < spec.z_le)
    if d["A"][k].sum() < 0.3 * 2 * np.pi * 5.0 * 0.5 * spec.z_le:
        return None
    return float(2 * np.median(d["r"][k]))


def blade_parameters(spec):
    mm = spec.mermap(); P = np.linspace(0, 1, 201)
    out = {"N": spec.N, "le_plane_z": round(spec.z_le, 3), "r_te": round(spec.r_te, 3), "le_round": spec.info.get("le_round"),
           "angles": "blade angle beta from the meridional direction, tan(beta) = r dtheta/dm of the camber; 90 deg = tangential",
           "thickness": "tangential (r * dtheta between the side sheets); normal ~ tangential * cos(beta)",
           "spans": {}}
    for name, v in (("root_incl_fillet", 0.02), ("span_25", 0.25), ("span_50", 0.5), ("tip", 0.98)):
        vv = np.full_like(P, v)
        R, Z = mm.rz(P, vv); th = spec.camber(P, vv)
        t = (spec.sides[1](P, vv) - spec.sides[-1](P, vv)) * R
        dm = np.r_[0, np.cumsum(np.hypot(np.diff(R), np.diff(Z)))]
        b = np.degrees(np.arctan(R * np.gradient(th, dm)))
        out["spans"][name] = {
            "LE_rz": [round(float(R[0]), 3), round(float(Z[0]), 3)], "TE_rz": [round(float(R[-1]), 3), round(float(Z[-1]), 3)],
            "meridional_length": round(float(dm[-1]), 2), "wrap_deg": round(float(np.degrees(abs(th[-1] - th[0]))), 2),
            "beta_inlet_deg": round(float(np.median(b[2:8])), 1), "beta_exit_deg": round(float(np.median(b[-8:-2])), 1),
            "thickness_normal_mm": {f"{q:.2f}": round(float(np.interp(q, P, t * np.cos(np.radians(b)))), 3)
                                    for q in (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.97)}}
    Rs, Zs = spec.shroud.at(np.linspace(0, 1, 11)); Rh, Zh = spec.hub.at(np.linspace(0, 1, 11))
    out["tip_line_rz"] = np.round(np.c_[Rs, Zs], 3).tolist()
    out["hub_line_rz"] = np.round(np.c_[Rh, Zh], 3).tolist()
    out["exit_blade_height"] = round(float(np.hypot(Rs[-1] - Rh[-1], Zs[-1] - Zh[-1])), 3)
    fam = getattr(spec, "families", None)
    if fam and fam.get("splitter"):
        ps = fam["p_split"]
        Rl, Zl = mm.rz(np.full(3, ps), np.array([0.0, 0.5, 1.0]))
        dm_all = out["spans"]["span_50"]["meridional_length"]
        out["families"] = {
            "full": {"N": fam["n_full"], "LE_p": 0.0, "LE_rz_hub_mid_tip": np.round(np.c_[mm.rz(np.zeros(3), np.array([0.0, 0.5, 1.0]))], 3).T.tolist(),
                     "scatter": fam.get("scatter", {}).get("full")},
            "splitter": {"N": fam["n_splitter"], "LE_p": ps, "LE_rz_hub_mid_tip": np.round(np.c_[Rl, Zl], 3).tolist(),
                         "meridional_length_span50_approx": round(dm_all * (1 - ps), 2),
                         "phase": "half pitch (folded with the full blades; same camber and thickness downstream of LE)",
                         "scatter": fam.get("scatter", {}).get("splitter")}}
    return out
