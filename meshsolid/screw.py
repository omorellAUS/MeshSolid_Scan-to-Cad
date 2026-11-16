"""Helical screw rotors (M8): screw-compressor male / female rotors, twin- and triple-screw pump screws,
variable-pitch screws.

Recognition (no per-part tuning; thresholds are geometric):
  axis      PCA candidates refined through the centres of the round (journal) slices; fallback: centroids
            of the lobed slices (N >= 2 profiles are centred)
  zone      the longest run of axial slices whose outer envelope r(theta) is lobed
  lobes N   dominant angular harmonic of the envelopes (verified by N-fold self-correlation)
  twist     phase drift of harmonic N along z: Theta(z) = -arg c_N(z) / N (unwrapped); straight line =
            constant lead (lead = 2 pi / |w|, hand = sign w: + right), a significant quadratic term =
            variable pitch (lead varies linearly with z)
  ends      flat end faces (normals along the axis) near the zone limits
  profile   every scan point of the zone unwound to the z_a section (phi - Theta(z)), folded onto one
            lobe, radial median per angle bin, then refined along the curve normal (steep flanks) and
            replicated to N lobes: a closed periodic B-spline, i.e. the transverse profile is FITTED, not
            assumed to be any standard form
Build: the profile swept with BRepOffsetAPI_MakePipeShell along the axis with an auxiliary curve on a
cylinder (u = Theta(v)): a line for constant lead (exact helicoid, as the gear builder), a B-spline for
variable pitch.  The pipeline joins the sweep to the turned journals like a gear: the zone enters the
revolve profile as its blank (end planes + tip cylinder) and the blank minus the rotor is cut away."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .gear import frame, cyl, refine_axis, _roundness

TWO_PI = 2 * math.pi


@dataclass
class ScrewSpec:
    lobes: int
    hand: str
    axis_point: list
    axis_dir: list
    z_a: float
    z_b: float
    r_tip: float
    r_root: float
    theta_poly: list          # Theta(z) - Theta(z_a) as polynomial in (z - z_a), highest power first (np.polyval)
    variable_pitch: bool
    lead_mm: float = None     # constant-lead rotors
    lead_range_mm: list = None
    profile: np.ndarray = None  # (M, 2) closed polyline (x, y) of the transverse section at z_a, rotor frame
    corners: list = field(default_factory=list)   # indices into profile of sharp corners (C0 joins in the build)
    fit: dict = field(default_factory=dict)
    kind: str = "screw_rotor"
    confidence: dict = field(default_factory=dict)

    def theta(self, z):
        return np.polyval(self.theta_poly, np.asarray(z, float) - self.z_a)

    def w_at(self, z):
        return np.polyval(np.polyder(self.theta_poly), np.asarray(z, float) - self.z_a)


def _envelopes(r, th, z, z0, z1, dz=0.5, nb=720):
    zs = np.arange(z0, z1, dz)
    out = []
    ib = ((th + math.pi) / TWO_PI * nb).astype(int) % nb
    order = np.argsort(z); zsrt = z[order]
    for a in zs:
        lo, hi = np.searchsorted(zsrt, [a, a + dz])
        k = order[lo:hi]
        if len(k) < 100:
            out.append(None); continue
        env = np.full(nb, np.nan); np.fmax.at(env, ib[k], r[k])
        good = ~np.isnan(env)
        if good.mean() < 0.6:
            out.append(None); continue
        gi = np.flatnonzero(good)
        env = np.interp(np.arange(nb), gi, env[gi], period=nb)
        out.append(dict(z=a + dz / 2, env=env))
    return out


def _axis(P):
    c = P.mean(0); _, V = np.linalg.eigh(np.cov((P - c).T))
    cands = []
    for u0 in V.T[::-1]:
        a0, u = refine_axis(P, c, u0)
        cands.append((_roundness(P, a0, u), a0, u))
    cands.sort(key=lambda t: -t[0])
    return cands[0][1], cands[0][2], [round(t[0], 3) for t in cands]


def _zone(sl, min_len_slices=10):
    lob = []
    for s in sl:
        if s is None:
            lob.append(False); continue
        e = s["env"]
        lob.append(bool((e.max() - e.min()) > max(1.0, 0.12 * e.max())))
    best, cur = (0, -1), None
    for i, f in enumerate(lob + [False]):
        if f and cur is None:
            cur = i
        if not f and cur is not None:
            if i - cur > best[1] - best[0] + 1:
                best = (cur, i - 1)
            cur = None
    if best[1] - best[0] + 1 < min_len_slices:
        return None
    return best


def analyse(mesh, tol=0.05, n_samples=900000, seed=0):
    """Detect a helical screw rotor.  Returns (ScrewSpec | None, info)."""
    import trimesh
    info = {}
    P, fi = trimesh.sample.sample_surface(mesh, n_samples, seed=seed)
    P = np.asarray(P); Nn = np.asarray(mesh.face_normals)[fi]
    a0, u, rnd = _axis(P)
    info["axis_roundness"] = rnd
    r, th, z = cyl(P, a0, u)
    dz = 0.5
    sl = _envelopes(r, th, z, z.min(), z.max(), dz)
    zn = _zone(sl)
    if zn is None:
        info["reason"] = "no lobed axial zone"
        return None, info
    i0, i1 = zn
    zr0, zr1 = sl[i0]["z"] - dz / 2, sl[i1]["z"] + dz / 2
    # fallback axis for parts without round journals: centroid line of the lobed slices
    if rnd[0] < 0.05:
        info["axis_from"] = "lobed-slice centroids"
    inner = [s for s in sl[i0 + 4:i1 - 3] if s is not None]
    if len(inner) < 6:
        info["reason"] = "lobed zone too short"
        return None, info
    E = np.array([s["env"] for s in inner]); nb = E.shape[1]
    F = np.fft.rfft(E - E.mean(1, keepdims=True), axis=1)
    amp = np.abs(F[:, 1:17]).mean(0)
    N = int(np.argmax(amp) + 1)
    # self-correlation check: the envelope repeats every 2 pi / N
    sh = nb // N
    rep = float(np.mean([np.corrcoef(e, np.roll(e, sh))[0, 1] for e in E])) if N > 1 else 1.0
    info["harmonic_amp"] = [round(float(a), 3) for a in amp]
    info["n_fold_correlation"] = round(rep, 4)
    if rep < 0.9:
        info["reason"] = f"envelope not {N}-fold periodic (corr {rep:.3f})"
        return None, info
    zz = np.array([s["z"] for s in inner])
    # bins start at theta = -pi: c_N = sum r e^{-i N (theta + pi)} ... phase offset is constant, irrelevant
    ph = np.unwrap(np.angle(F[:, N]))
    Th = -ph / N
    p1 = np.polyfit(zz, Th, 1); p2 = np.polyfit(zz, Th, 2)
    res1 = Th - np.polyval(p1, zz); res2 = Th - np.polyval(p2, zz)
    w_ends = np.polyval(np.polyder(p2), [zz[0], zz[-1]])
    rel = abs(w_ends[1] - w_ends[0]) / max(abs(np.mean(w_ends)), 1e-9)
    variable = bool(rel > 0.03 and np.sqrt(np.mean(res2 ** 2)) < 0.5 * np.sqrt(np.mean(res1 ** 2)))
    info["twist_fit"] = {"rms_linear_rad": float(np.sqrt(np.mean(res1 ** 2))), "rms_quadratic_rad": float(np.sqrt(np.mean(res2 ** 2))),
                         "twist_rate_change_frac": round(float(rel), 4)}
    poly = p2 if variable else p1
    w_mean = float(np.polyval(np.polyder(poly), 0.5 * (zz[0] + zz[-1])))
    # a screw rotor's lobes wrap visibly; a keyway, flat or straight-lobed part has (near) zero twist.
    # Require at least 15 deg of wrap over the zone and a tip displacement well above the noise
    tw = abs(float(np.polyval(poly, zz[-1]) - np.polyval(poly, zz[0])))
    r_out = float(np.median(E.max(1)))
    if math.degrees(tw) < 15.0 or tw * r_out < max(1.0, 20 * tol):
        info["reason"] = (f"straight lobes (no helix): wrap {math.degrees(tw):.2f} deg over the zone, "
                          f"{tw * r_out:.3f} mm at the tip: not a screw rotor")
        return None, info
    # ends: flat faces with axial normals near the zone limits
    nu = np.abs(Nn @ u)
    r_env_min = float(np.median(E.min(1)))

    def end(zlim, sgn):
        k = (nu > 0.95) & (np.abs(z - zlim) < 3.0) & (r > 0.5 * r_env_min) & (r < r_env_min - 0.3)
        return float(np.median(z[k])) if k.sum() >= 50 else zlim
    z_a, z_b = end(zr0, -1), end(zr1, +1)
    # unwound, folded profile from every scan point of the zone
    kz = (z > z_a + 1.5) & (z < z_b - 1.5) & (r > 0.5 * r_env_min)
    rr, tt, zp = r[kz], th[kz], z[kz]
    phi = tt - (np.polyval(poly, zp) - np.polyval(poly, z_a))
    lobe = TWO_PI / N
    x = np.mod(phi, lobe) / lobe
    M = 240
    bi = np.minimum((x * M).astype(int), M - 1)
    med = np.array([np.median(rr[bi == j]) if (bi == j).sum() > 5 else np.nan for j in range(M)])
    g = np.flatnonzero(~np.isnan(med))
    med = np.interp(np.arange(M), g, med[g], period=M)
    ang = (np.arange(M) + 0.5) / M * lobe
    prof = _replicate(np.c_[med * np.cos(ang), med * np.sin(ang)], N)
    # refine along the curve normal (polar medians are biased on steep flanks)
    Q = np.c_[rr * np.cos(np.mod(phi, TWO_PI)), rr * np.sin(np.mod(phi, TWO_PI))]
    prof, rms = _refine_profile(prof, Q, N, M)
    # sharp profile corners (trapezoid thread forms, land / flank kinks): the scan and the median fit round
    # them; rebuild each detected corner as the intersection of the two flanking curves fitted to the scan
    edge = float(np.median(mesh.edges_unique_length)) if hasattr(mesh, "edges_unique_length") else 0.6
    prof, corners, cinfo = _sharpen(prof, Q, N, M, edge, rms)
    rp = np.hypot(prof[:, 0], prof[:, 1])
    spec = ScrewSpec(lobes=N, hand="right" if w_mean > 0 else "left", axis_point=a0.tolist(), axis_dir=u.tolist(),
                     z_a=z_a, z_b=z_b, r_tip=float(rp.max()), r_root=float(rp.min()),
                     theta_poly=list(_shift_poly(poly, z_a)), variable_pitch=variable, profile=prof,
                     corners=corners)
    if variable:
        w0, w1 = spec.w_at(z_a), spec.w_at(z_b)
        spec.lead_range_mm = [float(TWO_PI / abs(w0)), float(TWO_PI / abs(w1))]
    else:
        spec.lead_mm = float(TWO_PI / abs(w_mean))
    spec.fit = {"profile_rms_mm": rms, "profile_points": int(kz.sum()), "wrap_deg": float(abs(math.degrees(spec.theta(z_b)))),
                "length_mm": float(z_b - z_a), "zone_rough": [float(zr0), float(zr1)],
                "corners_per_lobe": len(corners) // max(N, 1), "corner_fit": cinfo}
    spec.kind = _kind(spec)
    # acceptance: one rigid helicoid must explain the whole zone (rejects impeller blades, twisted
    # freeform parts and lobed parts whose lobes are not a constant/linearly varying screw)
    tw_mm = info["twist_fit"]["rms_quadratic_rad" if variable else "rms_linear_rad"] * spec.r_tip
    gate = max(4 * tol, 0.12)
    info["acceptance"] = {"profile_rms_mm": round(rms, 4), "twist_residual_mm_at_tip": round(tw_mm, 4), "gate_mm": gate}
    if rms > gate or tw_mm > max(4 * tol, 0.15):
        info["reason"] = f"not one helicoid: profile rms {rms:.3f} mm, twist residual {tw_mm:.3f} mm at the tip (gate {gate:.2f})"
        return None, info
    spec.confidence = {"lobes": "high" if rep > 0.97 else "medium",
                       "lead": "high" if info["twist_fit"]["rms_quadratic_rad"] < 0.01 else "medium",
                       "profile": "high" if rms < max(3 * tol, 0.1) else "medium (residual %.3f mm)" % rms}
    info["accepted"] = True
    return spec, info


def _kind(s):
    depth = (s.r_tip - s.r_root) / s.r_tip
    if s.variable_pitch:
        return "variable_pitch_screw"
    if s.lobes >= 3 and depth > 0.25:
        return "screw_compressor_rotor"
    return "screw_pump_screw"


def _shift_poly(p, z_a):
    """Coefficients of q(t) = p(t + z_a) - p(z_a)."""
    q = np.poly1d(p)(np.poly1d([1.0, z_a]))
    c = np.array(q.c, float); c[-1] = 0.0
    return c


def _replicate(lobe_pts, N):
    out = []
    for k in range(N):
        a = TWO_PI * k / N; c, s = math.cos(a), math.sin(a)
        out.append(lobe_pts @ np.array([[c, s], [-s, c]]))
    return np.vstack(out)


def _refine_profile(prof, Q, N, M, iters=3):
    """Move each profile vertex along the local normal by the median signed offset of the scan points
    whose closest curve vertex it is; returns (profile, rms of the point residuals)."""
    from scipy.spatial import cKDTree
    rms = None
    for it in range(iters + 1):
        dense = _densify(prof, 8)
        T = np.roll(dense, -1, 0) - np.roll(dense, 1, 0); T /= np.linalg.norm(T, axis=1, keepdims=True)
        Nrm = np.c_[T[:, 1], -T[:, 0]]
        if np.mean(np.einsum("ij,ij->i", Nrm, dense)) < 0:
            Nrm = -Nrm
        d, j = cKDTree(dense).query(Q)
        sd = np.einsum("ij,ij->i", Q - dense[j], Nrm[j])
        rms = float(np.sqrt(np.mean(sd ** 2)))
        if it == iters:
            break
        L = len(prof)
        jv = j // 8                                        # owning coarse vertex
        off = np.zeros(L)
        # median per vertex, then fold the N lobes together (equal lobes)
        order = np.argsort(jv); sdo = sd[order]; bounds = np.searchsorted(jv[order], np.arange(L + 1))
        for v in range(L):
            a, b = bounds[v], bounds[v + 1]
            off[v] = np.median(sdo[a:b]) if b - a > 3 else 0.0
        off = off.reshape(N, M).mean(0) if len(off) == N * M else off
        off = np.convolve(np.r_[off[-2:], off, off[:2]], np.ones(5) / 5, "valid")
        lobe = prof[:M] + off[:, None] * Nrm[::8][:M]
        prof = _replicate(lobe, N)
    return prof, rms


def _rot(P, a):
    c, s_ = math.cos(a), math.sin(a)
    return P @ np.array([[c, s_], [-s_, c]])


def _quad_side(X, toward):
    """Quadratic y(x) in the principal frame of the 2-D points X; x axis oriented toward the point `toward`."""
    c = X.mean(0)
    _, _, vt = np.linalg.svd(X - c, full_matrices=False)
    t = vt[0]
    if (toward - c) @ t < 0:
        t = -t
    nrm = np.array([-t[1], t[0]])
    x = (X - c) @ t; y = (X - c) @ nrm
    co = np.polyfit(x, y, 2)
    res = float(np.sqrt(np.mean((np.polyval(co, x) - y) ** 2)))
    return dict(c=c, t=t, n=nrm, co=co, x0=float(x.min()), x1=float(x.max()), res=res)


def _side_pts(f, xa, xb, n):
    x = np.linspace(xa, xb, n)
    return f["c"] + np.outer(x, f["t"]) + np.outer(np.polyval(f["co"], x), f["n"])


def _side_tan(f):
    dy = np.polyval(np.polyder(f["co"]), f["x1"])
    v = f["t"] + dy * f["n"]
    return v / np.linalg.norm(v)


def _sharpen(prof, Q, N, M, edge, noise):
    """Detect and rebuild sharp corners of the lobe profile.

    For every profile vertex, quadratics are fitted to the scan points on each side, outside a gap of
    half-width e (the zone the scan + median fit rounds).  A vertex is a corner when both side fits are
    clean (residual near the noise), their directions differ by > 12 deg and their extrapolations cross at a
    point clearly outside the current (rounded) profile.  On a smooth curve the two extrapolations cross on
    the curve itself, so smooth flanks of any curvature are left alone.  The gap is then replaced by the two
    extrapolated curves meeting at the crossing point, which becomes a C0 corner of the built profile.
    Returns (profile, corner indices, info)."""
    from scipy.spatial import cKDTree
    L = len(prof)
    if L != N * M:
        return prof, [], {"corners": 0}
    lobe = TWO_PI / N
    e = max(1.0, 2.0 * edge); Lf = max(2.5, 4.0 * edge)
    if N > 1:
        a = np.mod(np.arctan2(Q[:, 1], Q[:, 0]), TWO_PI)
        k = np.minimum(np.floor(a / lobe), N - 1)
        Qf = np.empty_like(Q)
        for kk in range(N):
            m = k == kk
            Qf[m] = _rot(Q[m], -kk * lobe)
    else:
        Qf = Q
    if len(Qf) > 250000:
        Qf = Qf[np.random.default_rng(0).choice(len(Qf), 250000, replace=False)]
    if N > 1:
        Qx = np.vstack([Qf, _rot(Qf, lobe), _rot(Qf, -lobe)])
        E = np.vstack([prof[-M:], prof[:M], prof[M:2 * M] if N > 2 else prof[-M:]])     # lobe 0 in the middle
        if N == 2:
            E = np.vstack([prof[M:], prof[:M], prof[M:]])
    else:
        Qx = Qf
        E = np.vstack([prof, prof, prof])
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(E, axis=0), axis=1))]
    D = _densify(E, 8)[: 8 * (len(E) - 1) + 1]
    sd = np.interp(np.arange(len(D)) / 8.0, np.arange(len(E)), s)
    dq, jq = cKDTree(D).query(Qx, distance_upper_bound=max(0.5, 6 * noise))
    ok = np.isfinite(dq)
    sq = sd[jq[ok]]; Qk = Qx[ok]
    order = np.argsort(sq); sq = sq[order]; Qk = Qk[order]
    res_lim = 1.5 * noise + 0.01

    def apex(i, gap):
        """Crossing of the two side quadratics fitted outside a gap of half-width `gap` around vertex i."""
        si = s[i]
        la, lb = np.searchsorted(sq, [si - gap - Lf, si - gap]); ra, rb = np.searchsorted(sq, [si + gap, si + gap + Lf])
        if lb - la < 30 or rb - ra < 30:
            return None
        fl = _quad_side(Qk[la:lb], E[i]); fr = _quad_side(Qk[ra:rb], E[i])
        if max(fl["res"], fr["res"]) > res_lim:
            return None
        tl, tr_ = _side_tan(fl), _side_tan(fr)            # both point toward the corner
        turn = math.degrees(math.acos(np.clip(-tl @ tr_, -1, 1)))         # 0 = straight through
        A = _side_pts(fl, fl["x1"], fl["x1"] + 2.5 * gap, 120); B = _side_pts(fr, fr["x1"], fr["x1"] + 2.5 * gap, 120)
        dd = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)
        ia, ib = np.unravel_index(np.argmin(dd), dd.shape)
        if dd[ia, ib] > 0.05 or ia in (0, 119) or ib in (0, 119):
            return None
        return dict(X=0.5 * (A[ia] + B[ib]), fl=fl, fr=fr, turn=turn)

    cands = []; rejected = []
    for i in range(M, 2 * M):
        c = apex(i, e)
        if c is None or c["turn"] < 12.0:
            continue
        X = c["X"]
        lo, hi = max(0, i - 40), min(len(E), i + 41)
        loc = _densify(E[lo:hi], 8)[: 8 * (hi - lo - 1) + 1]
        h = float(np.min(np.linalg.norm(loc - X, axis=1)))
        if h < max(0.03, noise):
            continue
        # a real corner is where both sides meet whatever the gap: refit outside a 1.6x wider gap; on a smooth
        # curve of high curvature the extrapolated crossing drifts with the gap
        c2 = apex(i, 1.6 * e)
        drift = None if c2 is None else float(np.linalg.norm(c2["X"] - X))
        if drift is None or drift > max(0.03, 0.25 * h):
            rejected.append(i)
            continue
        c.update(i=i, h=h, drift=drift)
        cands.append(c)
    chosen = []                                            # one corner per cluster of candidates
    per = s[2 * M] - s[M]                                  # lobe perimeter: the profile is periodic per lobe

    def sep(a, b):
        d = abs(s[a] - s[b]) % per
        return min(d, per - d)
    for c in sorted(cands, key=lambda c: -c["h"]):
        # side-fit windows must not reach a stronger corner (a fit across a corner is not a clean side)
        if all(sep(c["i"], o["i"]) > e + Lf for o in chosen):
            chosen.append(c)
    if not chosen:
        return prof, [], {"corners": 0, "candidates": len(cands), "rejected_by_gap_drift": len(rejected)}
    lobe0 = prof[:M].copy()
    corners0 = []
    for c in chosen:
        i, X, fl, fr = c["i"], c["X"], c["fl"], c["fr"]
        iL = int(np.searchsorted(s, s[i] - e)); iR = int(np.searchsorted(s, s[i] + e))
        ic = int(np.argmin(np.linalg.norm(E[iL:iR + 1] - X, axis=1))) + iL
        ic = min(max(ic, iL + 2), iR - 2)
        xl = (X - fl["c"]) @ fl["t"]; xr = (X - fr["c"]) @ fr["t"]
        left = _side_pts(fl, fl["x1"], xl, ic - iL + 1); left[-1] = X
        right = _side_pts(fr, fr["x1"], xr, iR - ic + 1)[::-1]; right[0] = X
        for j, P in zip(range(iL, iR + 1), np.vstack([left, right[1:]])):
            lobe0[(j - M) % M] = P
        corners0.append((ic - M) % M)
    out = _replicate(lobe0, N)
    corners = sorted(int(c + kk * M) for kk in range(N) for c in corners0)
    info = {"corners": len(corners0), "gap_half_width_mm": round(e, 3), "rejected_by_gap_drift": len(rejected),
            "per_corner": [{"turn_deg": round(c["turn"], 1), "rebuilt_by_mm": round(c["h"], 4),
                            "side_fit_rms_mm": [round(c["fl"]["res"], 4), round(c["fr"]["res"], 4)],
                            "apex_drift_mm": round(c["drift"], 4)} for c in chosen]}
    return out, corners, info


def _densify(P, k):
    Q = np.roll(P, -1, 0)
    t = np.arange(k)[None, :, None] / k
    return (P[:, None, :] * (1 - t) + Q[:, None, :] * t).reshape(-1, 2)


# ----------------------------------------------------------------------------- build
def profile_curve(profile):
    """Closed periodic C2 B-spline through the profile points (z = 0 plane)."""
    from OCP.GeomAPI import GeomAPI_Interpolate
    from OCP.collections import HArray1_gp_Pnt as TColgp_HArray1OfPnt
    from OCP.gp import gp_Pnt
    n = len(profile)
    arr = TColgp_HArray1OfPnt(1, n)
    for i, (x, y) in enumerate(profile):
        arr.SetValue(i + 1, gp_Pnt(float(x), float(y), 0.0))
    it = GeomAPI_Interpolate(arr, True, 1e-6)
    it.Perform()
    if not it.IsDone():
        raise RuntimeError("profile interpolation failed")
    return it.Curve()


def _corner_wire(prof, corners, step):
    """Closed wire of open B-splines between consecutive sharp corners (C0 at the corners)."""
    from OCP.GeomAPI import GeomAPI_Interpolate
    from OCP.collections import HArray1_gp_Pnt
    from OCP.gp import gp_Pnt
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeVertex
    L = len(prof); cs = sorted(corners)
    verts = {c: BRepBuilderAPI_MakeVertex(gp_Pnt(float(prof[c][0]), float(prof[c][1]), 0.0)).Vertex() for c in cs}
    mw = BRepBuilderAPI_MakeWire()
    for a, b in zip(cs, cs[1:] + [cs[0] + L]):
        inner = list(range(a + 1, b))[::step]
        idx = [a] + [i for i in inner if b - i >= max(1, step // 2)] + [b]
        pts = prof[np.mod(idx, L)]
        arr = HArray1_gp_Pnt(1, len(pts))
        for j, (x, y) in enumerate(pts):
            arr.SetValue(j + 1, gp_Pnt(float(x), float(y), 0.0))
        it = GeomAPI_Interpolate(arr, False, 1e-6); it.Perform()
        if not it.IsDone():
            raise RuntimeError("profile segment interpolation failed")
        mw.Add(BRepBuilderAPI_MakeEdge(it.Curve(), verts[a % L], verts[b % L]).Edge())
    if not mw.IsDone():
        raise RuntimeError("corner profile wire failed")
    return mw.Wire()


def build_rotor(spec: ScrewSpec, per_lobe=120):
    """Solid of the rotor zone (z_a..z_b) in world coordinates: exact sweep of the fitted profile."""
    from OCP.gp import gp_Pnt, gp_Dir, gp_Ax3, gp_Pnt2d, gp_Dir2d, gp_Trsf, gp_Vec
    from OCP.Geom import Geom_CylindricalSurface
    from OCP.Geom2d import Geom2d_Line
    from OCP.Geom2dAPI import Geom2dAPI_Interpolate
    from OCP.collections import HArray1_gp_Pnt2d as TColgp_HArray1OfPnt2d
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire, BRepBuilderAPI_Transform
    from OCP.BRepOffsetAPI import BRepOffsetAPI_MakePipeShell
    from OCP.BRepFill import BRepFill_NoContact
    from OCP.BRepLib import BRepLib
    from OCP.TopAbs import TopAbs_SOLID
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS
    prof = spec.profile
    step = max(1, len(prof) // (per_lobe * spec.lobes)) if per_lobe else 1
    if spec.corners:
        wire = _corner_wire(prof, spec.corners, step)
    else:
        prof = prof[::step]
        wire = BRepBuilderAPI_MakeWire(BRepBuilderAPI_MakeEdge(profile_curve(prof)).Edge()).Wire()
    H = spec.z_b - spec.z_a
    spine = BRepBuilderAPI_MakeWire(BRepBuilderAPI_MakeEdge(gp_Pnt(0, 0, 0), gp_Pnt(0, 0, H)).Edge()).Wire()
    ps = BRepOffsetAPI_MakePipeShell(spine)
    cs = Geom_CylindricalSurface(gp_Ax3(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), 10.0)
    if not spec.variable_pitch:
        w = float(np.polyval(np.polyder(spec.theta_poly), 0.0))
        he = BRepBuilderAPI_MakeEdge(Geom2d_Line(gp_Pnt2d(0, 0), gp_Dir2d(w, 1.0)), cs, 0.0, H * math.sqrt(1 + w * w)).Edge()
    else:
        vs = np.linspace(0, H, 41)
        arr = TColgp_HArray1OfPnt2d(1, len(vs))
        for i, v in enumerate(vs):
            arr.SetValue(i + 1, gp_Pnt2d(float(np.polyval(spec.theta_poly, v)), float(v)))
        it = Geom2dAPI_Interpolate(arr, False, 1e-9); it.Perform()
        he = BRepBuilderAPI_MakeEdge(it.Curve(), cs).Edge()
    BRepLib.BuildCurves3d_s(he, 1e-7)
    ps.SetMode(BRepBuilderAPI_MakeWire(he).Wire(), False, BRepFill_NoContact)
    ps.Add(wire, False, False)
    ps.Build()
    if not ps.IsDone():
        raise RuntimeError("helical sweep failed")
    if not ps.MakeSolid():
        raise RuntimeError("sweep could not be closed into a solid")
    sp = ps.Shape()
    e1, e2, uu = frame(np.asarray(spec.axis_dir))
    a0 = np.asarray(spec.axis_point)
    loc = gp_Trsf()
    loc.SetValues(*[float(v) for v in (e1[0], e2[0], uu[0], a0[0], e1[1], e2[1], uu[1], a0[1], e1[2], e2[2], uu[2], a0[2])])
    Z = gp_Trsf(); Z.SetTranslation(gp_Vec(0, 0, spec.z_a))
    out = BRepBuilderAPI_Transform(sp, loc.Multiplied(Z), True).Shape()
    ex = TopExp_Explorer(out, TopAbs_SOLID)
    if not ex.More():
        raise RuntimeError("sweep produced no solid")
    return TopoDS.Solid(ex.Current())


def to_json(s: ScrewSpec):
    return {"kind": s.kind, "lobes": s.lobes, "hand": s.hand, "variable_pitch": s.variable_pitch,
            "lead_mm": None if s.lead_mm is None else round(s.lead_mm, 3),
            "lead_range_mm": None if s.lead_range_mm is None else [round(v, 3) for v in s.lead_range_mm],
            "wrap_deg": round(s.fit.get("wrap_deg", 0.0), 2), "rotor_length_mm": round(float(s.z_b - s.z_a), 3),
            "tip_diameter": round(2 * s.r_tip, 4), "root_diameter": round(2 * s.r_root, 4),
            "axis_point": [round(v, 4) for v in s.axis_point], "axis_dir": [round(v, 6) for v in s.axis_dir],
            "z_ends_axis_frame": [round(s.z_a, 3), round(s.z_b, 3)],
            "theta_poly_rad": [float(c) for c in s.theta_poly],
            "profile_points": len(s.profile), "profile_corners": len(s.corners), "profile_fit_rms_mm": round(s.fit.get("profile_rms_mm", 0.0), 4),
            "confidence": s.confidence}
