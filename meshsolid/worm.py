"""Cylindrical worm recognition (M12): starts, lead, hand, axial module, diameters, axial flank angle.

On a worm the surface radius about the axis is a function of the helical phase s = z - h L theta / 2 pi
(mod L) only (L = lead, h = +1 right hand).  We search L and h by the fraction of radial variance the
binned phase explains (thread.phase_score), refine L, count the crests in one lead (= starts z1), fold
to one axial pitch p_x = L / z1 and read the axial profile: tip / root radii (histogram peaks), the two
flanks as straight lines s(r) (ZA worm: straight in the axial section; the straightness is reported) and
the axial tooth thickness.  Module m = p_x / pi snapped to the DIN 780 / ISO 54 series; d1 = da - 2 m,
q = d1 / m, lead angle gamma = atan(z1 / q).

Gates (else None + reason, so screw threads and screw rotors are not called worms): m within 1.5 % of a
standard module, axial flank angle 12..25 deg (ZA/ZI/ZN/ZK 20 deg, 14.5 deg older), tooth depth 1.9..2.7 m.
"""
from __future__ import annotations

import math

import numpy as np

from .gear import STD_MODULE, cyl
from .spline import _axis, _peak
from .thread import phase_score

WORM_MODULES = sorted(set(STD_MODULE) | {1.6, 3.15, 6.3, 10.0, 12.5, 16.0, 20.0})
STD_Q = (6, 7, 7.5, 8, 8.5, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 22, 25)


def _toothed_slices(r, th, z, nr, dz=1.0, nb=180, tol=0.05):
    keep = nr > -0.3
    zs = np.arange(z.min(), z.max() + dz, dz)
    flags = []
    for a in zs[:-1]:
        k = keep & (z >= a) & (z < a + dz)
        if k.sum() < 200:
            flags.append(0.0); continue
        bi = ((th[k] + np.pi) / (2 * np.pi) * nb).astype(int) % nb
        env = np.full(nb, np.nan); np.fmax.at(env, bi, r[k])
        flags.append(float(np.nanmax(env) - np.nanmin(env)) if np.isfinite(env).mean() > 0.8 else 0.0)
    f = np.array(flags) > max(0.5, 10 * tol)
    best, cur = (0, -1), None
    for i, v in enumerate(f):
        if v and cur is None:
            cur = i
        if (not v or i == len(f) - 1) and cur is not None:
            j = i if v else i - 1
            if j - cur > best[1] - best[0]:
                best = (cur, j)
            cur = None
    if best[1] < best[0] + 4:
        return None
    return zs[best[0]], zs[best[1] + 1]


def analyse(mesh, tol=0.05, n_samples=500000, seed=0):
    """Returns (worm dict | None, info)."""
    import trimesh
    info = {}
    P, fi = trimesh.sample.sample_surface(mesh, n_samples, seed=seed)
    P = np.asarray(P); Nn = np.asarray(mesh.face_normals)[fi]
    a0, u = _axis(P)
    r, th, z = cyl(P, a0, u)
    radial = P - a0 - np.outer((P - a0) @ u, u)
    nr = np.einsum("ij,ij->i", Nn, radial / np.maximum(np.linalg.norm(radial, axis=1, keepdims=True), 1e-9))
    zz = _toothed_slices(r, th, z, nr, tol=tol)
    if zz is None:
        info["reason"] = "no axial zone with a radial pattern"; return None, info
    za, zb = zz
    k = (z > za + 1.0) & (z < zb - 1.0) & (nr > -0.3)
    rr, tt, zk = r[k], th[k], z[k]
    if len(rr) < 5000:
        info["reason"] = "patterned zone too small"; return None, info
    rng = np.random.default_rng(seed)
    sub = rng.choice(len(rr), min(60000, len(rr)), replace=False)
    rs, ts, zs_ = rr[sub], tt[sub], zk[sub]
    nb = 48
    best = (0.0, None, None)
    span = zb - za
    for L in np.exp(np.arange(math.log(1.0), math.log(max(4.0, 20 * span)), 1.0 / nb)):
        for h in (1, -1):
            sc, _ = phase_score(rs, zs_, ts, L, h, nb)
            if sc > best[0]:
                best = (sc, L, h)
    sc, L, h = best
    info.update(coarse_lead=None if L is None else round(float(L), 3), coarse_score=round(sc, 3))
    if L is None or sc < 0.8:
        info["reason"] = f"no helical phase explains the radius (best score {sc:.2f})"; return None, info
    lo, hi = L * (1 - 1.5 / nb), L * (1 + 1.5 / nb)               # golden-section refine on all zone points
    g = (math.sqrt(5) - 1) / 2
    f = lambda x: -phase_score(rr, zk, tt, x, h, 120)[0]          # noqa: E731
    c, d = hi - g * (hi - lo), lo + g * (hi - lo); fc, fd = f(c), f(d)
    for _ in range(40):
        if fc < fd:
            hi, d, fd = d, c, fc; c = hi - g * (hi - lo); fc = f(c)
        else:
            lo, c, fc = c, d, fd; d = lo + g * (hi - lo); fd = f(d)
    L = 0.5 * (lo + hi)
    sc, prof = phase_score(rr, zk, tt, L, h, 720)
    F = np.abs(np.fft.rfft(prof - prof.mean()))
    z1 = int(np.argmax(F[1:13]) + 1)
    px = L / z1
    s = np.mod(zk - h * L * tt / (2 * np.pi), px)                   # folded axial phase
    # crest centre: circular mean of the folded profile above mid height
    b2 = np.minimum((s / px * 240).astype(int), 239)
    cnt = np.bincount(b2, minlength=240); pf = np.bincount(b2, rr, minlength=240) / np.maximum(cnt, 1)
    mid = 0.5 * (pf.max() + pf.min())
    ang = (np.arange(240) + 0.5) / 240 * 2 * np.pi
    sc_c = (np.angle(np.sum(np.exp(1j * ang[pf > mid]))) % (2 * np.pi)) / (2 * np.pi) * px
    ds = np.mod(s - sc_c + px / 2, px) - px / 2
    sig = max(tol, 0.02)
    r_tip = _peak(rr, float(np.percentile(pf, 97)), 0.4, sig)
    r_root = _peak(rr, float(np.percentile(pf, 3)), 0.4, sig)
    depth = r_tip - r_root
    flank = {}
    band = (rr > r_root + 0.25 * depth) & (rr < r_tip - 0.25 * depth)
    for side, sel in (("leading", ds < 0), ("trailing", ds > 0)):
        q = band & sel
        A = np.c_[rr[q], np.ones(q.sum())]
        coef = np.linalg.lstsq(A, ds[q], rcond=None)[0]
        res = ds[q] - A @ coef
        keep = np.abs(res) < 3 * max(np.std(res), 1e-3)
        coef = np.linalg.lstsq(A[keep], ds[q][keep], rcond=None)[0]
        res = ds[q] - A @ coef
        c2 = np.polyfit(rr[q][keep], ds[q][keep], 2)
        flank[side] = dict(angle_deg=round(math.degrees(math.atan(abs(coef[0]))), 3),
                           rms_mm=round(float(np.sqrt(np.mean(res[keep] ** 2))), 4),
                           curvature_1_per_mm=round(float(2 * c2[0]), 5), coef=coef)
    m_free = px / math.pi
    m_std = min(WORM_MODULES, key=lambda v: abs(v - m_free))
    alpha = 0.5 * (flank["leading"]["angle_deg"] + flank["trailing"]["angle_deg"])
    # reference diameter: where the axial tooth thickness between the two fitted flank lines is p_x / 2
    # (robust: the flanks are well sampled, the narrow crest land is rounded by the scan)
    dc = flank["trailing"]["coef"] - flank["leading"]["coef"]
    r1 = float((px / 2 - dc[1]) / dc[0]) if abs(dc[0]) > 1e-6 else r_tip - m_std
    thick_tip_minus_m = float(np.polyval(dc, r_tip - m_std))
    for v in flank.values():
        v.pop("coef")
    d1 = 2 * r1; q = d1 / m_std
    q_std = min(STD_Q, key=lambda v: abs(v - q))
    w = dict(kind="worm", starts=z1, hand="right" if h > 0 else "left", lead=round(L, 4), axial_pitch=round(px, 4),
             module_fit=round(m_free, 4), module=m_std, tip_dia=round(2 * r_tip, 4), root_dia=round(2 * r_root, 4),
             ref_dia=round(d1, 4), tip_dia_standard=round(d1 + 2 * m_std, 4), root_dia_standard=round(d1 - 2.4 * m_std, 4),
             addendum_measured_in_modules=round((r_tip - r1) / m_std, 3), q=round(q, 3), q_standard=q_std if abs(q - q_std) < 0.15 else None,
             lead_angle_deg=round(math.degrees(math.atan(L / (math.pi * d1))), 3),
             axial_flank_deg=round(alpha, 3), flanks=flank, axial_thickness_at_tip_minus_m=round(thick_tip_minus_m, 4),
             depth_in_modules=round(depth / m_std, 3), length=round(zb - za, 2), phase_score=round(sc, 4),
             axis_point=[float(x) for x in a0 + za * u], axis_dir=[float(x) for x in u],
             profile_type="ZA (straight axial flanks)" if max(flank["leading"]["rms_mm"], flank["trailing"]["rms_mm"]) < max(3 * tol, 0.05)
             else "curved axial flanks (ZI/ZN/ZK)")
    info.update(lead=w["lead"], starts=z1, module_fit=w["module_fit"])
    why = []
    if abs(m_free - m_std) > 0.015 * m_std:
        why.append(f"axial pitch {px:.3f} mm is not pi x a standard module ({m_free:.3f})")
    if not 12.0 <= alpha <= 25.0:
        why.append(f"axial flank angle {alpha:.1f} deg is outside 12..25")
    if not 1.9 <= depth / m_std <= 2.7:
        why.append(f"tooth depth {depth / m_std:.2f} m is outside 1.9..2.7 m")
    if why:
        info["reason"] = "helical, but not a standard worm: " + "; ".join(why)
        info["candidate"] = w
        return None, info
    return w, info
