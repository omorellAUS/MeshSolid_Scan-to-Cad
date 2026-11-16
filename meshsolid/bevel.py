"""Straight bevel gear recognition (M12): teeth, cone angles, outer cone distance, module, face width.

Axis from the round slices (bore / hub, spline._axis).  Axial slices through the teeth show an N-fold
envelope (gear._slices) whose tip and root radii change linearly along the axis: the tip and root cones.
Each cone is fitted as a line r = a z + c in the meridian half-plane through the tip-land / root-land
points (band around the line, refit, 3 passes).  Tip and root lines meeting the axis at one point = teeth
converging at the apex (standard straight bevel; the mismatch is reported).  With the tip angle
delta_a = delta + theta_a, root angle delta_f = delta - theta_f and tan theta_a / tan theta_f = h_a / h_f
for a tooth system (h_f / h_a = 1.25 ISO 53 / AGMA full depth, 1.188 Gleason-like, 1.2), the pitch angle
delta follows; outer cone distance Re = L_tip,outer cos theta_a, module m = 2 Re sin delta / N.  The
system is the one whose module from the tooth depth agrees with the module from the pitch cone.  Spiral /
skew teeth (tooth phase turning along the face) are refused, not modelled.
"""
from __future__ import annotations

import math

import numpy as np

from .gear import STD_MODULE, _slices, cyl
from .spline import _axis

SYSTEMS = {"ISO 53 / AGMA full depth (hf = 1.25 m)": 1.25, "Gleason-like (hf = 1.188 m)": 1.188,
           "hf = 1.2 m": 1.2}


def _line_band(r, z, a, c, lo, hi, iters=4):
    for _ in range(iters):
        d = r - (a * z + c)
        k = (d > -lo) & (d < hi)
        if k.sum() < 200:
            return a, c, k, np.inf
        a, c = np.polyfit(z[k], r[k], 1)
        lo, hi = min(lo, 0.12), min(hi, 0.12)
    d = r[k] - (a * z[k] + c)
    return a, c, k, float(np.sqrt(np.mean(d ** 2)))


def _robust_line(z, r, iters=4):
    k = np.ones(len(z), bool)
    for _ in range(iters):
        a, c = np.polyfit(z[k], r[k], 1)
        d = r - (a * z + c); mad = np.median(np.abs(d[k] - np.median(d[k]))) + 1e-6
        k2 = np.abs(d) < 3.5 * 1.4826 * mad + 0.02
        if k2.sum() < 4 or (k2 == k).all():
            break
        k = k2
    return a, c


def analyse(mesh, tol=0.05, n_samples=600000, seed=0, min_teeth=6):
    """Returns (bevel dict | None, info)."""
    import trimesh
    info = {}
    P, fi = trimesh.sample.sample_surface(mesh, n_samples, seed=seed)
    P = np.asarray(P)
    a0, u = _axis(P)
    r, th, z = cyl(P, a0, u)
    sl = _slices(r, th, z, dz=0.5)
    sig = max(0.2, 4 * tol)
    good = [q for q in sl if q["h"] >= min_teeth and q["amp"] > sig]
    if len(good) < 6:
        info["reason"] = "no N-fold toothed slices"; return None, info
    hs = np.array([q["h"] for q in good])
    N = int(np.bincount(hs).argmax())
    run, cur = [], []
    zl = [q for q in sl]
    for q in zl:                                              # longest run of consecutive N slices
        if q["h"] == N and q["amp"] > sig:
            cur.append(q)
        else:
            if len(cur) > len(run):
                run = cur
            cur = []
    if len(cur) > len(run):
        run = cur
    if len(run) < 6:
        info["reason"] = f"toothed zone with {N} teeth too short"; return None, info
    zc = np.array([q["z"] for q in run]); rmax = np.array([q["rmax"] for q in run]); rmin = np.array([q["rmin"] for q in run])
    ph_all = np.array([q["phase"] for q in run])
    # slices through the back cone also cut the teeth (N-fold); there the outer radius grows toward the
    # apex.  The outer-radius maximum splits the zone; the tip-cone side is the one whose tip and root lines
    # meet the axis at one apex (tried both)
    i = int(np.argmax(rmax))
    cands = []
    for sel in (slice(i, None), slice(0, i + 1)):
        if len(zc[sel]) >= 4:
            c = _fit_side(P, a0, u, r, z, N, zc[sel], rmax[sel], rmin[sel], ph_all[sel], tol)
            if c is not None:
                cands.append(c)
    info.update(teeth=N, zone_slices=len(run))
    if not cands:
        info["reason"] = "no side of the toothed zone has falling tip and root radii"; return None, info
    g, extra = min(cands, key=lambda c: c[0]["apex_mismatch_mm"] / max(c[0]["outer_cone_distance"], 1e-6))
    info.update(extra)
    info.update(module_fit=g["module_fit"], pitch_angle_deg=g["pitch_angle_deg"])
    why = []
    if abs(g["tip_angle_deg"] - g["root_angle_deg"]) < 0.5 or g["tip_angle_deg"] < 3.0:
        why.append("tip radius constant along the axis: cylindrical gear, not bevel")
    if extra["twist_over_face_mm"] > max(0.3, 4 * tol):
        why.append(f"tooth phase turns {extra['twist_over_face_mm']:.2f} mm across the face: spiral/skew bevel, not supported")
    if abs(g["module_fit"] - g["module"]) > 0.02 * g["module"]:
        why.append(f"module {g['module_fit']:.3f} is not a standard module")
    if g["apex_mismatch_mm"] > max(0.03 * g["outer_cone_distance"], 0.5):
        why.append(f"tip and root cones miss each other's apex by {g['apex_mismatch_mm']:.2f} mm")
    if why:
        info["reason"] = "; ".join(why); info["candidate"] = g
        return None, info
    return g, info


def _fit_side(P, a0, u, r, z, N, zc, rmax, rmin, ph, tol):
    a_t, c_t = _robust_line(zc, rmax); a_r, c_r = _robust_line(zc, rmin)
    if a_t > 0:                                               # orient: apex toward +z (radius falls)
        u = -u; z = -z; zc = -zc; a_t = -a_t; a_r = -a_r
    if a_r >= 0:
        return None
    twist_mm = float(np.ptp(np.polyval(np.polyfit(zc, np.unwrap(ph) / N, 1), zc))) * float(np.median(rmax))
    # band fits on all points: past the outer edge / toe nothing else lies near the tip or root cone line
    a_t, c_t, kt, rms_t = _line_band(r, z, a_t, c_t, 0.5, 0.3)
    a_r, c_r, kr, rms_r = _line_band(r, z, a_r, c_r, 0.3, 0.5)
    if not (a_t < 0 and a_r < 0):
        return None
    zA_t, zA_r = -c_t / a_t, -c_r / a_r
    d_a = math.atan(abs(a_t)); d_f = math.atan(abs(a_r))
    if d_a <= d_f:
        return None
    A = a0 + zA_t * u
    L_tip = np.linalg.norm(P[kt] - A, axis=1)
    L_out, L_in = float(np.percentile(L_tip, 99.7)), float(np.percentile(L_tip, 0.3))
    best = None
    for name, c in SYSTEMS.items():
        f = lambda dd: math.tan(d_a - dd) * c - math.tan(dd - d_f)   # noqa: E731
        lo, hi = d_f, d_a
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if f(lo) * f(mid) <= 0:
                hi = mid
            else:
                lo = mid
        dl = 0.5 * (lo + hi)
        th_a = d_a - dl; Re = L_out * math.cos(th_a)
        m_pitch = 2 * Re * math.sin(dl) / N
        m_add = Re * math.tan(th_a)
        cand = dict(system=name, delta=dl, theta_a=th_a, Re=Re, m_pitch=m_pitch, m_add=m_add,
                    agree=abs(m_add - m_pitch) / m_pitch)
        if best is None or cand["agree"] < best["agree"]:
            best = cand
    dl, th_a, Re, m_free = best["delta"], best["theta_a"], best["Re"], best["m_pitch"]
    m_std = min(STD_MODULE, key=lambda v: abs(v - m_free))
    b = (L_out - L_in) * math.cos(th_a)
    N2 = N / math.tan(dl)
    g = dict(kind="bevel_straight", teeth=N, module_fit=round(m_free, 4), module=m_std,
             module_from_addendum=round(best["m_add"], 4), tooth_system=best["system"],
             pitch_angle_deg=round(math.degrees(dl), 3), tip_angle_deg=round(math.degrees(d_a), 3),
             root_angle_deg=round(math.degrees(d_f), 3), outer_cone_distance=round(Re, 3), face_width=round(b, 3),
             outer_pitch_dia=round(m_std * N, 3), outer_tip_dia_measured=round(2 * L_out * math.sin(d_a), 3),
             outer_tip_dia_standard=round(m_std * N + 2 * m_std * math.cos(dl), 3),
             apex_mismatch_mm=round(float(abs(zA_t - zA_r)), 3), tip_cone_rms=round(rms_t, 4),
             root_cone_rms=round(rms_r, 4),
             mate_teeth_90deg=int(round(N2)) if abs(N2 - round(N2)) < 0.15 else None,
             mate_teeth_90deg_fit=round(N2, 3), pressure_angle_deg=None,
             apex=[float(x) for x in A], axis_dir=[float(x) for x in u])
    return g, dict(twist_over_face_mm=round(twist_mm, 3))
