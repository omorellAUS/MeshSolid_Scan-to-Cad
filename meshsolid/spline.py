"""M9: straight (parallel-axis) splines on shafts and in hubs, recognised and snapped to standards.

  zone     angular radius envelope per 0.5 mm axial slice (max radius per angle bin for shafts, min for
           bores), N-fold periodic (N >= 6) with no twist along the axis
  side     external (shaft) / internal (hub) from the surface normals of the zone points
  form     flank points (20..80% of the tooth height) folded over the N teeth, centred on the radius
           maxima (shaft teeth / hub spaces), fitted two ways:
             straight-sided  r sin|phi| = B / 2  (sides parallel to the centre line, ISO 14)
             involute 30 deg |phi| = psi_b - inv(acos(r_b / r)) with r_b = m N cos 30 / 2 for every standard
                             module (DIN 5480 series, ANSI B92.1 m = 25.4 / P); residual along the base
                             tangent (the involute normal), in mm
  snap     the better form must explain the flanks (rms <= max(3 tol, 0.08 mm)); straight-sided -> ISO 14
           row (N, d, D, B); involute -> the module with the lowest residual, then the DIN 5480 reference
           diameter (shaft tip + 0.2 m, hub minor + 2 m) or the ANSI B92.1 major / minor diameter check."""
from __future__ import annotations

import math

import numpy as np

from . import standards as STD
from .gear import cyl, refine_axis, _roundness

A30 = math.radians(30.0)


def _inv(a):
    return np.tan(a) - a


def _axis(P):
    c = P.mean(0); _, V = np.linalg.eigh(np.cov((P - c).T))
    best = None
    for u0 in V.T[::-1]:
        a0, u = refine_axis(P, c, u0)
        rn = _roundness(P, a0, u)
        if best is None or rn > best[0]:
            best = (rn, a0, u)
    return best[1], best[2]


def _peak(r, near, span, sig):
    """Mode of the radii within near +- span, refined as the mean within 2.5 sigma of it."""
    k = np.abs(r - near) < span
    if k.sum() < 50:
        return near
    h, e = np.histogram(r[k], bins=max(20, int(2 * span / 0.01)))
    h = np.convolve(h, np.ones(5) / 5, "same")
    m = 0.5 * (e[np.argmax(h)] + e[np.argmax(h) + 1])
    for _ in range(3):
        q = np.abs(r - m) < 2.5 * sig
        if q.sum() > 20:
            m = float(np.mean(r[q]))
    return m


def _zone(r, th, z, nr, mode, dz=1.0, nb=360, tol=0.05):
    """nr: radial component of the surface normals; a bore (internal) envelope only uses inward-facing
    and flank points, a shaft envelope only outward-facing and flank points (hollow shafts, hub outsides)."""
    keep = nr < 0.3 if mode == "internal" else nr > -0.3
    r, th, z = r[keep], th[keep], z[keep]
    z0, z1 = np.percentile(z, [0.2, 99.8])
    edges = np.arange(z0, z1 + dz, dz)
    rows = []
    for a, b in zip(edges[:-1], edges[1:]):
        k = (z >= a) & (z < b)
        if k.sum() < 3 * nb:
            rows.append(None); continue
        bi = np.minimum((np.mod(th[k], 2 * np.pi) / (2 * np.pi) * nb).astype(int), nb - 1)
        env = np.full(nb, np.nan)
        if mode == "external":
            np.fmax.at(env, bi, r[k])
        else:
            env2 = np.full(nb, np.inf); np.minimum.at(env2, bi, r[k]); env = np.where(np.isinf(env2), np.nan, env2)
        g = np.flatnonzero(~np.isnan(env))
        if len(g) < 0.8 * nb:
            rows.append(None); continue
        env = np.interp(np.arange(nb), g, env[g], period=nb)
        c = np.fft.rfft(env - env.mean())
        pw = np.abs(c) ** 2
        N = int(np.argmax(pw[6:121])) + 6
        frac = pw[N::N].sum() / max(pw[1:].sum(), 1e-12)
        ok = np.ptp(env) > max(0.4, 8 * tol) and frac > 0.85
        rows.append((0.5 * (a + b), N, c[N], env.min(), env.max()) if ok else None)
    # longest run of slices with the same N
    best, cur = [], []
    for row in rows + [None]:
        if row is not None and (not cur or row[1] == cur[-1][1]):
            cur.append(row)
        else:
            if len(cur) > len(best):
                best = cur
            cur = [row] if row is not None else []
    return best


def analyse(mesh, tol=0.05, n_samples=600000, seed=0):
    """Returns (spline dict | None, info)."""
    import trimesh
    info = {}
    P, fi = trimesh.sample.sample_surface(mesh, n_samples, seed=seed)
    P = np.asarray(P); Nn = np.asarray(mesh.face_normals)[fi]
    a0, u = _axis(P)
    r, th, z = cyl(P, a0, u)
    radial = P - a0 - np.outer((P - a0) @ u, u)
    nr = np.einsum("ij,ij->i", Nn, radial / np.maximum(np.linalg.norm(radial, axis=1, keepdims=True), 1e-9))
    found = None
    for mode in ("external", "internal"):
        zr = _zone(r, th, z, nr, mode, tol=tol)
        if len(zr) >= 6 and (found is None or len(zr) > len(found[1])):
            found = (mode, zr)
    if found is None:
        info["reason"] = "no N-fold (N >= 6) axial zone"
        return None, info
    mode, zr = found
    N = zr[0][1]
    zc = np.array([q[0] for q in zr]); ph = np.unwrap(np.angle([q[2] for q in zr])) / N
    r_hi0 = float(np.median([q[4] for q in zr])); r_lo0 = float(np.median([q[3] for q in zr]))
    twist_mm = float(np.ptp(np.polyval(np.polyfit(zc, ph, 1), zc))) * r_hi0
    info.update(teeth=N, envelope=mode, zone_slices=len(zr), twist_over_zone_mm=round(twist_mm, 4))
    if twist_mm > max(0.1, 2 * tol):
        info["reason"] = f"periodic zone twists ({twist_mm:.2f} mm at the tip): helical, not a straight spline"
        return None, info
    za, zb = zc[0] - 0.5, zc[-1] + 0.5
    k = (z > za + 1.5) & (z < zb - 1.5)
    k &= (r > r_lo0 - 0.3) if mode == "external" else (r < r_hi0 + 0.3)
    k &= (r < r_hi0 + 0.3) & (r > r_lo0 - 0.3)
    k &= (nr < 0.3) if mode == "internal" else (nr > -0.3)
    k &= np.abs(Nn @ u) < 0.5                                   # side surfaces only (no end faces / chamfers)
    rr, tt, nn, PP = r[k], th[k], Nn[k], P[k]
    # side from the normals: radial component of the land points
    e1 = PP - a0 - np.outer((PP - a0) @ u, u); e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    land = (rr > r_hi0 - 0.15 * (r_hi0 - r_lo0)) | (rr < r_lo0 + 0.15 * (r_hi0 - r_lo0))
    side = "external" if np.median(np.einsum("ij,ij->i", nn[land], e1[land])) > 0 else "internal"
    # envelope bin j samples the angle (j + 0.5) * 2 pi / nb: half a bin of phase
    thc = math.pi / 360 - np.angle(np.mean([q[2] for q in zr])) / N
    wrap = lambda t: (np.mod((t - thc) * N + np.pi, 2 * np.pi) - np.pi) / N   # angle from the nearest tooth/space centre
    phi = wrap(tt)
    sig = max(tol, 0.02)
    r_hi = _peak(rr, r_hi0, 0.3, sig); r_lo = _peak(rr, r_lo0, 0.3, sig)
    h = r_hi - r_lo
    fk = (rr > r_lo + 0.2 * h) & (rr < r_hi - 0.2 * h)
    for _ in range(2):                         # centre on the flank pair (equal flanks: mean angle 0)
        thc += float(np.mean(phi[fk])); phi = wrap(tt)
    rf, af = rr[fk], np.abs(phi[fk])
    if fk.sum() < 300:
        info["reason"] = "too few flank points"
        return None, info

    def robust_rms(res):
        med = np.median(res); mad = 1.4826 * np.median(np.abs(res - med)) + 1e-9
        inl = np.abs(res - med) < 3 * mad
        return float(np.sqrt(np.mean((res[inl] - med) ** 2))), float(med), float(inl.mean())

    y = rf * np.sin(af)
    s_rms, s_half, s_inl = robust_rms(y)
    straight = {"rms_mm": round(s_rms, 4), "width_B": round(2 * s_half, 4), "inliers": round(s_inl, 3)}
    cands = [("DIN 5480", m, None) for m in STD.DIN5480_MODULES] + [("ANSI B92.1", STD.IN / P_, (P_, Ps)) for P_, Ps in STD.B92_1_PITCHES]
    inv_rows = []
    for std, m, pp in cands:
        rb = m * N * math.cos(A30) / 2
        if rb >= rf.min() - 1e-6 or rb < 0.5 * r_lo:
            continue
        res = rb * (af + _inv(np.arccos(rb / rf)))
        rms, psi_rb, inl = robust_rms(res)
        inv_rows.append({"standard": std, "module": m, "pitch": pp, "rb": rb, "rms_mm": rms, "psi_b": psi_rb / rb, "inliers": inl})
    inv_rows.sort(key=lambda d: d["rms_mm"])
    gate = max(3 * tol, 0.08)
    info.update(straight_fit=straight, involute_best=[{k_: (round(v, 4) if isinstance(v, float) else v) for k_, v in d.items()}
                                                      for d in inv_rows[:3]], gate_mm=gate)
    best_inv = inv_rows[0] if inv_rows else None
    # spline length: axial extent of the flank-band side-surface points (the slice zone is 1 mm coarse)
    kk = (z > za - 2.0) & (z < zb + 2.0) & (np.abs(Nn @ u) < 0.5) & (r > r_lo + 0.2 * h) & (r < r_hi - 0.2 * h)
    if kk.sum() > 100:
        za, zb = (float(v) for v in np.percentile(z[kk], [0.02, 99.98]))
    base = {"teeth": N, "side": side, "axis_point": [round(float(v), 4) for v in a0], "axis_dir": [round(float(v), 6) for v in u],
            "z_range_axis_frame": [round(float(za), 3), round(float(zb), 3)], "length": round(float(zb - za), 3),
            "major_diameter": round(2 * r_hi, 4), "minor_diameter": round(2 * r_lo, 4), "tooth_height": round(h, 4)}
    if s_rms <= gate and (best_inv is None or s_rms < best_inv["rms_mm"]):
        B = 2 * s_half
        rows = STD.iso14_match(N, 2 * r_lo, 2 * r_hi, B)
        lim = max(0.15, 3 * tol)
        snap = rows[0] if rows and rows[0]["misfit_mm"] <= lim else None
        out = dict(base, form="straight-sided", width_B=round(B, 4), flank_rms_mm=round(s_rms, 4),
                   standard="ISO 14" if snap else None, designation=snap["designation"] if snap else None,
                   series=snap["series"] if snap else None, snap=snap, nearest=rows[:2], snap_limit_mm=lim)
        info["accepted"] = True
        return out, info
    if best_inv is None or best_inv["rms_mm"] > gate:
        info["reason"] = (f"flanks are neither straight-sided nor a 30 deg involute of a standard module "
                          f"(rms {s_rms:.3f} / {best_inv['rms_mm'] if best_inv else float('nan'):.3f} mm, gate {gate:.2f})")
        return None, info
    def check(b):
        m = b["module"]
        if b["standard"] == "DIN 5480":
            est = 2 * r_hi + 0.2 * m if side == "external" else 2 * r_lo + 2 * m
            dB = min(STD.DIN5480_DB, key=lambda v: abs(v - est))
            return abs(dB - est) < max(0.3 * m, 0.15), {"reference_diameter_dB": dB, "reference_diameter_estimate": round(est, 4)}
        P_, Ps = b["pitch"]
        exp = (N + 1) * m if side == "external" else (N - 1) * m
        meas = 2 * r_hi if side == "external" else 2 * r_lo
        return abs(exp - meas) < max(0.15, 3 * tol), {"expected_diameter": round(exp, 4), "measured_diameter": round(meas, 4)}
    # lowest flank residual among the standard sizes whose diameters also agree (flank shape alone barely
    # separates neighbouring modules over a 1-module tooth height)
    pick = None
    for b in inv_rows:
        if b["rms_mm"] > gate:
            break
        ok, extra = check(b)
        if ok:
            pick = (b, extra); break
    if pick is None:
        b = best_inv; ok, extra = check(b)
    else:
        b, extra = pick; ok = True
    m = b["module"]
    rp = m * N / 2
    s_ref = 2 * rp * (b["psi_b"] - float(_inv(A30)))          # tooth (shaft) / space (hub) thickness at d
    xm = (s_ref - math.pi * m / 2) / (2 * math.tan(A30))
    out = dict(base, form="involute 30 deg", module=round(m, 5), pitch_diameter=round(2 * rp, 4),
               base_diameter=round(2 * b["rb"], 4), flank_rms_mm=round(b["rms_mm"], 4),
               thickness_at_pitch=round(s_ref, 4), profile_shift_xm=round(xm, 4), standard=b["standard"] if ok else None,
               module_standard=b["standard"], diameters_consistent=ok, **extra)
    if b["standard"] == "DIN 5480":
        dB = extra["reference_diameter_dB"]
        out.update(designation=(f"{'W' if side == 'external' else 'N'} {dB:g} x {m:g} x {N}" if ok else None),
                   xm_din=round((dB - m * N) / 2 - 0.55 * m, 4))
    else:
        P_, Ps = b["pitch"]
        out.update(diametral_pitch=f"{P_:g}/{Ps:g}", designation=(f"{N}T {P_:g}/{Ps:g}" if ok else None))
    info["accepted"] = True
    return out, info
