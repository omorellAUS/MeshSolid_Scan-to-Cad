"""Involute gear recognition on scan meshes (spur and helical, external).

Detection (like threads): on the part axis, the per-angle outer radius of each axial slice has a strong
N-th harmonic inside a toothed zone; the phase of that harmonic drifts linearly with z for a helical
gear (lead, hand).  Fit: flank points are de-twisted into one transverse plane (theta' = theta - w z),
folded onto one tooth, and fitted with an involute of base radius rb plus a tooth thickness (both flanks,
residual per flank).  Design intent: normal DP (US) or module (metric) and normal pressure angle are
snapped to standard values whose base pitch matches the fitted base radius; helix angle follows from
the lead.  The snapped values rebuild an exact involute helical gear (see build_gear_cut).

Frame: axis u through point a0; theta measured in the right-handed frame (e1, e2, u).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict

import numpy as np

STD_DP = [8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 36, 40, 48, 64]          # normal diametral pitch
STD_MODULE = [0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0, 1.125, 1.25, 1.375, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0,
              3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0]
STD_PA = [14.5, 17.5, 20.0, 22.5, 25.0]
# addendum / dedendum systems (in units of the normal module)
# (fine-pitch AGMA dedendum is 1.2/P + 0.002 in, i.e. depends on the module: see _systems)
SYSTEMS = {"AGMA full depth": (1.0, 1.25), "AGMA full depth, shaved/ground": (1.0, 1.35),
           "ISO 53 profile A": (1.0, 1.25), "ISO 53 profile D": (1.0, 1.4), "AGMA 20 deg stub": (0.8, 1.0)}


def _systems(mn):
    out = dict(SYSTEMS)
    out["AGMA fine pitch (1.2/P + 0.002 in)"] = (1.0, 1.2 + 0.0508 / mn)
    return out


def inv(a):
    return math.tan(a) - a


def frame(u):
    u = np.asarray(u, float); u = u / np.linalg.norm(u)
    e1 = np.cross(u, [1.0, 0, 0])
    if np.linalg.norm(e1) < 0.1:
        e1 = np.cross(u, [0, 1.0, 0])
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(u, e1), u


def cyl(P, a0, u):
    e1, e2, u = frame(u)
    d = np.asarray(P, float) - a0
    z = d @ u; x = d @ e1; y = d @ e2
    return np.hypot(x, y), np.arctan2(y, x), z


@dataclass
class GearSpec:
    teeth: int
    z0: float                      # toothed zone (axial, incl. chamfers) in the axis frame
    z1: float
    axis_point: list
    axis_dir: list
    hand: str                      # "left" | "right" | "spur"
    lead: float                    # mm (inf for spur)
    w: float                       # d theta / d z (rad/mm), signed in the axis frame
    r_tip: float
    r_root: float
    chamfers: dict = field(default_factory=dict)
    fit: dict = field(default_factory=dict)         # free fit (rb, thickness, residuals)
    snapped: dict = field(default_factory=dict)     # standard values + derived geometry
    confidence: dict = field(default_factory=dict)
    zone_faces: object = None


# ----------------------------------------------------------------------------- detection
def refine_axis(P, a0, u, iters=3, exclude=None):
    """Axis through the centres of circle fits of axial slices (round parts only)."""
    for _ in range(iters):
        e1, e2, u = frame(u)
        d = P - a0; z = d @ u; x = d @ e1; y = d @ e2
        zs = np.linspace(z.min() + 1, z.max() - 1, 80)
        C = []
        for a, b in zip(zs[:-1], zs[1:]):
            k = (z >= a) & (z < b)
            if exclude is not None:
                k &= ~((z >= exclude[0]) & (z <= exclude[1]))
            if k.sum() < 300:
                continue
            A = np.c_[2 * x[k], 2 * y[k], np.ones(k.sum())]
            s = np.linalg.lstsq(A, x[k] ** 2 + y[k] ** 2, rcond=None)[0]
            rr = np.hypot(x[k] - s[0], y[k] - s[1])
            if np.std(rr) > 0.05 * np.mean(rr):      # not round (teeth, flats): skip
                continue
            C.append(a0 + (a + b) / 2 * u + s[0] * e1 + s[1] * e2)
        C = np.array(C)
        if len(C) < 4:
            break
        c = C.mean(0); _, _, Vt = np.linalg.svd(C - c)
        u = Vt[0] * np.sign(Vt[0] @ u); a0 = c
    return a0, u / np.linalg.norm(u)


def _roundness(P, a0, u, nb=60):
    """Area fraction of axial slices whose radius about the axis varies < 10% (round, incl. gear slices)."""
    r, th, z = cyl(P, a0, u)
    zs = np.linspace(z.min(), z.max(), nb + 1)
    ok = 0
    for a, b in zip(zs[:-1], zs[1:]):
        k = (z >= a) & (z < b)
        if k.sum() < 50:
            continue
        bi = ((th[k] + np.pi) / (2 * np.pi) * 36).astype(int) % 36
        rmax = np.zeros(36); np.maximum.at(rmax, bi, r[k])
        if (rmax > 0).mean() > 0.8 and np.std(rmax[rmax > 0]) < 0.1 * np.mean(rmax[rmax > 0]):
            ok += k.sum()
    return ok / len(P)


def _slices(r, th, z, dz=0.5, nb=720):
    zb = np.arange(z.min(), z.max() + dz, dz)
    out = []
    for a in zb[:-1]:
        k = (z >= a) & (z < a + dz)
        if k.sum() < 150:
            continue
        bi = ((th[k] + np.pi) / (2 * np.pi) * nb).astype(int) % nb
        rmax = np.full(nb, np.nan); np.fmax.at(rmax, bi, r[k])
        good = ~np.isnan(rmax)
        if good.mean() < 0.6:
            continue
        rr = np.where(good, rmax, np.nanmedian(rmax))
        F = np.fft.rfft(rr - rr.mean()); amp = np.abs(F) / nb * 2
        h = int(np.argmax(amp[5:nb // 4]) + 5)
        out.append(dict(z=a + dz / 2, h=h, cover=float(good.mean()), amp=float(amp[h]), phase=float(np.angle(F[h])),
                        rmax=float(np.nanmax(rmax)), rmin=float(np.nanmin(rmax)), rmed=float(np.nanmedian(rmax))))
    return out


GEAR_MIN_COVER = 0.75     # scanned pinion p01: 0.88 (gaps in the scan); faceted aileron AIL 2: 0.61


def detect_gear(P, N_, a0, u, min_teeth=6, tol=0.05):
    """P: surface samples, N_: their normals.  Returns GearSpec (unfitted) or None."""
    r, th, z = cyl(P, a0, u)
    S = _slices(r, th, z)
    if not S:
        return None
    amp_min = max(0.2, 3 * tol)
    tooth = [s["amp"] > amp_min and s["amp"] > 0.15 * (s["rmax"] - s["rmin"] + 1e-9) * 0 + amp_min for s in S]
    Hs = [s["h"] for s in S]
    cand = [h for h, t in zip(Hs, tooth) if t]
    if not cand:
        return None
    N = int(np.bincount(cand).argmax())
    if N < min_teeth:
        return None
    best, i = (0, 0, -1), 0
    while i < len(S):
        if tooth[i] and Hs[i] == N:
            j = i
            while j + 1 < len(S) and tooth[j + 1] and Hs[j + 1] == N:
                j += 1
            if j - i > best[0]:
                best = (j - i, i, j)
            i = j + 1
        else:
            i += 1
    _, i0, i1 = best
    G = S[i0:i1 + 1]
    if len(G) < 6:
        return None
    if np.median([g["cover"] for g in G]) < GEAR_MIN_COVER:   # a gear is a ring of teeth round the axis
        return None
    zg = np.array([g["z"] for g in G])
    core = slice(len(G) // 6, len(G) - len(G) // 6)
    ph = np.unwrap(np.array([g["phase"] for g in G]))
    k = np.polyfit(zg[core], ph[core], 1)[0]
    w = -k / N                                   # rad / mm
    lead = 2 * np.pi / abs(w) if abs(w) > 1e-6 else float("inf")
    hand = "spur" if not np.isfinite(lead) or lead > 1e5 else ("right" if w > 0 else "left")
    r_tip = float(np.median([g["rmax"] for g in G][core]))
    r_root = float(np.median([g["rmin"] for g in G][core]))
    # zone edges at 0.25 mm resolution: first/last z where samples reach beyond the root by > 25% depth
    dz = 0.5
    z0, z1 = float(zg[0] - dz / 2), float(zg[-1] + dz / 2)
    return GearSpec(teeth=N, z0=z0, z1=z1, axis_point=np.asarray(a0, float).tolist(), axis_dir=np.asarray(u, float).tolist(),
                    hand=hand, lead=float(lead), w=float(w), r_tip=r_tip, r_root=r_root)


# ----------------------------------------------------------------------------- ends / chamfers
def refine_ends(P, g, step=0.1):
    """Axial ends of the tooth spaces at `step` resolution, and the end chamfer of the tooth tips.

    A tooth space exists in a slice when samples reach the root band (r < root + 30% depth).  The tip
    chamfer is a straight line fitted to the per-slice crest radius in the last 1.5 mm of each end
    (the scan rounds tip edges, so the chamfer is reported with low confidence and its angle is
    snapped to 15/30/45/60 deg when within 7 deg)."""
    r, th, z = cyl(P, np.asarray(g.axis_point), np.asarray(g.axis_dir))
    depth = g.r_tip - g.r_root
    zz = np.arange(g.z0 - 2.0, g.z1 + 2.0, step)
    has_space, crest = [], []
    for a in zz:
        k = (z >= a) & (z < a + step)
        has_space.append(int(np.sum(r[k] < g.r_root + 0.3 * depth)) >= 3)
        kk = k & (r > g.r_root + 0.5 * depth) & (r < g.r_tip + 0.3 * depth)
        crest.append(float(np.percentile(r[kk], 98)) if kk.sum() >= 20 else np.nan)
    has_space = np.array(has_space); crest = np.array(crest)
    idx = np.where(has_space)[0]
    if len(idx) == 0:
        return {}
    # longest run of slices with spaces
    runs = np.split(idx, np.where(np.diff(idx) > 2)[0] + 1)
    run = max(runs, key=len)
    ze0, ze1 = float(zz[run[0]]), float(zz[run[-1]] + step)
    out = {"z_end0": ze0, "z_end1": ze1, "face_width": ze1 - ze0, "chamfers": {}}
    for name, sgn, zend in (("end0", 1, ze0), ("end1", -1, ze1)):
        # crest radius vs distance from the end, inside the tooth zone
        dd = (zz + step / 2 - zend) * sgn
        k = (dd > 0) & (dd < 1.5) & ~np.isnan(crest)
        if k.sum() < 4:
            continue
        low = k & (crest < g.r_tip - 0.05)
        if low.sum() < 2:
            out["chamfers"][name] = None
            continue
        A = np.polyfit(dd[low], crest[low], 1)
        if A[0] <= 0.05:
            out["chamfers"][name] = None
            continue
        ang = math.degrees(math.atan(A[0]))           # angle of the chamfer line to the axis
        ang_s = min((15.0, 30.0, 45.0, 60.0), key=lambda a: abs(a - ang))
        ang_use = ang_s if abs(ang_s - ang) <= 7.0 else ang
        r_end = float(np.polyval(A, 0.0))
        radial = max(g.r_tip - r_end, 0.0)
        out["chamfers"][name] = {"angle_to_axis_deg_fit": round(ang, 2), "angle_to_axis_deg": round(ang_use, 2),
                                 "radial_mm": round(radial, 3), "axial_mm": round(radial / math.tan(math.radians(ang_use)), 3),
                                 "confidence": "low (scan rounds tip edges)"}
    return out


# ----------------------------------------------------------------------------- fit
def _flank_points(P, N_, g, trim=2.5, lo=0.30, hi=0.15, r_lo=None, r_hi=None):
    """Flank samples, de-twisted to the z = 0 transverse plane.  The radial band is either
    [root + lo*depth, tip - hi*depth] or explicitly [r_lo, r_hi] (the mid-flank band)."""
    r, th, z = cyl(P, np.asarray(g.axis_point), np.asarray(g.axis_dir))
    e1, e2, u = frame(np.asarray(g.axis_dir))
    depth = g.r_tip - g.r_root
    a = g.r_root + lo * depth if r_lo is None else r_lo
    b = g.r_tip - hi * depth if r_hi is None else r_hi
    za, zb = g.z0, g.z1
    if g.chamfers:
        za, zb = g.chamfers.get("z_end0", za), g.chamfers.get("z_end1", zb)
    k = (z > za + trim) & (z < zb - trim) & (r > a) & (r < b)
    thp = th[k] - g.w * z[k]                                   # de-twist to the z = 0 transverse plane
    # flank side from the normal: tangential component in the transverse plane
    et = np.c_[-np.sin(th[k]), np.cos(th[k])]
    nt = (N_[k] @ e1) * et[:, 0] + (N_[k] @ e2) * et[:, 1]
    return r[k], thp, np.sign(nt), z[k]


def active_band(g, rb, lo=0.2, hi=0.8, r_tip=None):
    """Mid-flank band: lo..hi of the active profile height, from the form radius (the base circle or
    just above the root fillet, whichever is larger) to the tip."""
    rt = g.r_tip if r_tip is None else r_tip
    r_form = max(rb, g.r_root + 0.25 * (rt - g.r_root))
    return r_form + lo * (rt - r_form), r_form + hi * (rt - r_form), r_form


def tip_crest(P, g, fit, trim=2.5):
    """Tip radius from the crest centre: samples within the middle 40% of the tooth top (angularly),
    which the scan's edge rounding does not pull down."""
    r, th, z = cyl(P, np.asarray(g.axis_point), np.asarray(g.axis_dir))
    za, zb = g.chamfers.get("z_end0", g.z0), g.chamfers.get("z_end1", g.z1)
    N = g.teeth
    k = (z > za + trim) & (z < zb - trim) & (r > g.r_root + 0.85 * (g.r_tip - g.r_root))
    d = np.angle(np.exp(1j * N * (th[k] - g.w * z[k] - fit["clocking_rad"]))) / N
    half = _half_thickness(np.array([g.r_tip]), fit["rb"], fit["psi_b"])[0]
    c = np.abs(d) < 0.4 * max(half, 1e-4)
    if c.sum() < 50:
        return None
    rr = r[k][c]
    return {"r_tip_crest": float(np.median(rr)), "p90": float(np.percentile(rr, 90)), "n": int(c.sum()),
            "r_tip_slices": float(g.r_tip)}


def _half_thickness(r, rb, psi_b):
    """Half angular tooth thickness at radius r for an involute tooth of base radius rb."""
    a = np.arccos(np.clip(rb / np.maximum(r, rb), -1, 1))
    return psi_b - (np.tan(a) - a)


def fit_involute(r, thp, side, N, rb0, iters=2):
    """Least squares (c0, rb, psi_b) on de-twisted flank points.  side = +1 for flank points whose
    normal points to +theta (the flank on the +theta side of a tooth faces +theta)."""
    from scipy.optimize import least_squares
    pitch = 2 * np.pi / N
    # initial clocking: circular mean of N*theta' weighted by side, tooth centres sit between the flanks
    c0 = float(np.angle(np.mean(np.exp(1j * N * thp)))) / N

    def resid(p, rr, tt, ss):
        c, rb, psi = p
        d = np.angle(np.exp(1j * N * (tt - c))) / N          # angular offset from the nearest tooth centre
        return (np.abs(d) - _half_thickness(rr, rb, psi)) * rr * np.where(np.sign(d) == ss, 1.0, 3.0)

    p0 = [c0, rb0, 0.5 * pitch / 2 + inv(math.acos(min(rb0 / np.median(r), 0.999)))]
    keep = np.ones(len(r), bool)
    for _ in range(iters):
        sol = least_squares(resid, p0, args=(r[keep], thp[keep], side[keep]), loss="soft_l1", f_scale=0.05)
        p0 = sol.x
        res_all = resid(p0, r, thp, side)
        keep = np.abs(res_all) < max(0.15, 4 * np.median(np.abs(res_all)))
    c, rb, psi = p0
    d = np.angle(np.exp(1j * N * (thp - c))) / N
    res = (np.abs(d) - _half_thickness(r, rb, psi)) * r
    out = {"clocking_rad": float(c), "rb": float(rb), "psi_b": float(psi), "points": int(len(r)), "inliers": int(keep.sum())}
    for name, m in (("flank_plus", d > 0), ("flank_minus", d < 0)):
        mm = m & keep
        if mm.any():
            out[name] = {"rms": round(float(np.sqrt(np.mean(res[mm] ** 2))), 4), "p95": round(float(np.percentile(np.abs(res[mm]), 95)), 4),
                         "n": int(mm.sum())}
    # each flank side separately (asymmetry / thickness check)
    for name, sgn in (("psi_b_plus", 1), ("psi_b_minus", -1)):
        mm = keep & (np.sign(d) == sgn)
        if mm.sum() > 50:
            out[name] = float(np.median(np.abs(d[mm]) + inv_arr(np.arccos(np.clip(rb / np.maximum(r[mm], rb), -1, 1)))))
    return out


def inv_arr(a):
    return np.tan(a) - a


def snap(g, fit, tol, tip=None):
    """Standard (pitch system, PA, helix, tooth system) whose geometry matches the free fit."""
    N = g.teeth; rb = fit["rb"]; L = g.lead
    spur = g.hand == "spur"
    cands = [("DP", dp, 25.4 / dp) for dp in STD_DP] + [("module", m, m) for m in STD_MODULE]
    rows = []
    for kind, val, mn in cands:
        sb = 0.0 if spur else math.pi * N * mn / L
        if sb >= 0.9:
            continue
        beta = math.asin(sb)
        rp = N * mn / (2 * math.cos(beta))
        if not (g.r_root < rp < g.r_tip):
            continue
        at_fit = math.acos(min(rb / rp, 1.0))
        an_fit = math.degrees(math.atan(math.tan(at_fit) * math.cos(beta)))
        for pa in STD_PA:
            at = math.atan(math.tan(math.radians(pa)) / math.cos(beta))
            rb_s = rp * math.cos(at)
            rows.append(dict(system=kind, value=val, mn=mn, pa=pa, beta=math.degrees(beta), rp=rp, rb=rb_s,
                             drb=rb_s - rb, an_fit=an_fit))
    if not rows:
        return {}
    # base radius must match; among near-equal matches prefer the pitch whose addendum is standard
    for r_ in rows:
        ha = (g.r_tip - r_["rp"]) / r_["mn"]
        r_["addendum_ratio"] = ha
        r_["score"] = abs(r_["drb"]) / max(tol, 1e-3) + 2.0 * min(abs(ha - 1.0), abs(ha - 0.8)) + (0.3 if r_["system"] == "module" else 0.0)
    rows.sort(key=lambda r_: r_["score"])
    b = rows[0]
    mn, beta, rp = b["mn"], math.radians(b["beta"]), b["rp"]
    at = math.atan(math.tan(math.radians(b["pa"])) / math.cos(beta))
    rb_s = rp * math.cos(at)
    # tooth thickness from the fit at the pitch circle -> profile shift
    psi_p = fit["psi_b"] - inv(math.acos(min(rb / rp, 1.0)))
    s_t = 2 * rp * psi_p; s_n = s_t * math.cos(beta)
    pa_r = math.radians(b["pa"])
    x = (s_n / mn - math.pi / 2) / (2 * math.tan(pa_r))
    thinning = math.pi / 2 * mn - s_n                     # normal-plane thinning vs a standard x = 0 tooth
    # tip: the scan rounds the tips, so the crest-centre radius is compared with the standard tips of two
    # readings of the thin tooth: (a) profile shift x (tip moves with x), (b) x = 0 with the tooth thinned
    # for backlash (tip stays standard).  Root: measured directly (concave, not rounded) vs standard.
    r_tip_meas = tip["r_tip_crest"] if tip else g.r_tip
    lim = 0.1 * mn                                       # snap only within a tenth of a module
    hyps = []
    for name, (ha_s, hf_s) in _systems(mn).items():
        for reading, xh in (("profile shift", x), ("x = 0, thinned for backlash", 0.0)):
            hyps.append(dict(system=name, reading=reading, x=xh, ha=ha_s, hf=hf_s,
                             r_tip=rp + (ha_s + xh) * mn, r_root=rp - (hf_s - xh) * mn))
    ht = min(hyps, key=lambda h: (round(abs(h["r_tip"] - r_tip_meas), 4), h["reading"] != "profile shift"))
    tip_snapped = abs(ht["r_tip"] - r_tip_meas) <= lim
    roots = [h for h in hyps if h["reading"] == ht["reading"]]
    hr = min(roots, key=lambda h: abs(h["r_root"] - g.r_root))
    root_snapped = abs(hr["r_root"] - g.r_root) <= lim
    ha = (r_tip_meas - rp) / mn; hf = (rp - g.r_root) / mn
    sysname = hr["system"] if root_snapped else ht["system"]
    L_s = float("inf") if spur else 2 * math.pi * rp / math.tan(beta)
    xr = ht["x"]
    # span measurement over k teeth (normal plane) and over-pins check, from the snapped tooth (thickness as fitted)
    kspan = max(2, int(round(N * at / math.pi + 0.5)))
    W = mn * math.cos(pa_r) * ((kspan - 0.5) * math.pi + N * inv(at)) + 2 * x * mn * math.sin(pa_r)
    pins = over_pins(N, mn, b["pa"], beta, s_n)
    psi_b_s = s_t / (2 * rp) + inv(at)
    build = {"rb": rb_s, "psi_b": psi_b_s, "r_tip": ht["r_tip"] if tip_snapped else r_tip_meas,
             "r_root": hr["r_root"] if root_snapped else g.r_root, "w": 0.0 if spur else math.copysign(2 * math.pi / L_s, g.w),
             "lead": L_s}
    alt = [dict(system=r_["system"], value=r_["value"], pa=r_["pa"], beta_deg=round(r_["beta"], 3), drb_mm=round(r_["drb"], 4),
                score=round(r_["score"], 2)) for r_ in rows[1:5]]
    return {"pitch_system": b["system"], "normal_DP": round(25.4 / mn, 4) if b["system"] == "DP" else None,
            "normal_module": round(mn, 5), "transverse_module": round(mn / math.cos(beta), 5),
            "normal_pressure_angle_deg": b["pa"], "pressure_angle_fit_deg": round(b["an_fit"], 3),
            "helix_angle_pitch_deg": round(math.degrees(beta), 4), "lead_mm": round(L_s, 3) if np.isfinite(L_s) else None,
            "lead_measured_mm": round(L, 2) if np.isfinite(L) else None,
            "pitch_diameter": round(2 * rp, 4), "base_diameter": round(2 * rb_s, 4), "base_diameter_fit": round(2 * rb, 4),
            "profile_shift_x_from_thickness": round(x, 4), "tooth_thinning_normal_mm": round(thinning, 4),
            "thin_tooth_reading": ht["reading"], "profile_shift_x": round(xr, 4),
            "addendum_ratio_measured": round(ha, 4), "dedendum_ratio_measured": round(hf, 4),
            "tooth_system": sysname, "tooth_system_ha_hf": [round(ht["ha"], 4), round(hr["hf"] if root_snapped else hf + ht["x"], 4)],
            "tip_diameter_crest": round(2 * r_tip_meas, 4), "tip_diameter_slices_max": round(2 * g.r_tip, 4),
            "tip_diameter_standard": round(2 * ht["r_tip"], 4), "tip_snapped": bool(tip_snapped),
            "root_diameter_measured": round(2 * g.r_root, 4), "root_diameter_standard": round(2 * hr["r_root"], 4),
            "root_snapped": bool(root_snapped), "snap_limit_mm": round(lim, 4),
            "span": {"teeth_spanned": kspan, "W_normal_mm": round(W, 4)}, "over_pins": pins,
            "tooth_thickness_normal_at_pitch": round(s_n, 4), "alternatives": alt, "build": build}


def over_pins(N, mn, pa_deg, beta, s_n, pin=None):
    """Measurement over two pins (balls, for a helical gear) for normal tooth thickness s_n at the pitch
    circle: inv(phi) = s_t/d + inv(at) + dp/(db cos(beta_b)) - pi/N (transverse plane)."""
    an = math.radians(pa_deg)
    at = math.atan(math.tan(an) / math.cos(beta))
    rp = N * mn / (2 * math.cos(beta)); rb = rp * math.cos(at)
    bb = math.atan(math.tan(beta) * math.cos(at))
    dp = pin if pin else 1.728 * mn                          # AGMA preferred pin for external gears (1.728/P)
    s_t = s_n / math.cos(beta)
    inv_c = s_t / (2 * rp) + inv(at) + dp / (2 * rb * math.cos(bb)) - math.pi / N
    phi = 0.6
    for _ in range(80):
        f = math.tan(phi) - phi - inv_c
        phi = max(1e-3, phi - f / (math.tan(phi) ** 2 + 1e-12))
    R = rb / math.cos(phi)
    M = 2 * R + dp if N % 2 == 0 else 2 * R * math.cos(math.pi / (2 * N)) + dp
    return {"pin_diameter_mm": round(dp, 4), "M_over_pins_mm": round(M, 4),
            "note": "balls, fitted tooth thickness; odd tooth count uses the cos(90/N) correction"}


# ----------------------------------------------------------------------------- transverse space profile
def _flank_xy(rb, psi_b, r0, r1, n=600):
    """+ flank of tooth 0 (radial line below rb, involute above) from r0 to r1, as (n, 2) points and radii."""
    rs = np.linspace(r0, r1, n)
    a = np.arccos(np.clip(rb / np.maximum(rs, rb), -1, 1))
    th = psi_b - (np.tan(a) - a)
    return np.c_[rs * np.cos(th), rs * np.sin(th)], rs


def _fillet_geom(rb, psi_b, r_root, rho, N):
    """Root fillet of radius rho tangent to the root circle and to the flank (involute, or the radial line
    below the base circle).  Returns dict(fc = centre angle, rc = centre radius, rT = flank tangent radius,
    T = tangent point, G = root tangent point) or None when it does not fit in the space."""
    pitch = 2 * math.pi / N
    rc = r_root + rho
    if rho <= 0:
        return None
    Fp, rs = _flank_xy(rb, psi_b, max(r_root - 0.01, 0.1), rc + rho + 0.5, 1500)

    def dist(fc):
        c = rc * np.array([math.cos(fc), math.sin(fc)])
        d = np.linalg.norm(Fp - c, axis=1)
        i = int(np.argmin(d))
        return d[i] - rho, i
    lo = math.atan2(*_flank_xy(rb, psi_b, rc, rc, 1)[0][0][::-1])
    hi = pitch / 2
    if dist(hi)[0] < 0:
        return None
    for _ in range(60):
        m = 0.5 * (lo + hi)
        if dist(m)[0] < 0:
            lo = m
        else:
            hi = m
    fc = hi
    _, i = dist(fc)
    T = Fp[i]
    G_ = r_root * np.array([math.cos(fc), math.sin(fc)])
    if fc >= pitch / 2 - 1e-6:
        return None
    return {"fc": fc, "rc": rc, "rT": float(rs[i]), "T": T, "G": G_}


def space_polyline(rb, psi_b, r_root, rho, N, r_top, n=60):
    """Dense 2D polyline (x, y) of one tooth space (tooth 0 centred at angle 0), from the + flank of tooth 0
    at r_top down to the root and up the - flank of tooth 1."""
    pitch = 2 * math.pi / N
    f = _fillet_geom(rb, psi_b, r_root, rho, N)
    if f is None:
        raise ValueError("root fillet does not fit")
    up, _ = _flank_xy(rb, psi_b, f["rT"], r_top, n)
    up = up[::-1]
    C = f["rc"] * np.array([math.cos(f["fc"]), math.sin(f["fc"])])
    a0_, a1_ = math.atan2(*(f["T"] - C)[::-1]), math.atan2(*(f["G"] - C)[::-1])
    da = math.remainder(a1_ - a0_, 2 * math.pi)
    tt = np.linspace(0, 1, 12)[1:]
    arc = C + rho * np.c_[np.cos(a0_ + tt * da), np.sin(a0_ + tt * da)]
    rootang = np.linspace(f["fc"], pitch - f["fc"], 20)[1:-1]
    root = r_root * np.c_[np.cos(rootang), np.sin(rootang)]
    half = np.vstack([up, arc])
    c, s_ = math.cos(pitch), math.sin(pitch)
    mir = np.c_[half[:, 0] * c + half[:, 1] * s_, half[:, 0] * s_ - half[:, 1] * c][::-1]
    return np.vstack([half, root, mir])


def fit_root_fillet(P, g, rb, psi_b, clock, trim=2.5):
    """Root fillet radius: grid search on the de-twisted samples below the form radius."""
    from scipy.spatial import cKDTree
    r, th, z = cyl(P, np.asarray(g.axis_point), np.asarray(g.axis_dir))
    za, zb = g.chamfers.get("z_end0", g.z0), g.chamfers.get("z_end1", g.z1)
    N = g.teeth; pitch = 2 * np.pi / N
    k = (z > za + trim) & (z < zb - trim) & (r < min(rb, g.r_root + 0.35 * (g.r_tip - g.r_root))) & (r > g.r_root - 0.3)
    t = np.mod(th[k] - g.w * z[k] - clock, pitch)               # tooth 0 at 0, space centre at pitch/2
    X = np.c_[r[k] * np.cos(t), r[k] * np.sin(t)]
    mn_ = 2 * (rb + 1e-9) / N                                   # rough module scale
    best = None
    for rho in np.linspace(0.05, 0.6, 23) * mn_ * 1.0:
        try:
            Q = space_polyline(rb, psi_b, g.r_root, rho, N, rb + 0.01, n=10)
        except ValueError:
            continue
        # densify
        Q = np.vstack([np.linspace(Q[i], Q[i + 1], 6, endpoint=False) for i in range(len(Q) - 1)])
        dd, _ = cKDTree(Q).query(X)
        sc = float(np.mean(np.minimum(dd, 0.15) ** 2))
        if best is None or sc < best[0]:
            best = (sc, float(rho), float(np.sqrt(np.mean(dd[dd < 0.15] ** 2))) if (dd < 0.15).any() else None, int(len(X)))
    if best is None:
        return None
    return {"rho": best[1], "rho_over_module": round(best[1] / mn_, 3), "rms": best[2], "n": best[3]}


# ----------------------------------------------------------------------------- OCC
def build_gear_cut(N, rb, psi_b, r_root, rho, r_tip, w, clock, z_a, z_b, a0, u):
    """Solid compound of the N tooth spaces of an exact involute (helical) gear, in world coordinates.
    Flanks: the involute (B-spline through 60 exact points, < 1e-5 mm) swept along a helix of
    twist w rad/mm (exact helicoid by construction, checked < 2e-5 mm); radial line below the base
    circle, root fillet arc, root circle arc.  Above the tip the space is closed radially so tooth
    tops cannot cross; the blank's tip cylinder then trims it."""
    from OCP.gp import gp_Pnt, gp_Dir, gp_Ax3, gp_Ax1, gp_Pnt2d, gp_Dir2d, gp_Trsf, gp_Vec
    from OCP.Geom import Geom_CylindricalSurface
    from OCP.Geom2d import Geom2d_Line
    from OCP.GeomAbs import GeomAbs_C2
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire, BRepBuilderAPI_Transform
    from OCP.BRepOffsetAPI import BRepOffsetAPI_MakePipeShell
    from OCP.BRepFill import BRepFill_NoContact
    from OCP.GeomAPI import GeomAPI_PointsToBSpline
    from OCP.collections import Array1_gp_Pnt
    from OCP.GC import GC_MakeArcOfCircle, GC_MakeSegment
    from OCP.BRepLib import BRepLib
    from OCP.TopoDS import TopoDS_Compound
    from OCP.BRep import BRep_Builder
    pitch = 2 * math.pi / N
    f = _fillet_geom(rb, psi_b, r_root, rho, N)
    if f is None:
        raise ValueError("root fillet does not fit")
    r_inv_top = r_tip + 0.1
    a_top = math.acos(min(rb / r_inv_top, 1.0))
    if psi_b - inv(a_top) <= 0:
        raise ValueError("pointed tooth below the tip")
    r_out = r_tip + 1.0
    P = lambda r, t: gp_Pnt(r * math.cos(t), r * math.sin(t), 0.0)
    rT = f["rT"]

    def flank(sgn, c):
        # involute part (exact points, B-spline < 1e-6 mm) from max(rT, rb); radial line below rb
        r0 = max(rT, rb)
        rs = np.linspace(r0, r_inv_top, 60)
        arr = Array1_gp_Pnt(1, len(rs))
        for i, r in enumerate(rs):
            a = math.acos(min(rb / r, 1.0)); th = c + sgn * (psi_b - inv(a))
            arr.SetValue(i + 1, P(r, th))
        return GeomAPI_PointsToBSpline(arr, 3, 8, GeomAbs_C2, 1e-6).Curve()
    th_top1 = psi_b - inv(a_top); th_top2 = pitch - th_top1
    th_T1 = psi_b - inv(math.acos(min(rb / rT, 1.0))) if rT > rb else psi_b
    th_T2 = pitch - th_T1
    t1, t2 = P(rT, th_T1), P(rT, th_T2)
    f1c, f2c = f["fc"], pitch - f["fc"]
    rc = f["rc"]
    g1, g2 = P(r_root, f1c), P(r_root, f2c)

    def fil(cang, a_from, a_to):
        cx, cy = rc * math.cos(cang), rc * math.sin(cang)
        A = math.atan2(a_from.Y() - cy, a_from.X() - cx); B = math.atan2(a_to.Y() - cy, a_to.X() - cx)
        m = A + 0.5 * math.remainder(B - A, 2 * math.pi)
        return GC_MakeArcOfCircle(a_from, gp_Pnt(cx + rho * math.cos(m), cy + rho * math.sin(m), 0), a_to).Value()
    E = [flank(+1, 0.0),
         GC_MakeSegment(P(r_inv_top, th_top1), P(r_out, th_top1)).Value(),
         GC_MakeArcOfCircle(P(r_out, th_top1), P(r_out, pitch / 2), P(r_out, th_top2)).Value(),
         GC_MakeSegment(P(r_out, th_top2), P(r_inv_top, th_top2)).Value(),
         flank(-1, pitch).Reversed()]
    if rT < rb - 1e-6:
        E.append(GC_MakeSegment(P(rb, pitch - psi_b), t2).Value())
    E += [fil(f2c, t2, g2), GC_MakeArcOfCircle(g2, P(r_root, pitch / 2), g1).Value(), fil(f1c, g1, t1)]
    if rT < rb - 1e-6:
        E.append(GC_MakeSegment(t1, P(rb, psi_b)).Value())
    mw = BRepBuilderAPI_MakeWire()
    for c in E:
        mw.Add(BRepBuilderAPI_MakeEdge(c).Edge())
    if not mw.IsDone():
        raise RuntimeError("space profile wire not closed")
    wire = mw.Wire()
    H = z_b - z_a
    spine = BRepBuilderAPI_MakeWire(BRepBuilderAPI_MakeEdge(gp_Pnt(0, 0, 0), gp_Pnt(0, 0, H)).Edge()).Wire()
    ps = BRepOffsetAPI_MakePipeShell(spine)
    if abs(w) > 1e-9:
        cs = Geom_CylindricalSurface(gp_Ax3(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), 10.0)
        he = BRepBuilderAPI_MakeEdge(Geom2d_Line(gp_Pnt2d(0, 0), gp_Dir2d(w, 1.0)), cs, 0.0, H * math.sqrt(1 + w * w)).Edge()
        BRepLib.BuildCurves3d_s(he)
        ps.SetMode(BRepBuilderAPI_MakeWire(he).Wire(), False, BRepFill_NoContact)
    else:
        ps.SetMode(gp_Dir(1, 0, 0))                       # fixed binormal: straight (spur) sweep
    ps.Add(wire, False, False)
    ps.Build()
    if not ps.IsDone():
        raise RuntimeError("helical sweep failed")
    ps.MakeSolid()
    sp = ps.Shape()
    e1, e2, uu = frame(u)
    bld = BRep_Builder(); comp = TopoDS_Compound(); bld.MakeCompound(comp)
    loc = gp_Trsf()          # local (e1, e2, u) frame at a0 -> world: columns e1, e2, u, translation a0
    loc.SetValues(*[float(v) for v in (e1[0], e2[0], uu[0], a0[0], e1[1], e2[1], uu[1], a0[1], e1[2], e2[2], uu[2], a0[2])])
    for k in range(N):
        T = gp_Trsf(); T.SetRotation(gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), clock + w * z_a + k * pitch)
        Z = gp_Trsf(); Z.SetTranslation(gp_Vec(0, 0, z_a))
        M = loc.Multiplied(Z.Multiplied(T))
        bld.Add(comp, BRepBuilderAPI_Transform(sp, M, True).Shape())
    return comp


# ----------------------------------------------------------------------------- end to end
def analyse(mesh, tol, n_samples=600000, seed=0, min_teeth=6):
    """Detect and fit an external involute gear on a scan mesh.  Returns (GearSpec | None, info dict).

    Accepted only when the involute explains the flanks: mid-flank rms <= max(0.1 m_n, 3 tol), at least
    70% inliers, and a whole depth of 1.8..3.2 m_n (rejects lobed couplings, splines of other forms,
    impeller blades)."""
    import trimesh
    info = {}
    P, fi = trimesh.sample.sample_surface(mesh, n_samples, seed=seed)
    P = np.asarray(P); Nn = np.asarray(mesh.face_normals)[fi]
    c = P.mean(0); _, V = np.linalg.eigh(np.cov((P - c).T))
    best = None
    # the part axis is the PCA axis about which the axial slices are round (a short gear can be wider
    # than long, so the longest axis is not always it)
    cands = []
    for u0 in V.T[::-1]:
        a0, u = refine_axis(P, c, u0)
        cands.append((_roundness(P, a0, u), a0, u))
    cands.sort(key=lambda t: -t[0])
    info["axis_roundness"] = [round(t[0], 3) for t in cands]
    for rnd, a0, u in cands[:1]:
        g = detect_gear(P, Nn, a0, u, min_teeth=min_teeth, tol=tol)
        if g is not None:
            best = (a0, u, g)
    if best is None:
        info["reason"] = "no N-fold periodic zone"
        return None, info
    a0, u, g = best
    a0, u = refine_axis(P, a0, u, exclude=(g.z0 - 3, g.z1 + 3))
    g = detect_gear(P, Nn, a0, u, min_teeth=min_teeth, tol=tol)
    if g is None:
        info["reason"] = "periodic zone lost after axis refinement"
        return None, info
    g.chamfers = refine_ends(P, g)
    if not g.chamfers:
        info["reason"] = "no tooth spaces found"
        return None, info
    r, thp, side, _ = _flank_points(P, Nn, g)
    if len(r) < 500:
        info["reason"] = "too few flank samples"
        return None, info
    f = fit_involute(r, thp, side, g.teeth, 0.94 * (g.r_tip + g.r_root) / 2)
    tc = tip_crest(P, g, f)
    lo, hi, r_form = active_band(g, f["rb"], r_tip=tc["r_tip_crest"] if tc else None)
    r, thp, side, _ = _flank_points(P, Nn, g, r_lo=lo, r_hi=hi)
    f = fit_involute(r, thp, side, g.teeth, f["rb"])
    f["mid_flank_band_r"] = [round(lo, 4), round(hi, 4)]; f["form_radius"] = round(r_form, 4)
    sn = snap(g, f, tol, tc)
    if not sn:
        info["reason"] = "no standard pitch/pressure angle near the fitted base circle"
        return None, info
    mn = sn["normal_module"]
    rms = max(f.get("flank_plus", {}).get("rms", 9), f.get("flank_minus", {}).get("rms", 9))
    depth = (g.r_tip - g.r_root) / mn
    ok = rms <= max(0.1 * mn, 3 * tol) and f["inliers"] >= 0.7 * f["points"] and 1.8 <= depth <= 3.2
    info.update({"flank_rms_max": rms, "whole_depth_over_module": round(depth, 3), "inlier_frac": round(f["inliers"] / f["points"], 3)})
    # plausible tooth proportions about the snapped pitch circle (rejects e.g. a 6-"tooth" m8 fit on a
    # faceted airfoil whose "root" lies near the axis)
    ded = (sn["pitch_diameter"] / 2 - g.r_root) / mn
    add = sn.get("addendum_ratio_measured")
    xs = sn.get("profile_shift_x_from_thickness")
    info.update(dedendum_over_module=round(ded, 3))
    implausible = [why for bad, why in ((not 0.8 <= ded <= 1.9, f"dedendum {ded:.2f} m"),
                                        (add is not None and add < 0.5, f"addendum {add} m"),
                                        (xs is not None and abs(xs) > 1.5, f"thickness implies x = {xs}"))
                   if bad]
    if ok and implausible:
        info["reason"] = "implausible tooth proportions: " + ", ".join(implausible)
        return None, info
    if not ok:
        info["reason"] = f"involute does not explain the flanks (rms {rms:.3f}, depth {depth:.2f} m, inliers {info['inlier_frac']})"
        return None, info
    free = {"rb": f["rb"], "psi_b": f["psi_b"], "r_tip": tc["r_tip_crest"] if tc else g.r_tip, "r_root": g.r_root, "w": g.w}
    for par in (free, sn["build"]):
        rf = fit_root_fillet(P, g, par["rb"], par["psi_b"], f["clocking_rad"])
        par["rho"] = rf["rho"] if rf else 0.25 * mn
        par["root_fillet_fit"] = rf
    g.fit = f; g.fit["tip"] = tc; g.fit["build"] = free
    g.snapped = sn
    alts = sn.get("alternatives") or []
    drb0 = (sn["base_diameter"] - sn["base_diameter_fit"]) / 2
    g.confidence = {
        "teeth": "high",
        "hand": "high" if g.hand != "spur" and abs(g.lead) < 1e5 else "n/a",
        "pitch_and_pressure_angle": ("high" if abs(drb0) <= 0.1 * mn and (not alts or abs(drb0) <= 0.35 * abs(alts[0]["drb_mm"]))
                                     else "medium"),
        "helix": "high" if (sn["lead_mm"] is None or abs(sn["lead_mm"] - sn["lead_measured_mm"]) / sn["lead_measured_mm"] < 0.01) else "medium",
        "tip_diameter": "medium (scan rounds tips; " + ("snapped to standard)" if sn["tip_snapped"] else "crest-centre measurement)"),
        "root_diameter": "high (snapped)" if sn["root_snapped"] else "medium (measured, non-standard dedendum)",
        "chamfers": "low (scan rounds edges)",
        "tooth_thickness": "medium (measured; profile shift vs backlash thinning is ambiguous from one gear)",
    }
    return g, info


def zones(P, g, band=None):
    """Sub-zones of scan points P in a gear zone: 'mid_flank' (0.2..0.8 of the active profile, away from
    the ends), 'tip_edge_rounding' (tip band, end chamfers / tooth end edges), 'root' (fillet and root
    circle).  Returns dict name -> bool mask, plus 'gear' (all)."""
    r, th, z = cyl(P, np.asarray(g.axis_point), np.asarray(g.axis_dir))
    za, zb = g.chamfers.get("z_end0", g.z0), g.chamfers.get("z_end1", g.z1)
    lo, hi = band or g.fit["mid_flank_band_r"]
    gear = (z >= za - 0.05) & (z <= zb + 0.05) & (r > g.r_root - 0.3)
    ends = gear & ((z < za + 1.5) | (z > zb - 1.5))
    mid = gear & ~ends & (r >= lo) & (r <= hi)
    tipz = gear & ((r > hi) | ends) & ~(r < lo)
    root = gear & ~mid & ~tipz
    return {"gear": gear, "mid_flank": mid, "tip_edge_rounding": tipz, "root": root}


def to_json(g):
    sn = dict(g.snapped); sn.pop("alternatives", None)
    alts = (g.snapped.get("alternatives") or [])[:4]
    return {"teeth": g.teeth, "hand": g.hand, "axis_point": [round(v, 4) for v in g.axis_point],
            "axis_dir": [round(v, 6) for v in g.axis_dir], "face_width_mm": round(g.chamfers["face_width"], 3),
            "z_ends_axis_frame": [round(g.chamfers["z_end0"], 3), round(g.chamfers["z_end1"], 3)],
            "end_chamfers": g.chamfers.get("chamfers"),
            "free_fit": {"lead_mm": round(g.lead, 3) if np.isfinite(g.lead) and abs(g.lead) < 1e5 else None, "base_diameter": round(2 * g.fit["rb"], 4),
                         "pressure_angle_normal_deg": sn.get("pressure_angle_fit_deg"),
                         "tip_diameter_crest": sn.get("tip_diameter_crest"), "root_diameter": round(2 * g.r_root, 4),
                         "tooth_thickness_normal_at_pitch": sn.get("tooth_thickness_normal_at_pitch"),
                         "root_fillet_radius": round(g.fit["build"]["rho"], 4),
                         "flank_residual": {k: g.fit.get(k) for k in ("flank_plus", "flank_minus")},
                         "mid_flank_band_r": g.fit.get("mid_flank_band_r"), "tip_crest": g.fit.get("tip")},
            "snapped": {k: v for k, v in sn.items() if k != "build"} | {"root_fillet_radius": round(sn["build"]["rho"], 4)},
            "alternatives": alts, "confidence": g.confidence}
