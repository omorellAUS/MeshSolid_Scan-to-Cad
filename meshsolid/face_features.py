"""Adaptive classification of features on an impeller's front (nose) and back (balance-plane) faces.

Deviations from the base face of revolution are segmented and classified:

(a) Balance cuts and drill marks: irregular or non-periodic about the axis.  Not modelled
    (the solid keeps the base face).  Named Metrology exclusion, greyed, with a geometric
    boundary dilated by 2 mm.

(b) Periodic functional features: N-fold about the axis with consistent depth and profile
    (Hirth tooth couplings, drive dogs, keyways).  Recognised and reported (tooth count,
    included angle, inner/outer radius, height).  Modelled when a builder exists and the
    parameters are complete; otherwise marked pending.

(c) Concentric features (bores, spigots, registers): reported and left to the hub of
    revolution (already modelled on a recognised impeller).

Uncertain features are not modelled; they are excluded and flagged like (a).

This replaces rim-slot cutting on impellers: a one-off back-face slot is irregular, so it
is (a) and is not cut.  A Hirth coupling is (b).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class FaceFeatureConfig:
    axial_cos: float = 0.85
    plane_band: float = 8.0          # mm around the end plane (Hirth teeth can be several mm high)
    end_plane_band: float = 0.4      # mm: points used to locate the base plane
    depth_min: float = 0.2           # mm from the base to count as a feature
    link: float = 1.5                # mm, neighbour distance when growing patches
    min_points: int = 25
    min_area: float = 4.0            # mm^2
    max_cut_span_deg: float = 80.0   # wider than this and not a ring -> uncertain
    nfold_min: int = 3
    nfold_max: int = 120
    period_snr: float = 4.0          # FFT peak / median in the N-band
    period_cv: float = 0.30          # max relative std of per-tooth depth
    period_spacing_cv: float = 0.12  # max relative std of the N angular gaps
    hirth_min_teeth: int = 6
    dog_max_teeth: int = 8
    margin_mm: float = 2.0           # exclusion boundary dilation (Metrology)
    auto_exclude_mm: float = 2.0     # leftover end-face depth above this is excluded
    bore_min_r: float = 1.0
    step_min: float = 0.4            # mm axial step for a register/spigot


# ---------------------------------------------------------------------------
# frame and helpers
# ---------------------------------------------------------------------------
def _basis(axis_dir, e1=None):
    u = np.asarray(axis_dir, float)
    u = u / np.linalg.norm(u)
    if e1 is None:
        e1 = np.cross(u, [1.0, 0, 0] if abs(u[0]) < 0.9 else [0, 1.0, 0])
    else:
        e1 = np.asarray(e1, float)
        e1 = e1 - (e1 @ u) * u
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(u, e1)
    return u, e1, e2


def to_frame(V, N, axis_point, axis_dir, e1=None):
    u, e1, e2 = _basis(axis_dir, e1)
    D = np.asarray(V, float) - np.asarray(axis_point, float)
    P = np.c_[D @ e1, D @ e2, D @ u]
    Nl = None if N is None else np.c_[np.asarray(N, float) @ e1, np.asarray(N, float) @ e2, np.asarray(N, float) @ u]
    return P, Nl, (u, e1, e2)


def _components(X, link):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    if len(X) == 0:
        return np.zeros(0, int), 0
    pairs = cKDTree(X).query_pairs(link, output_type="ndarray")
    n = len(X)
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n)) if len(pairs) else coo_matrix((n, n))
    k, lab = connected_components(g, directed=False)
    return lab, k


def _area(xy, cell=0.7):
    g = np.unique(np.floor(np.asarray(xy, float) / cell).astype(int), axis=0)
    return float(len(g) * cell * cell)


def _in_arc(th, t0, t1):
    """True where angle th (rad) lies on the CCW arc t0 -> t1 (wraps).  A full circle is all True."""
    d = (t1 - t0) % (2 * math.pi)
    if d < 1e-12 or d > 2 * math.pi - 1e-12:
        return np.ones(np.shape(th), bool)
    return ((np.asarray(th) - t0) % (2 * math.pi)) <= d + 1e-12


def end_planes(P, N, cfg=FaceFeatureConfig()):
    """The two axial end faces: (z0, s, name_hint) at the low end (material above) and the high end."""
    z = P[:, 2]
    ax = np.abs(N[:, 2]) > cfg.axial_cos
    out = []
    lo = ax & (N[:, 2] < 0) & (z < np.percentile(z, 2) + cfg.end_plane_band)
    hi = ax & (N[:, 2] > 0) & (z > np.percentile(z, 98) - cfg.end_plane_band)
    if lo.sum() > 20:
        out.append((float(np.median(z[lo])), +1, "low"))
    if hi.sum() > 20:
        out.append((float(np.median(z[hi])), -1, "high"))
    return out


def _name_faces(planes):
    """Label the plane nearer z = 0 as 'back' (impeller convention: back plane at z = 0)."""
    if not planes:
        return []
    named = []
    # Impeller frames put the back plane at z = 0.  Only use back/front when one plane
    # is actually at z ≈ 0; otherwise keep low/high so a mid-part origin is not guessed.
    at_zero = [abs(z0) <= 5.0 for z0, s, h in planes]
    if len(planes) == 1:
        z0, s, hint = planes[0]
        name = "back" if at_zero[0] else hint
        named.append((z0, s, name))
        return named
    if any(at_zero):
        back_i = int(np.argmin([abs(z0) for z0, s, h in planes]))
        for i, (z0, s, h) in enumerate(planes):
            named.append((z0, s, "back" if i == back_i else "front"))
        return named
    for z0, s, h in planes:
        named.append((z0, s, h))
    return named


def _ring_z(r, z, step=1.0, min_n=8):
    """Median z in 1 mm radial rings; gaps linearly interpolated, leading/trailing filled."""
    b = np.floor(r / step).astype(int)
    keys = np.unique(b)
    med = {}
    for k in keys:
        m = b == k
        if m.sum() >= min_n:
            med[int(k)] = float(np.median(z[m]))
    if not med:
        return lambda rr: np.full(np.shape(rr), float(np.median(z)) if len(z) else 0.0)
    ks = np.array(sorted(med), int)
    vs = np.array([med[int(k)] for k in ks], float)
    def ev(rr):
        x = np.asarray(rr, float) / step
        return np.interp(x, ks, vs, left=vs[0], right=vs[-1])
    return ev


def _periodogram_n(theta, weights, nmin, nmax, snr):
    """Dominant integer fold N from a weighted angle histogram.  Returns (N, peak/median) or (None, 0)."""
    if len(theta) < 20:
        return None, 0.0
    nb = 720
    w = None if weights is None else np.asarray(weights, float)
    h, _ = np.histogram(np.asarray(theta, float) % (2 * math.pi), bins=nb,
                        weights=w, range=(0.0, 2 * math.pi))
    h = h.astype(float)
    h -= h.mean()
    mag = np.abs(np.fft.rfft(h))
    hi = min(nmax, len(mag) - 1)
    if hi < nmin:
        return None, 0.0
    band = mag[nmin:hi + 1].copy()
    band0 = mag.copy(); band0[0] = 0
    i_rel = int(np.argmax(band))
    n = nmin + i_rel
    peak = float(band[i_rel])
    med = float(np.median(band)) if band.size else 0.0
    ratio = peak / med if med > 1e-12 else (peak if peak > 0 else 0.0)
    # prefer the fundamental: if N has a strong divisor N/k that is also a peak, take the smallest
    # with at least 0.6 of this peak
    if ratio < snr:
        return None, ratio
    fund = n
    for k in range(2, 7):
        if n % k:
            continue
        n2 = n // k
        if n2 < nmin:
            continue
        p2 = float(mag[n2]) if n2 < len(mag) else 0.0
        if p2 >= 0.6 * peak:
            fund = n2
    return int(fund), float(mag[fund] / med if med > 1e-12 else ratio)


def _folded_profile(theta, res, N, nb=72):
    """Mean residual vs folded angle in [0, 2pi)."""
    phi = (np.asarray(theta) * N) % (2 * math.pi)
    bins = np.linspace(0.0, 2 * math.pi, nb + 1)
    i = np.clip(np.digitize(phi, bins) - 1, 0, nb - 1)
    mu = np.full(nb, np.nan)
    for k in range(nb):
        m = i == k
        if m.sum() >= 3:
            mu[k] = float(np.median(res[m]))
    return 0.5 * (bins[:-1] + bins[1:]), mu


def _hirth_params(theta, r, res, N):
    """Included angle, height, r-band from a folded triangular profile.  None if not triangular."""
    mid = (np.percentile(r, 25), np.percentile(r, 75))
    sel = (r >= mid[0]) & (r <= mid[1])
    if sel.sum() < 30:
        sel = np.ones(len(r), bool)
    phi, mu = _folded_profile(theta[sel], res[sel], N)
    ok = np.isfinite(mu)
    if ok.sum() < 8:
        return None
    # height = peak-to-valley of the folded residual (into-material is positive)
    height = float(np.nanmax(mu) - np.nanmin(mu))
    if height < 0.15:
        return None
    # valley at the max (into the material); crest at the min
    i_v = int(np.nanargmax(mu)); i_c = int(np.nanargmin(mu))
    # walk both ways from the valley until mid-height, fit a slope on each flank
    midh = 0.5 * (np.nanmax(mu) + np.nanmin(mu))
    def flank(step):
        xs, ys = [], []
        j = i_v
        for _ in range(len(mu)):
            if not np.isfinite(mu[j]):
                j = (j + step) % len(mu)
                continue
            # unwrapped arc from valley, in bins
            xs.append((j - i_v) if step > 0 else (i_v - j))
            ys.append(mu[j])
            if (step > 0 and (j - i_v) % len(mu) > 1 and mu[j] <= midh) or (
                    step < 0 and (i_v - j) % len(mu) > 1 and mu[j] <= midh):
                break
            j = (j + step) % len(mu)
        if len(xs) < 3:
            return None
        return np.polyfit(xs, ys, 1)[0]
    s1, s2 = flank(+1), flank(-1)
    if s1 is None or s2 is None:
        return None
    # folded-angle bin width; physical arc at median r is r * dphi / N
    dphi = 2 * math.pi / len(mu)
    rm = float(np.median(r[sel]))
    ds = max(rm * dphi / max(N, 1), 1e-9)

    def angle_from_slope(sl):
        dzds = sl / ds
        return math.atan(abs(1.0 / dzds)) if abs(dzds) > 1e-9 else math.pi / 2
    # included angle at the valley: 2 * atan(ds/dz) if symmetric, else sum
    a1, a2 = angle_from_slope(s1), angle_from_slope(s2)
    included = math.degrees(a1 + a2)
    if not (20.0 <= included <= 160.0):
        return None
    # triangular if each flank is reasonably linear and the crest-valley split is not a flat floor
    # a rectangular dog has a long flat at the max; reject if > 30 % of bins sit near the peak
    near_peak = ok & (mu >= np.nanmax(mu) - 0.15 * height)
    if near_peak.sum() > 0.30 * ok.sum():
        return None
    return {"included_angle_deg": round(float(included), 2), "height_mm": round(height, 3),
            "r_inner": round(float(np.percentile(r, 1)), 3), "r_outer": round(float(np.percentile(r, 99)), 3)}


def _per_tooth_depth(theta, res, N):
    """Median residual in each of the N folded sectors (the tooth / pocket depths)."""
    sec = (np.floor((np.asarray(theta) % (2 * math.pi)) * N / (2 * math.pi)) % N).astype(int)
    d = []
    for k in range(N):
        m = sec == k
        d.append(float(np.median(res[m])) if m.sum() >= 5 else float("nan"))
    return np.asarray(d, float)


def _spacing_cv(theta, N):
    """Relative std of the N sector-centre gaps.  0 = equally spaced; nan if too few sectors."""
    theta = np.asarray(theta, float)
    sec = (np.floor((theta % (2 * math.pi)) * N / (2 * math.pi)) % N).astype(int)
    cents = []
    for k in range(N):
        m = sec == k
        if m.sum() >= 5:
            cents.append(float(np.angle(np.mean(np.exp(1j * theta[m])))))
    if len(cents) < max(3, int(0.8 * N)):
        return float("nan")
    ang = np.sort(np.asarray(cents))
    gaps = np.diff(np.r_[ang, ang[0] + 2 * math.pi])
    mu = float(gaps.mean())
    return float(gaps.std() / mu) if mu > 1e-9 else float("nan")


def _sector_bounds(th, r, z, margin_mm, axis_point, u, e1, t_pad=None):
    """Tight angular/radial/axial bounds plus a sector-zone dict with margin_mm."""
    th = np.asarray(th, float)
    thc = float(np.angle(np.mean(np.exp(1j * th))))
    dth = np.angle(np.exp(1j * (th - thc)))
    t0 = thc + float(np.percentile(dth, 0.5))
    t1 = thc + float(np.percentile(dth, 99.5))
    r0 = float(np.percentile(r, 0.5)); r1 = float(np.percentile(r, 99.5))
    z0 = float(np.percentile(z, 0.5)); z1 = float(np.percentile(z, 99.5))
    if t_pad:
        z0 -= t_pad; z1 += t_pad
    zone = {"type": "sector",
            "point": [float(x) for x in np.asarray(axis_point, float)],
            "dir": [float(x) for x in np.asarray(u, float)],
            "e1": [float(x) for x in np.asarray(e1, float)],
            "r": [round(r0, 3), round(r1, 3)],
            "theta_deg": [round(math.degrees(t0) % 360.0, 3),
                          round(math.degrees(t1) % 360.0, 3) if (t1 - t0) % (2 * math.pi) > 1e-9
                          else round(math.degrees(t0) % 360.0 + 360.0, 3)],
            "t": [round(z0, 3), round(z1, 3)],
            "margin_mm": float(margin_mm)}
    # if the arc wrapped past 360 in the listed numbers, keep them as a CCW pair even if t1 < t0
    # (zone_mask uses CCW t0 -> t1).  When t1 was wrapped with % 360 it can be < t0; that's correct.
    if (t1 - t0) >= 2 * math.pi - 1e-6:
        zone["theta_deg"] = [0.0, 360.0]
    return dict(centre_deg=round(math.degrees(thc) % 360.0, 3),
                span_deg=round(math.degrees(t1 - t0), 3),
                r_inner=round(r0, 3), r_outer=round(r1, 3),
                z_min=round(z0, 3), z_max=round(z1, 3),
                boundary=zone)


def _feature(cls, kind, face, label, reason, n_points, depth, area, bounds, extra=None,
             exclude=None, modelled=False, model_pending=False):
    if exclude is None:
        exclude = cls in ("a", "uncertain")
    out = {"class": cls, "kind": kind, "face": face, "label": label, "reason": reason,
           "n_points": int(n_points), "depth_mm": round(float(depth), 3),
           "area_mm2": round(float(area), 1), "exclude": bool(exclude),
           "modelled": bool(modelled), "model_pending": bool(model_pending)}
    out.update(bounds)
    if extra:
        out.update(extra)
    return out


# ---------------------------------------------------------------------------
# classification of one end face
# ---------------------------------------------------------------------------
def classify_face(P, N, z0, s, face, cfg, axis_point, u, e1, R=None):
    """Classify deviations from the base plane z = z0 (material on side s) of one end face.

    P, N are already in the axis frame.  Returns (features, vertex-mask of points to exclude).
    """
    P = np.asarray(P, float); N = np.asarray(N, float)
    r = np.hypot(P[:, 0], P[:, 1]); z = P[:, 2]; th = np.arctan2(P[:, 1], P[:, 0])
    if R is None or R <= 0:
        R = float(np.percentile(r, 99.5)) if len(r) else 1.0
    ax = np.abs(N[:, 2]) > cfg.axial_cos
    band = ax & (np.abs(z - z0) < cfg.plane_band) & (r <= R * 1.05)
    # also keep mildly tilted floor/flank points inside the band (Hirth flanks, pocket walls)
    near = (np.abs(z - z0) < cfg.plane_band) & (r <= R * 1.05) & (s * (z - z0) > -0.5)
    face_m = band | (near & (np.abs(N[:, 2]) > 0.35))
    mask = np.zeros(len(P), bool)
    features = []
    if face_m.sum() < cfg.min_points:
        return features, mask

    # base of revolution: median z of near-plane axial points per radial ring (ignores deep pockets)
    base_pts = ax & (np.abs(z - z0) < max(cfg.end_plane_band * 3, 1.0)) & (r <= R * 1.02)
    if base_pts.sum() < cfg.min_points:
        base_pts = band
    z_of = _ring_z(r[base_pts], z[base_pts])
    # depth into the material from the base of revolution (positive = into material)
    dep = s * (z - z_of(r))

    # ---- (c) concentric: bore and axial steps that go all the way around ------------------------
    # bore: the face is empty about the axis out to a sharp inner lip (area-normalised density
    # jumps from ~0 to the face density).  A full disc has points down to r ≈ 0 and is not a bore.
    face_ax = band
    r_face = r[face_ax]
    r_w = None
    wall = (np.abs(z - z0) < cfg.plane_band + 2.0) & (r > cfg.bore_min_r) & (np.hypot(N[:, 0], N[:, 1]) > 0.85)
    if len(r_face) >= cfg.min_points:
        edges = np.arange(0.0, max(float(R) + 1.5, 2.0), 1.0)
        if len(edges) >= 3:
            cnt, _ = np.histogram(r_face, bins=edges)
            area = math.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
            dens = cnt / np.maximum(area, 1e-9)
            occupied = dens > 0
            typ = float(np.median(dens[cnt >= 8])) if (cnt >= 8).any() else 0.0
            # first radius whose density is a substantial fraction of the typical face density
            lip = np.where((dens >= 0.3 * typ) & (cnt >= 8))[0] if typ > 0 else np.array([], int)
            if len(lip):
                i0 = int(lip[0])
                r_lip = float(edges[i0])
                expected = typ * math.pi * r_lip * r_lip
                actual = float((r_face < r_lip - 0.5).sum())
                if r_lip >= cfg.bore_min_r and expected >= 40 and actual <= 0.15 * expected:
                    r_w = r_lip
        # prefer a cylindrical wall radius when one is present at the lip
        if wall.sum() >= 20:
            rw = r[wall]
            if r_w is None:
                # wall-only bore (few axial face points at the lip)
                # a tight radial cluster of wall points spanning the circle
                r_med = float(np.median(rw))
                if r_med >= cfg.bore_min_r and (np.abs(rw - r_med) < max(1.5, 0.1 * r_med)).mean() > 0.5:
                    cov = len(np.unique(np.round((th[wall] % (2 * math.pi)) * 8 / (2 * math.pi))))
                    if cov >= 6:
                        r_w = r_med
            elif np.abs(rw - r_w).min() < max(1.5, 0.1 * r_w):
                r_w = float(np.median(rw[np.abs(rw - r_w) < max(1.5, 0.1 * r_w)]))
        if r_w is not None and r_w >= cfg.bore_min_r:
            thw = th[wall] if wall.sum() >= 20 else th[face_ax & (np.abs(r - r_w) < 2.0)]
            cov = len(np.unique(np.round((thw % (2 * math.pi)) * 8 / (2 * math.pi)))) if len(thw) else 0
            if len(thw) >= 20 and cov >= 6:
                sel = face_ax & (r <= r_w + 1.0)
                bds = _sector_bounds(th[sel] if sel.sum() else thw, r[sel] if sel.sum() else r[wall],
                                     z[sel] if sel.sum() else z[wall], cfg.margin_mm, axis_point, u, e1)
                # a bore is a full circle
                bds["boundary"]["theta_deg"] = [0.0, 360.0]
                bds["span_deg"] = 360.0
                features.append(_feature(
                    "c", "bore", face,
                    f"bore Ø{2 * r_w:.3f} mm on the {face} face",
                    "concentric circular opening about the axis; modelled with the hub of revolution",
                    int(sel.sum() or wall.sum()), 0.0, math.pi * r_w * r_w, bds,
                    extra={"d_mm": round(2 * r_w, 3)}, exclude=False, modelled=True))

    # registers / spigots: a 360° axial step in z(r)
    if face_ax.sum() >= 80:
        step = 1.0
        rb = np.floor(r[face_ax] / step).astype(int)
        rings = []
        for k in np.unique(rb):
            m = rb == k
            if m.sum() < 12:
                continue
            zz = z[face_ax][m]; tt = th[face_ax][m]
            # full-circle coverage, and z must be axisymmetric (a toothed ring is not a register)
            cov = len(np.unique(np.round((tt % (2 * math.pi)) * 12 / (2 * math.pi))))
            if cov < 8 or float(np.ptp(zz)) > max(cfg.step_min, 2.5 * float(np.median(np.abs(zz - np.median(zz))) or 0.1)):
                continue
            if float(np.percentile(zz, 90) - np.percentile(zz, 10)) > cfg.step_min:
                continue
            rings.append((k * step, float(np.median(zz)), cov))
        for i in range(1, len(rings)):
            dz = rings[i][1] - rings[i - 1][1]
            if abs(dz) < cfg.step_min:
                continue
            r_step = 0.5 * (rings[i][0] + rings[i - 1][0])
            # skip if this is just the bore lip
            if any(f["kind"] == "bore" and abs(0.5 * f.get("d_mm", 0) - r_step) < 2.0 for f in features):
                continue
            kind = "spigot" if s * dz < 0 else "register"
            # points in a 3 mm radial band about the step
            sel = face_ax & (np.abs(r - r_step) < 3.0)
            bds = _sector_bounds(th[sel], r[sel], z[sel], cfg.margin_mm, axis_point, u, e1)
            bds["boundary"]["theta_deg"] = [0.0, 360.0]
            bds["span_deg"] = 360.0
            features.append(_feature(
                "c", kind, face,
                f"{kind} Ø{2 * r_step:.2f} mm, {abs(dz):.2f} mm step on the {face} face",
                "concentric axial step about the axis; modelled with the hub of revolution",
                int(sel.sum()), abs(dz), _area(P[sel, :2]) if sel.sum() else 0.0, bds,
                extra={"d_mm": round(float(2 * r_step), 3), "step_mm": round(float(abs(dz)), 3)},
                exclude=False, modelled=True))

    # residual points that belong to a modelled concentric feature stay out of (a)/(b)
    taken = np.zeros(len(P), bool)
    from .metrology_exclude import zone_mask as _zmask
    # zone_mask expects world points; convert local (x,y,z) -> world via axis_point, e1, e2, u
    e2 = np.cross(u, e1)
    def _world(idx=None):
        Q = P if idx is None else P[idx]
        return np.asarray(axis_point, float) + Q[:, [0]] * e1 + Q[:, [1]] * e2 + Q[:, [2]] * u

    for f in features:
        if f.get("boundary"):
            taken |= _zmask(_world(), {**f["boundary"], "margin_mm": 0.5})

    # ---- (b) N-fold periodicity in the residual -------------------------------------------------
    # Use only axial-normal points so blade sides (tangential) cannot register as an end-face coupling.
    feat_m = band & ~taken & (np.abs(dep) >= cfg.depth_min)
    Nfold, snr = _periodogram_n(th[feat_m], np.abs(dep[feat_m]), cfg.nfold_min, cfg.nfold_max, cfg.period_snr)
    if Nfold is not None and feat_m.sum() >= cfg.min_points:
        depths = _per_tooth_depth(th[feat_m], np.abs(dep[feat_m]), Nfold)
        ok_d = depths[np.isfinite(depths)]
        cv = float(ok_d.std() / ok_d.mean()) if len(ok_d) >= max(3, Nfold // 2) and ok_d.mean() > 1e-6 else 1.0
        sp = _spacing_cv(th[feat_m], Nfold)
        r_f, th_f, dep_f = r[feat_m], th[feat_m], dep[feat_m]
        r_span = (float(np.percentile(r_f, 99)) - float(np.percentile(r_f, 1))) if len(r_f) else 0.0
        # a coupling lives in an annulus; a signal from r ≈ 0 to the rim is the blading, not an end-face feature
        blade_like = (float(np.percentile(r_f, 1)) < 0.25 * R) and (float(np.percentile(r_f, 99)) > 0.70 * R)
        consistent = (cv <= cfg.period_cv and len(ok_d) >= max(3, int(0.8 * Nfold))
                      and np.isfinite(sp) and sp <= cfg.period_spacing_cv
                      and not blade_like and r_span > 2.0)
        if consistent:
            # typify with slightly tilted flank points as well (Hirth flanks are not axial)
            r0b, r1b = float(np.percentile(r_f, 1)), float(np.percentile(r_f, 99))
            typ_m = (np.abs(N[:, 2]) > 0.35) & ~taken & (np.abs(dep) >= 0.5 * cfg.depth_min)                 & (np.abs(z - z0) < cfg.plane_band) & (r >= r0b - 1.0) & (r <= r1b + 1.0)
            if typ_m.sum() < max(feat_m.sum(), cfg.min_points):
                typ_m = feat_m
            hirth = _hirth_params(th[typ_m], r[typ_m], dep[typ_m], Nfold) if Nfold >= cfg.hirth_min_teeth else None
            if hirth:
                kind, model_pending = "hirth", True
                label = (f"Hirth coupling, {Nfold} teeth, included angle {hirth['included_angle_deg']:.1f}°, "
                         f"r {hirth['r_inner']:.2f}–{hirth['r_outer']:.2f} mm, height {hirth['height_mm']:.3f} mm "
                         f"on the {face} face")
                reason = ("N-fold radial triangular teeth, consistent depth and profile; "
                          "recognised as a Hirth coupling")
                extra = {"N": int(Nfold), "period_snr": round(snr, 2), **hirth}
            elif Nfold <= cfg.dog_max_teeth:
                kind, model_pending = "drive_dog", True
                r0, r1 = float(np.percentile(r_f, 1)), float(np.percentile(r_f, 99))
                ht = float(np.percentile(np.abs(dep_f), 90))
                label = (f"{Nfold}-fold drive dogs, r {r0:.2f}–{r1:.2f} mm, depth {ht:.3f} mm on the {face} face")
                reason = "N-fold repeated end-face dogs with consistent depth; recognised, modelling pending"
                extra = {"N": int(Nfold), "period_snr": round(snr, 2),
                         "r_inner": round(r0, 3), "r_outer": round(r1, 3), "height_mm": round(ht, 3)}
            else:
                kind, model_pending = "periodic", True
                r0, r1 = float(np.percentile(r_f, 1)), float(np.percentile(r_f, 99))
                ht = float(np.percentile(np.abs(dep_f), 90))
                label = f"{Nfold}-fold periodic feature on the {face} face (r {r0:.2f}–{r1:.2f} mm)"
                reason = "N-fold repeated about the axis with consistent depth; recognised, modelling pending"
                extra = {"N": int(Nfold), "period_snr": round(snr, 2),
                         "r_inner": round(r0, 3), "r_outer": round(r1, 3), "height_mm": round(ht, 3)}
            bds = _sector_bounds(th_f, r_f, z[feat_m], cfg.margin_mm, axis_point, u, e1)
            # the family occupies the whole circle in its radial band
            bds["boundary"]["theta_deg"] = [0.0, 360.0]
            bds["span_deg"] = 360.0
            features.append(_feature(
                "b", kind, face, label, reason, int(feat_m.sum()),
                float(np.percentile(np.abs(dep_f), 90)), _area(P[feat_m, :2]), bds,
                extra=extra, exclude=False, modelled=False, model_pending=model_pending))
            taken |= feat_m
        # inconsistent N-fold (uneven spacing or depth, e.g. a handful of balance cuts):
        # do not group them; they fall through to per-patch class (a).

    # ---- (a) irregular / non-periodic patches, plus leftover > auto_exclude_mm -------------------
    rest = face_m & ~taken & (np.abs(dep) >= cfg.depth_min)
    idx = np.where(rest)[0]
    if len(idx) >= cfg.min_points:
        lab, nc = _components(P[idx], cfg.link)
        for c in range(nc):
            q = idx[lab == c]
            if len(q) < cfg.min_points:
                continue
            a = _area(P[q, :2])
            if a < cfg.min_area:
                continue
            dpth = float(np.percentile(np.abs(dep[q]), 90))
            bds = _sector_bounds(th[q], r[q], z[q], cfg.margin_mm, axis_point, u, e1)
            span = bds["span_deg"]
            if span >= 300:
                # leftover ring not already called a register — uncertain
                cls, kind = "uncertain", "ring"
                label = f"unclassified ring on the {face} face, depth {dpth:.3f} mm"
                reason = "a near-full-circle band that is not a clean concentric step; excluded and flagged, not modelled"
            elif span > cfg.max_cut_span_deg:
                cls, kind = "uncertain", "wide_depression"
                label = f"wide depression on the {face} face at {bds['centre_deg']:.1f}°, span {span:.1f}°"
                reason = "too wide to be a compact balance cut and not periodic; excluded and flagged, not modelled"
            else:
                cls, kind = "a", "balance_cut" if dpth >= 0.15 else "drill_mark"
                if dpth < 0.4 and a < 15.0:
                    kind = "drill_mark"
                label = (f"{kind.replace('_', ' ')} on the {face} face at {bds['centre_deg']:.1f}°, "
                         f"r {bds['r_inner']:.2f}–{bds['r_outer']:.2f} mm, depth {dpth:.3f} mm, "
                         f"span {span:.1f}°")
                reason = ("irregular or non-periodic about the axis; not design geometry — "
                          "the solid keeps the base face of revolution; named Metrology exclusion "
                          f"with a {cfg.margin_mm:g} mm boundary")
            features.append(_feature(cls, kind, face, label, reason, len(q), dpth, a, bds,
                                     exclude=True, modelled=False))
            mask[q] = True
            taken[q] = True

    # safety net: any remaining end-face points more than auto_exclude_mm into the material
    leftover = face_m & ~taken & (dep >= cfg.auto_exclude_mm)
    if leftover.sum() >= cfg.min_points:
        lab, nc = _components(P[leftover], cfg.link)
        idx = np.where(leftover)[0]
        for c in range(nc):
            q = idx[lab == c]
            if len(q) < cfg.min_points:
                continue
            dpth = float(np.percentile(dep[q], 90))
            bds = _sector_bounds(th[q], r[q], z[q], cfg.margin_mm, axis_point, u, e1)
            features.append(_feature(
                "uncertain", "deep_unclassified", face,
                f"unclassified {dpth:.2f} mm depression on the {face} face at {bds['centre_deg']:.1f}° "
                f"(auto-excluded at {cfg.auto_exclude_mm:g} mm)",
                f"deeper than the {cfg.auto_exclude_mm:g} mm auto-exclusion threshold and not recognised "
                f"as a periodic or concentric feature; excluded with a {cfg.margin_mm:g} mm boundary, not modelled",
                len(q), dpth, _area(P[q, :2]), bds, exclude=True, modelled=False))
            mask[q] = True

    # dilate each excluded feature's mask by margin_mm around its points (and the zone does the same)
    if mask.any():
        near = cKDTree(P[mask, :2]).query(P[:, :2], distance_upper_bound=cfg.margin_mm)[0] < np.inf
        in_band = np.abs(z - z0) < cfg.plane_band + cfg.margin_mm
        mask = (mask | near) & in_band & ~taken | mask
        # don't paint over modelled concentric / periodic points
        # (taken includes excluded already; strip modelled)
        modelled_pts = np.zeros(len(P), bool)
        for f in features:
            if not f.get("exclude") and f.get("boundary"):
                modelled_pts |= _zmask(_world(), {**f["boundary"], "margin_mm": 0.0})
        mask &= ~modelled_pts
    return features, mask


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def classify_points(P, N, axis_point, axis_dir, e1=None, R=None, cfg=FaceFeatureConfig(), planes=None):
    """Classify front and back end-face features.  P, N may be in world coordinates.

    Returns (report, exclude_vertex_mask) in the same vertex order as P.
    """
    cfg = cfg if isinstance(cfg, FaceFeatureConfig) else FaceFeatureConfig(**(cfg or {}))
    Pl, Nl, (u, e1, e2) = to_frame(P, N, axis_point, axis_dir, e1)
    r = np.hypot(Pl[:, 0], Pl[:, 1])
    if R is None or R <= 0:
        R = float(np.percentile(r, 99.5)) if len(r) else 1.0
    planes = _name_faces(planes if planes is not None else end_planes(Pl, Nl, cfg))
    features, mask = [], np.zeros(len(P), bool)
    faces_out = []
    for z0, s, name in planes:
        feats, m = classify_face(Pl, Nl, z0, s, name, cfg, axis_point, u, e1, R)
        features.extend(feats)
        mask |= m
        faces_out.append({"name": name, "z0": round(float(z0), 3), "material_side": int(s),
                          "n_features": len(feats)})
    exclusions = [f for f in features if f.get("exclude")]
    n_a = sum(1 for f in features if f["class"] == "a")
    n_b = sum(1 for f in features if f["class"] == "b")
    n_c = sum(1 for f in features if f["class"] == "c")
    n_u = sum(1 for f in features if f["class"] == "uncertain")
    hirth = next((f for f in features if f.get("kind") == "hirth"), None)
    report = {
        "faces": faces_out,
        "features": features,
        "exclusions": exclusions,
        "summary": {
            "n_a_balance_or_drill": n_a,
            "n_b_periodic": n_b,
            "n_c_concentric": n_c,
            "n_uncertain": n_u,
            "n_excluded": len(exclusions),
            "hirth_N": None if hirth is None else hirth.get("N"),
            "margin_mm": cfg.margin_mm,
            "auto_exclude_mm": cfg.auto_exclude_mm,
        },
        "config": asdict(cfg),
        "axis_point": [float(x) for x in np.asarray(axis_point, float)],
        "axis_dir": [float(x) for x in u],
        "e1": [float(x) for x in e1],
        "R": round(float(R), 3),
        "basis": ("deviations from the base face of revolution on the front and back end faces; "
                  "(a) irregular = exclude + 2 mm boundary, (b) N-fold functional = recognise, "
                  "(c) concentric = model with the hub, uncertain = exclude and flag"),
    }
    return report, mask


def classify_mesh(mesh, axis_point, axis_dir, e1=None, R=None, cfg=FaceFeatureConfig(), planes=None):
    V = np.asarray(mesh.vertices, float)
    N = np.asarray(mesh.vertex_normals, float)
    return classify_points(V, N, axis_point, axis_dir, e1, R, cfg, planes)


def exclusion_vertex_mask(P, report):
    """Rebuild the exclusion mask from the report's sector boundaries (world-frame points P)."""
    from .metrology_exclude import zone_mask
    P = np.asarray(P, float)
    m = np.zeros(len(P), bool)
    for f in report.get("exclusions") or []:
        z = f.get("boundary")
        if z:
            m |= zone_mask(P, z)
    return m


def exclusion_face_mask(mesh, report):
    v = exclusion_vertex_mask(np.asarray(mesh.vertices), report)
    return v[np.asarray(mesh.faces)].any(1)


# ---------------------------------------------------------------------------
# Hirth builder (cut N triangular radial grooves into an end face)
# ---------------------------------------------------------------------------
def hirth_params_complete(feat):
    return (feat.get("kind") == "hirth" and feat.get("N") and feat.get("included_angle_deg")
            and feat.get("r_inner") is not None and feat.get("r_outer") is not None
            and feat.get("height_mm") and feat.get("z_min") is not None)


def cut_hirth(shape, axis_point, axis_dir, e1, feat, over=0.3):
    """Cut a recognised Hirth coupling into `shape`.  The crest is the air-side of the feature
    (z_min or z_max depending on material_side, recovered from the bounds)."""
    from OCP.BRepBuilderAPI import (BRepBuilderAPI_MakePolygon, BRepBuilderAPI_MakeFace,
                                    BRepBuilderAPI_Transform)
    from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCP.gp import gp_Pnt, gp_Vec, gp_Trsf, gp_Ax1, gp_Ax3, gp_Dir
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.TopAbs import TopAbs_SOLID
    from . import occ_util as U

    N = int(feat["N"])
    alpha = float(feat["included_angle_deg"])
    r0 = float(feat["r_inner"]); r1 = float(feat["r_outer"])
    h = float(feat["height_mm"])
    s = int(feat.get("material_side") or 1)
    # crest is the air side of [z_min, z_max]
    z_lo, z_hi = float(feat["z_min"]), float(feat["z_max"])
    z_crest = z_lo if s > 0 else z_hi
    half = h * math.tan(math.radians(alpha) / 2.0)
    if half < 1e-4 or r1 <= r0 or h < 1e-4:
        raise ValueError("Hirth parameters do not describe a cuttable tooth")
    u, e1, e2 = _basis(axis_dir, e1)
    a0 = np.asarray(axis_point, float)
    place = gp_Trsf()
    place.SetTransformation(gp_Ax3(gp_Pnt(*map(float, a0)), gp_Dir(*map(float, u)), gp_Dir(*map(float, e1))))
    place.Invert()
    # triangle in the yz plane, prism along +x from r0-over to r1+over
    pad = 0.15 * half
    z_v = z_crest + s * (h + over)
    z_out = z_crest - s * over
    poly = BRepBuilderAPI_MakePolygon()
    poly.Add(gp_Pnt(r0 - over, 0.0, z_v))
    poly.Add(gp_Pnt(r0 - over, +(half + pad), z_out))
    poly.Add(gp_Pnt(r0 - over, -(half + pad), z_out))
    poly.Close()
    face = BRepBuilderAPI_MakeFace(poly.Wire()).Face()
    prism = BRepPrimAPI_MakePrism(face, gp_Vec(r1 - r0 + 2 * over, 0, 0)).Shape()
    out = shape
    for i in range(N):
        rot = gp_Trsf()
        rot.SetRotation(gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), i * 2 * math.pi / N)
        cutter = BRepBuilderAPI_Transform(prism, rot, True).Shape()
        cutter = BRepBuilderAPI_Transform(cutter, place, True).Shape()
        out = BRepAlgoAPI_Cut(out, cutter).Shape()
    if not (BRepCheck_Analyzer(out).IsValid() and U.count(out, TopAbs_SOLID) == 1
            and U.volume(out) < U.volume(shape)):
        raise RuntimeError("Hirth cut did not produce one valid solid with material removed")
    return out


def apply_modelled(shape, report, axis_point, axis_dir, e1, log=print):
    """Apply builders for recognised (b) features whose parameters are complete.  Currently Hirth.

    On failure the feature is left as model_pending and the input shape is kept for that cut.
    Returns (shape, report).
    """
    feats = report.get("features") or []
    for f in feats:
        if f.get("kind") != "hirth" or f.get("modelled") or not f.get("model_pending"):
            continue
        if not hirth_params_complete(f):
            f["model_pending"] = True
            f["not_modelled_reason"] = "Hirth parameters incomplete"
            continue
        # material_side from the parent face
        if "material_side" not in f:
            face = next((x for x in (report.get("faces") or []) if x.get("name") == f.get("face")), None)
            if face:
                f["material_side"] = face.get("material_side", 1)
        try:
            new = cut_hirth(shape, axis_point, axis_dir, e1, f)
            shape = new
            f["modelled"] = True
            f["model_pending"] = False
            log(f"[face_features] cut Hirth N={f['N']} on the {f.get('face')} face")
        except Exception as ex:  # noqa: BLE001
            f["modelled"] = False
            f["model_pending"] = True
            f["not_modelled_reason"] = f"{type(ex).__name__}: {ex}"
    return shape, report
