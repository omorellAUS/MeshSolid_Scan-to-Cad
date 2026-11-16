"""Recognised design features of turned parts (M10): rolling-bearing boundary dimensions and
labyrinth seal teeth.  Report-only: the geometry comes from the regular builders.

bearing_ring  annular part (one bore, one outside cylinder, two end planes square to the axis): bore d,
              outside D and width B from the fitted surfaces, matched to the 60/62/63/NU2 tables
              (meshsolid.standards); accepted when every dimension is within max(0.05 mm, 2 tol)
labyrinth     N >= 3 equal, evenly spaced teeth on the bore or the outside: from the radial profile
              r(z) (bore: minimum radius per axial slice; outside: maximum), tip and groove levels,
              tooth count, pitch, tip land, root width, flank angle, tooth height"""
from __future__ import annotations

import math

import numpy as np


def _axis(regions):
    cy = [r for r in regions if r.primitive is not None and r.primitive.kind == "cylinder"]
    if not cy:
        return None
    ref = max(cy, key=lambda r: r.area)
    u = np.asarray(ref.primitive.params["axis"], float); u /= np.linalg.norm(u)
    return np.asarray(ref.primitive.params["point"], float), u


def _coaxial(r, a0, u, tol):
    q = r.primitive.params
    ax = np.asarray(q["axis"], float); ax /= np.linalg.norm(ax)
    d = np.asarray(q["point"], float) - a0
    return abs(abs(ax @ u) - 1) < 1e-3 and np.linalg.norm(d - (d @ u) * u) < max(0.1, 5 * tol)


def bearing_ring(mesh, regions, tol):
    from .standards import bearing_match
    ax = _axis(regions)
    if ax is None:
        return None
    a0, u = ax
    V = np.asarray(mesh.vertices) - a0
    z = V @ u; r = np.linalg.norm(V - np.outer(z, u), axis=1)
    cyl = [g for g in regions if g.primitive is not None and g.primitive.kind == "cylinder" and _coaxial(g, a0, u, tol)]
    if len(cyl) < 2:
        return None
    rad = sorted(float(g.primitive.params["radius"]) for g in cyl)
    ri, ro = rad[0], rad[-1]
    # the bore and the outside must be the extreme radii of the part
    if abs(np.percentile(r, 0.5) - ri) > max(0.5, 0.05 * ri) or abs(np.percentile(r, 99.5) - ro) > max(0.5, 0.02 * ro):
        return None
    pl = [g for g in regions if g.primitive is not None and g.primitive.kind == "plane"
          and abs(abs(np.asarray(g.primitive.params["normal"]) @ u) - 1) < 1e-3]
    if len(pl) < 2:
        return None
    zs = sorted(float((np.asarray(g.primitive.params["point"]) - a0) @ u) for g in pl)
    B = zs[-1] - zs[0]
    d, D = 2 * ri, 2 * ro
    cands = bearing_match(d, D, B)
    lim = max(0.05, 2 * tol)
    best = cands[0] if cands and cands[0]["misfit_mm"] <= lim else None
    out = {"bore_d": round(d, 4), "outside_D": round(D, 4), "width_B": round(B, 4), "match": None,
           "axis_point": [round(float(v), 5) for v in a0 + zs[0] * u], "axis_dir": [round(float(v), 7) for v in u],
           "nearest": cands[:3], "match_limit_mm": round(lim, 4)}
    if best:
        same = [c["designation"] for c in cands if abs(c["misfit_mm"] - best["misfit_mm"]) < 1e-9]
        out["match"] = {"designations": same, "d": best["d"], "D": best["D"], "B": best["B"], "misfit_mm": best["misfit_mm"],
                        "note": "boundary dimensions only: a ring with these sizes (bearing, or a spacer or housing insert "
                                "sized to one) - internal design (balls, rollers, cage) is not visible in a closed scan"}
    return out


def labyrinth(mesh, regions, tol, dz=0.05):
    ax = _axis(regions)
    if ax is None:
        return []
    a0, u = ax
    import trimesh
    # dense area samples (vertices are too sparse for 0.05 mm slices: a slice with no vertex on a narrow
    # tooth tip would read the far side of the part)
    n = int(min(2e6, max(2e5, mesh.area / (0.5 * dz) ** 2)))
    V = np.asarray(trimesh.sample.sample_surface(mesh, n, seed=0)[0]) - a0
    z = V @ u; r = np.linalg.norm(V - np.outer(z, u), axis=1)
    zb = np.arange(z.min(), z.max() + dz, dz)
    idx = np.clip(((z - z.min()) / dz).astype(int), 0, len(zb) - 1)
    out = []
    for side in ("bore", "outside"):
        prof = np.full(len(zb), np.nan)
        if side == "bore":
            np.fmin.at(prof, idx, r)
        else:
            np.fmax.at(prof, idx, r)
        ok = ~np.isnan(prof)
        if ok.sum() < 20:
            continue
        p = np.interp(np.arange(len(zb)), np.flatnonzero(ok), prof[ok])
        tip = np.percentile(p, 2) if side == "bore" else np.percentile(p, 98)
        sgn = 1 if side == "bore" else -1
        # candidate groove level: the most common level beyond the tips
        h = sgn * (p - tip)
        if h.max() < max(0.5, 10 * tol):
            continue
        hist, edges = np.histogram(h[h > max(0.3, 5 * tol)], bins=60)
        if hist.sum() == 0:
            continue
        g = 0.5 * (edges[np.argmax(hist)] + edges[np.argmax(hist) + 1])
        thr = 0.5 * g
        low = h < thr                                       # tooth (near the tip level)
        # runs of tooth slices bounded by groove on both sides
        runs, i = [], 0
        while i < len(low):
            if low[i]:
                j = i
                while j + 1 < len(low) and low[j + 1]:
                    j += 1
                if i > 0 and j < len(low) - 1:
                    runs.append((i, j))
                i = j + 1
            else:
                i += 1
        if len(runs) < 3:
            continue
        cen = np.array([0.5 * (zb[a] + zb[b]) for a, b in runs])
        wid = np.array([(b - a + 1) * dz for a, b in runs])
        # keep the largest group of teeth with equal width (+-30%)
        mw = np.median(wid); keep = np.abs(wid - mw) < 0.3 * mw + 2 * dz
        cen, wid = cen[keep], wid[keep]
        if len(cen) < 3:
            continue
        pitch = np.diff(cen)
        if np.std(pitch) > 0.1 * np.mean(pitch) + dz:
            continue
        # tip land: slices within 0.1 g of the tip level, per tooth; root width at 0.9 g
        def width_at(level):
            w = []
            for c in cen:
                k = np.abs(zb - c) < 0.5 * np.mean(pitch)
                w.append(float((h[k] < level).sum() * dz))
            return float(np.median(w))
        w15 = width_at(0.15 * g); w85 = width_at(0.85 * g)
        # straight flanks: extrapolate the widths at 15% / 85% height to the tip and the groove floor
        tip_land = max(0.0, w15 - (w85 - w15) * 0.15 / 0.7)
        root_w = w85 + (w85 - w15) * 0.15 / 0.7
        # tip / groove radii from the fitted coaxial cylinders when one sits at that level (the per-slice
        # extreme of noisy samples is biased outwards by the noise)
        rads = [float(q.primitive.params["radius"]) for q in regions if q.primitive is not None
                and q.primitive.kind == "cylinder" and _coaxial(q, a0, u, tol)]
        def snap_r(x):
            c = min(rads, key=lambda v: abs(v - x)) if rads else None
            return c if c is not None and abs(c - x) < max(0.2, 4 * tol) else x
        tip = snap_r(float(tip)); grv = snap_r(float(tip + sgn * g))
        hgt = abs(grv - tip)
        flank = math.degrees(math.atan2(max(w85 - w15, 0.0) / 2, 0.7 * hgt))
        out.append({"side": side, "teeth": int(len(cen)), "pitch": round(float(np.mean(pitch)), 4),
                    "pitch_std": round(float(np.std(pitch)), 4), "tip_diameter": round(2 * float(tip), 4),
                    "groove_diameter": round(2 * grv, 4), "tooth_height": round(hgt, 4), "root_width": round(root_w, 4),
                    "tip_land": round(tip_land, 4), "flank_angle_deg": round(flank, 2),
                    "first_tooth_z_axis_frame": round(float(cen[0]), 3),
                    "basis": "radial profile of the scan (min/max radius per 0.05 mm slice of dense surface samples); tip land, "
                             "root width and flank angle from the widths at 15% and 85% of the tooth height extrapolated "
                             "along straight flanks (the scan rounds the tip corners); diameters from the fitted cylinders"})
    return out


def rolling_elements(mesh, regions, tol):
    """Balls or rollers visible between the rings of an open bearing: sphere regions (balls) or
    cylinder regions with axes parallel to the bearing axis but off it (rollers), whose centres sit on one
    pitch circle at equal angular spacing.  Returns a dict or None."""
    ax = _axis(regions)
    if ax is None:
        return None
    a0, u = ax

    def polar(c):
        d = np.asarray(c, float) - a0
        z = float(d @ u); rv = d - z * u
        return float(np.linalg.norm(rv)), rv, z
    items = []
    for g in regions:
        p = g.primitive
        if p is None:
            continue
        if p.kind == "sphere":
            rad, rv, z = polar(p.params["center"])
            items.append(("ball", float(p.params["radius"]), rad, rv, z, g.area))
        elif p.kind == "cylinder":
            a = np.asarray(p.params["axis"], float); a /= np.linalg.norm(a)
            if abs(abs(a @ u) - 1) > 2e-3:
                continue
            rad, rv, z = polar(p.params["point"])
            if rad > max(1.0, 0.1 * float(p.params["radius"])):
                items.append(("roller", float(p.params["radius"]), rad, rv, None, g.area))
    best = None
    for kind in ("ball", "roller"):
        it = [q for q in items if q[0] == kind]
        if len(it) < 3:
            continue
        rads = np.array([q[1] for q in it]); rps = np.array([q[2] for q in it])
        r0 = float(np.median(rads)); rp0 = float(np.median(rps))
        ok = [q for q in it if abs(q[1] - r0) < max(0.1, 0.04 * r0) and abs(q[2] - rp0) < max(0.2, 0.03 * rp0)]
        # one element can show as several patches (both faces of the bearing): merge by centre angle
        e1 = ok[0][3] / max(np.linalg.norm(ok[0][3]), 1e-9) if ok else None
        if not ok:
            continue
        e2 = np.cross(u, e1)
        ang = sorted(math.atan2(q[3] @ e2, q[3] @ e1) % (2 * math.pi) for q in ok)
        groups = []
        for a in ang:
            if groups and (a - groups[-1][-1]) < max(0.05, 0.5 * r0 / rp0):
                groups[-1].append(a)
            else:
                groups.append([a])
        if len(groups) > 1 and (groups[0][0] + 2 * math.pi - groups[-1][-1]) < max(0.05, 0.5 * r0 / rp0):
            groups[0] = groups[-1] + groups[0]; groups.pop()
        Z = len(groups)
        if Z < 3:
            continue
        cen = sorted(float(np.mean(np.unwrap(g_))) % (2 * math.pi) for g_ in groups)
        gaps = np.diff(cen + [cen[0] + 2 * math.pi])
        spacing_cv = float(np.std(gaps) / np.mean(gaps))
        if spacing_cv > 0.08:
            continue
        cand = {"elements": kind, "count": Z, "diameter": round(2 * r0, 4), "pitch_diameter": round(2 * rp0, 4),
                "spacing_deg": round(360.0 / Z, 3), "spacing_cv": round(spacing_cv, 4),
                "patches": len(ok), "element_fit_spread_mm": round(float(np.ptp([q[1] for q in ok])) * 2, 4)}
        if kind == "ball":
            cand["centre_plane_z_axis_frame"] = round(float(np.median([q[4] for q in ok])), 4)
        if best is None or Z * len(ok) > best["count"] * best["patches"]:
            best = cand
    if best is not None:
        best["basis"] = ("fitted " + ("sphere" if best["elements"] == "ball" else "cylinder") + " patches between the rings; "
                         "count from distinct centre angles on one pitch circle; diameters measured (not snapped: "
                         "ball and roller sizes are internal design data the catalogue boundary dimensions do not fix)")
    return best
