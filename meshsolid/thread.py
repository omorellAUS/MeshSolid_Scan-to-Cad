"""Thread recognition on scan meshes.

A helical thread breaks region growing into many coaxial ring-like patches (tori, cylinders, cones).
We detect it on the raw mesh instead: for a candidate axis (from clusters of coaxial regions) the radius
of the surface points is a periodic function of the helical phase  s = z - h * p * theta / (2 pi)  (mod p)
for the right pitch p and handedness h, while it looks like noise for any other (p, h). We score each
standard pitch by the fraction of radial variance explained by the phase (binned means), refine the best
pitch, find the axial extent where the pattern holds, and fit major/minor diameter and taper.

Convention (recorded in the spec): the thread zone is replaced in the solid by a plain cylinder at the
MAJOR diameter for external threads and at the MINOR diameter for internal threads (the diameter a Fusion
thread feature is applied to: a rod at major diameter, a tapped hole at minor/tap-drill diameter).
"""
from __future__ import annotations

import numpy as np

from . import fit as F

# (designation, major diameter mm, pitch mm, kind) ; kind: metric | bspp | npt (tapered) | unc | unf
_TPI = lambda n: 25.4 / n  # noqa: E731
STANDARDS = []
for d, coarse, fines in [(3, .5, [.35]), (4, .7, [.5]), (5, .8, [.5]), (6, 1.0, [.75]), (8, 1.25, [1.0, .75]),
                         (10, 1.5, [1.25, 1.0, .75]), (12, 1.75, [1.5, 1.25, 1.0]), (14, 2.0, [1.5]),
                         (16, 2.0, [1.5, 1.0]), (18, 2.5, [2.0, 1.5]), (20, 2.5, [2.0, 1.5, 1.0]),
                         (22, 2.5, [2.0, 1.5]), (24, 3.0, [2.0, 1.5]), (27, 3.0, [2.0]), (30, 3.5, [2.0, 1.5]),
                         (33, 3.5, [2.0]), (36, 4.0, [3.0, 2.0]), (42, 4.5, [3.0, 2.0]), (48, 5.0, [3.0, 2.0])]:
    STANDARDS.append((f"M{d}x{coarse:g}" if False else f"M{d}", float(d), coarse, "metric"))
    for f_ in fines:
        STANDARDS.append((f"M{d}x{f_:g}", float(d), f_, "metric"))
for name, d, tpi in [("G1/8", 9.728, 28), ("G1/4", 13.157, 19), ("G3/8", 16.662, 19), ("G1/2", 20.955, 14),
                     ("G3/4", 26.441, 14), ("G1", 33.249, 11)]:
    STANDARDS.append((name + " (BSPP)", d, _TPI(tpi), "bspp"))
for name, d, tpi in [("1/8 NPT", 10.287, 27), ("1/4 NPT", 13.716, 18), ("3/8 NPT", 17.145, 18),
                     ("1/2 NPT", 21.336, 14), ("3/4 NPT", 26.670, 14), ("1 NPT", 33.401, 11.5)]:
    STANDARDS.append((name, d, _TPI(tpi), "npt"))
for name, d, tpi in [("1/4-20 UNC", 6.35, 20), ("5/16-18 UNC", 7.938, 18), ("3/8-16 UNC", 9.525, 16),
                     ("1/2-13 UNC", 12.7, 13), ("1/4-28 UNF", 6.35, 28), ("3/8-24 UNF", 9.525, 24),
                     ("1/2-20 UNF", 12.7, 20)]:
    STANDARDS.append((name, d, _TPI(tpi), "unc" if "UNC" in name else "unf"))
CANDIDATE_PITCHES = sorted({round(s[2], 4) for s in STANDARDS})


def _frame(axis):
    a = axis / np.linalg.norm(axis)
    t = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(a, t); u /= np.linalg.norm(u)
    return a, u, np.cross(a, u)


def cylindrical(P, point, axis):
    a, u, v = _frame(axis)
    d = P - point
    z = d @ a
    w = d - np.outer(z, a)
    r = np.linalg.norm(w, axis=1)
    th = np.arctan2(w @ v, w @ u)
    return r, z, th


def phase_score(r, z, th, pitch, hand, nb=24):
    """Fraction of the variance of r explained by binned helical phase; also the bin means."""
    ph = np.mod((z - hand * pitch * th / (2 * np.pi)) / pitch, 1.0)
    b = np.minimum((ph * nb).astype(int), nb - 1)
    cnt = np.bincount(b, minlength=nb)
    if (cnt < 3).any():
        return 0.0, None
    mean = np.bincount(b, r, minlength=nb) / cnt
    var_tot = float(np.var(r))
    if var_tot <= 0:
        return 0.0, None
    resid = float(np.mean((r - mean[b]) ** 2))
    return 1.0 - resid / var_tot, mean


def _axis_candidates(regions, diag, min_regions=5):
    """Clusters of coaxial cylinder/cone/torus regions -> (axis, point, region ids)."""
    items = []
    for r in regions:
        p = r.primitive
        if p is None:
            continue
        if p.kind == "cylinder":
            items.append((r.id, np.asarray(p.params["axis"]), np.asarray(p.params["point"]), r.area))
        elif p.kind == "cone":
            items.append((r.id, np.asarray(p.params["axis"]), np.asarray(p.params["apex"]), r.area))
        elif p.kind == "torus":
            items.append((r.id, np.asarray(p.params["axis"]), np.asarray(p.params["center"]), r.area))
    used, out = set(), []
    for i, (ri, ai, pi, wi) in enumerate(items):
        if ri in used:
            continue
        mem = []
        for rj, aj, pj, wj in items:
            if rj in used or abs(ai @ aj) < np.cos(np.radians(4.0)):
                continue
            d = pj - pi
            off = np.linalg.norm(d - (d @ ai) * ai)
            if off < 0.02 * diag + 0.5:
                mem.append((rj, aj if aj @ ai > 0 else -aj, pj, wj))
        if len(mem) >= min_regions:
            used.update(m[0] for m in mem)
            W = np.array([m[3] for m in mem])
            A = np.average(np.array([m[1] for m in mem]), axis=0, weights=W)
            out.append((A / np.linalg.norm(A), np.average(np.array([m[2] for m in mem]), axis=0, weights=W),
                        [m[0] for m in mem]))
    return out


def _refine_axis(P, axis, point):
    """Least-squares cylinder through the points (axis refinement; radius discarded)."""
    try:
        c = F.fit_cylinder(P, None, axis0=axis)
        a = np.asarray(c.params["axis"])
        if abs(a @ axis) > np.cos(np.radians(5)):
            return a / np.linalg.norm(a), np.asarray(c.params["point"])
    except Exception:  # noqa: BLE001
        pass
    return axis, point


def standard_designation(major_d, pitch, tapered):
    best = None
    for name, d, p, kind in STANDARDS:
        if (kind == "npt") != bool(tapered):
            continue
        ed = abs(major_d - d) / max(0.03 * d, 0.15)
        ep = abs(pitch - p) / (0.03 * p)
        e = np.hypot(ed, ep)
        if best is None or e < best[0]:
            best = (e, name, d, p, kind)
    if best is None:
        return None
    e, name, d, p, kind = best
    if name.startswith("M") and "x" not in name:
        name = f"{name}x{p:g}"   # coarse pitch written explicitly
    return {"designation": name, "nominal_major_d": d, "nominal_pitch": round(p, 4), "standard": kind,
            "match_error": round(float(e), 3), "match": "good" if e < 1 else ("approximate" if e < 2.5 else "poor")}


def detect_threads(mesh, regions, labels, tol, min_turns=3, min_score=0.55):
    """Returns a list of thread specs (dicts) with the mesh faces they cover."""
    V = np.asarray(mesh.vertices)
    Fc = np.asarray(mesh.faces)
    cent = np.asarray(mesh.triangles_center)
    fn = np.asarray(mesh.face_normals)
    diag = float(np.linalg.norm(mesh.extents))
    specs = []
    claimed = np.zeros(len(Fc), bool)
    for axis, point, rids in _axis_candidates(regions, diag):
        fids = np.concatenate([regions_by_id(regions)[i].faces for i in rids])
        Pc = cent[fids]
        axis, point = _refine_axis(Pc, axis, point)
        r_c, z_c, _ = cylindrical(Pc, point, axis)
        z0, z1 = np.percentile(z_c, [0.5, 99.5])
        rlo, rhi = np.percentile(r_c, [2, 98])
        # all faces in that axial range and radial band (threads span many small regions + unassigned)
        r_all, z_all, th_all = cylindrical(cent, point, axis)
        band = (z_all >= z0 - 1) & (z_all <= z1 + 1) & (r_all >= 0.6 * rlo) & (r_all <= 1.15 * rhi + 1) & ~claimed
        # flanks/crests/roots only: normals not nearly axial
        _, _, _ = None, None, None
        ax_n = np.abs(fn @ axis)
        # external and internal surfaces separately (a pipe fitting has a bore on the same axis): faces
        # whose normal points away from (external) or toward (internal) the axis, and only the main
        # radial mode of those (+-2 mm), so other coaxial surfaces do not swamp the helical signal
        rad_v = (cent - point) - np.outer(z_all, axis)
        rad_v /= np.maximum(np.linalg.norm(rad_v, axis=1, keepdims=True), 1e-12)
        n_rad = np.einsum("ij,ij->i", fn, rad_v)
        area_f = np.asarray(mesh.area_faces)
        for orient in (1.0, -1.0):
            sel = band & (ax_n < 0.95)
            sel = sel & (orient * n_rad > 0.2)
            if sel.sum() < 300:
                continue
            hb = np.round(r_all[sel] / 0.25).astype(int)
            mode_r = 0.25 * np.bincount(hb - hb.min(), weights=area_f[sel]).argmax() + 0.25 * hb.min()
            sel = sel & (np.abs(r_all - mode_r) < 2.0)
            if sel.sum() < 300:
                continue
            r, z, th = r_all[sel], z_all[sel], th_all[sel]
            # coarse search on the whole band and on 10 mm sub-windows (a thread usually covers only part
            # of the coaxial band; plain surfaces in the band dilute a whole-band score)
            best = (0.0, None, None, None)
            spans = [(z.min(), z.max() + 1e-9)]
            if z.max() - z.min() > 14:
                spans += [(a, a + 10.0) for a in np.arange(z.min(), z.max() - 5, 5.0)]
            for a_, b_ in spans:
                mw = (z >= a_) & (z < b_)
                if mw.sum() < 300:
                    continue
                for p in CANDIDATE_PITCHES:
                    if p * min_turns > min(b_ - a_, z.max() - z.min()) or p < 4 * tol:
                        continue
                    for h in (1, -1):
                        s, _ = phase_score(r[mw], z[mw], th[mw], p, h)
                        if s > best[0]:
                            best = (s, p, h, mw)
            s0, p0, h0, mbest = best
            if p0 is None or s0 < 0.3:
                continue
            # refine pitch continuously around the best standard value (on the best window)
            ps = np.linspace(0.97 * p0, 1.03 * p0, 61)
            sc = [phase_score(r[mbest], z[mbest], th[mbest], p, h0)[0] for p in ps]
            p_best = float(ps[int(np.argmax(sc))])
            # axial extent: sliding windows of 2 pitches where the phase pattern holds
            win = 2 * p_best
            edges = np.arange(z.min(), z.max() + 1e-9, p_best / 2)
            good = []
            for e in edges:
                m = (z >= e) & (z < e + win)
                if m.sum() < 60:
                    good.append(False)
                    continue
                # within a window: fit phase with the global bin means, require explained variance
                sw, _ = phase_score(r[m], z[m], th[m], p_best, h0, nb=12)
                good.append(sw >= min_score)
            good = np.array(good)
            if not good.any():
                continue
            # longest run of good windows
            best_run, cur, start = (0, 0, 0), 0, 0
            for i, g_ in enumerate(good):
                if g_:
                    if cur == 0:
                        start = i
                    cur += 1
                    if cur > best_run[0]:
                        best_run = (cur, start, i)
                else:
                    cur = 0
            n_run, i0, i1 = best_run
            zs, ze = float(edges[i0]), float(edges[i1] + win)
            zs, ze = _refine_extent(r, z, th, p_best, h0, zs, ze)
            if ze - zs < min_turns * p_best:
                continue
            m = (z >= zs) & (z <= ze)
            score, means = phase_score(r[m], z[m], th[m], p_best, h0)
            if means is None or score < min_score:
                continue
            # crest/root radii and taper: per-window high/low phase-bin means vs z
            zc, rcrest, rroot = [], [], []
            for e in np.arange(zs, ze - win + 1e-9, p_best):
                mm = m & (z >= e) & (z < e + win)
                sw, mw = phase_score(r[mm], z[mm], th[mm], p_best, h0, nb=24)
                if mw is None:
                    sw, mw = phase_score(r[mm], z[mm], th[mm], p_best, h0, nb=12)
                if mw is None:
                    continue
                zc.append(e + win / 2); rcrest.append(mw.max()); rroot.append(mw.min())
            zc, rcrest, rroot = map(np.asarray, (zc, rcrest, rroot))
            if len(zc) >= 3:
                k, c = np.polyfit(zc, rcrest, 1)
            else:
                k, c = 0.0, float(np.mean(rcrest)) if len(rcrest) else float(means.max())
            taper_deg = float(np.degrees(np.arctan(abs(k))))
            tapered = taper_deg > 0.9 and abs(k) * (ze - zs) > 4 * tol     # NPT: 1.79 deg half angle
            # internal vs external: face normals point away from the axis on an external thread
            radial = (cent[sel][m] - point) - np.outer(z[m], axis)
            radial /= np.maximum(np.linalg.norm(radial, axis=1, keepdims=True), 1e-12)
            outward = float(np.mean(np.einsum("ij,ij->i", fn[sel][m], radial)))
            kind = "external" if orient > 0 else "internal"
            del outward
            crest = float(np.median(rcrest)) if len(rcrest) else float(means.max())
            root = float(np.median(rroot)) if len(rroot) else float(means.min())
            major_r, minor_r = (crest, root) if kind == "external" else (root, crest)
            zmid = 0.5 * (zs + ze)
            major_d = 2 * (major_r if not tapered else (c + k * zmid) if kind == "external" else major_r)
            std = standard_designation(major_d, p_best, tapered)
            faces = np.where(sel)[0][m]
            # the zone: every face in the band within [zs, ze] (including near-axial flank pieces)
            zone = band & (z_all >= zs) & (z_all <= ze)
            claimed |= zone
            r_meas = major_r if kind == "external" else minor_r
            r_model = r_meas
            if std and std["match"] == "good" and not tapered:
                if kind == "external":
                    r_model = std["nominal_major_d"] / 2
                else:   # ISO basic internal minor diameter D1 = D - 1.0825 P
                    r_model = (std["nominal_major_d"] - 1.0825 * std["nominal_pitch"]) / 2
            conf = float(np.clip((score - min_score) / (0.95 - min_score), 0, 1))
            if std and std["match"] != "good":
                conf *= 0.7
            specs.append({"_tol": float(tol),
                "kind": kind, "axis": axis.tolist(), "axis_point": point.tolist(),
                "start": (point + zs * axis).tolist(), "end": (point + ze * axis).tolist(),
                "length": ze - zs, "major_d": major_d, "minor_d": 2 * minor_r, "pitch": p_best,
                "turns": (ze - zs) / p_best, "hand": "right" if h0 > 0 else "left",
                "taper_half_angle_deg": taper_deg if tapered else 0.0, "tapered": bool(tapered),
                "phase_score": score, "standard": std, "confidence": round(conf, 3),
                "modelled_as": f"cylinder at {'major' if kind == 'external' else 'minor'} diameter",
                "_zone_faces": np.where(zone)[0], "_axis": axis, "_point": point, "_z": (zs, ze),
                "model_diameter": 2 * r_model,
                "model_diameter_source": "nominal (standard match good)" if r_model != r_meas else "measured",
                "_r_model": r_model, "_k": k, "_c": c,
                # material the plain cylinder adds over the real threaded zone (groove for external,
                # ridge for internal): pi L |R_model^2 - <r^2>| with <r^2> over the phase-binned profile
                "volume_delta_mm3": float(np.pi * (ze - zs) * abs(
                    (r_model if not tapered else (c + k * zmid)) ** 2 - float(np.mean(np.asarray(means) ** 2)))),
            })
    return specs


def _refine_extent(r, z, th, p, h, zs, ze, nb=24):
    """Move each end of the zone to the change point between 'thread' (r explained by the helical phase
    means) and 'non-thread' (r explained by a constant per side) within +-1.5 pitches."""
    m = (z >= zs) & (z <= ze)
    _, means = phase_score(r[m], z[m], th[m], p, h, nb)
    if means is None:
        return zs, ze
    ph = np.mod((z - h * p * th / (2 * np.pi)) / p, 1.0)
    res_t = (r - means[np.minimum((ph * nb).astype(int), nb - 1)]) ** 2

    def change(lo, hi, outside_below):
        cand = np.linspace(lo, hi, 61)
        best, bz = None, None
        for b in cand:
            out = (z < b) if outside_below else (z > b)
            band = (z >= lo - p) & (z <= hi + p)
            o, t = band & out, band & ~out
            if o.sum() < 10 or t.sum() < 10:
                continue
            cost = res_t[t].sum() + ((r[o] - np.median(r[o])) ** 2).sum()
            if best is None or cost < best:
                best, bz = cost, b
        return bz
    a = change(zs - 1.5 * p, zs + 1.5 * p, True)
    b = change(ze - 1.5 * p, ze + 1.5 * p, False)
    return (a if a is not None else zs), (b if b is not None else ze)


def regions_by_id(regions):
    return {r.id: r for r in regions}


def apply_threads(mesh, regions, labels, specs, Region):
    """Replace each thread zone by one cylinder (or cone, if tapered) region. Returns new
    (regions, labels) renumbered 0..n-1."""
    labels = labels.copy()
    new = list(regions)
    area = np.asarray(mesh.area_faces)
    next_id = max(r.id for r in regions) + 1
    for s in specs:
        z = s["_zone_faces"]
        if s["tapered"]:
            a, p0 = s["_axis"], s["_point"]
            k, c = s["_k"], s["_c"]
            # r(z) = c + k z  -> cone with apex where r = 0
            al = float(np.arctan(abs(k)))
            ax = a if k < 0 else -a          # cone axis points from apex toward growing radius
            apex = p0 + a * (-c / k)
            ax = a if k > 0 else -a
            prim = F.Primitive("cone", {"apex": apex, "axis": ax, "half_angle": al})
        else:
            prim = F.Primitive("cylinder", {"point": s["_point"], "axis": s["_axis"], "radius": float(s["_r_model"])})
        prim.rms, prim.max_err, prim.normal_err_deg = 0.0, 0.0, 0.0
        depth = abs(s["major_d"] - s["minor_d"]) / 2
        keep_dist = 1.2 * depth + 0.1
        # a plain cylinder of the same diameter on the same axis (e.g. a shank at major diameter) is the
        # same CAD face: merge the zone into it, the thread feature then covers [start, end] of that face
        host = None
        if not s["tapered"]:
            for r in new:
                q = r.primitive
                if q is None or q.kind != "cylinder" or "thread_zone" in r.flags:
                    continue
                a = np.asarray(q.params["axis"])
                if abs(a @ s["_axis"]) < np.cos(np.radians(1.0)):
                    continue
                d = np.asarray(q.params["point"]) - s["_point"]
                if np.linalg.norm(d - (d @ s["_axis"]) * s["_axis"]) > 0.1 + 0.02 * s["_r_model"]:
                    continue
                if abs(float(q.params["radius"]) - s["_r_model"]) <= max(0.1, 0.01 * s["_r_model"]):
                    host = r
                    break
        if host is not None:
            labels[z] = host.id
            host.flags = list(host.flags) + ["thread_zone"]
            host.keep_dist = keep_dist
            s["region"] = host.id
            s["merged_into_existing_cylinder"] = True
            continue
        labels[z] = next_id
        reg = Region(next_id, z, prim, float(area[z].sum()))
        reg.flags = ["thread_zone"]
        reg.keep_dist = keep_dist
        reg.confidence = s["confidence"]
        new.append(reg)
        s["region"] = next_id
        next_id += 1
    for s in specs:
        _absorb_runout(mesh, new, labels, s, s["region"])
    # drop emptied regions, shrink partially covered ones, renumber
    out, remap = [], {}
    for r in new:
        f = np.where(labels == r.id)[0]
        if len(f) == 0:
            continue
        r.faces = f
        r.area = float(area[f].sum())
        remap[r.id] = len(out)
        out.append(r)
    lab2 = -np.ones_like(labels)
    for old, nid in remap.items():
        lab2[labels == old] = nid
    for r in out:
        r.id = remap[r.id]
        # region-id tags (coaxial partner) must follow the renumbering
        fl = []
        for x in r.flags:
            if x.startswith("coaxial_cone_with_"):
                j = int(x.rsplit("_", 1)[1])
                if j not in remap:
                    continue
                x = f"coaxial_cone_with_{remap[j]}"
            fl.append(x)
        r.flags = fl
    for s in specs:
        s["region"] = remap.get(s["region"])
    return out, lab2


def _absorb_runout(mesh, regions, labels, s, tid):
    """Thread run-out and chamfer-through-thread debris: small regions lying inside the thread band
    (axially within a pitch of the zone, radially between minor and major) are pieces of the helix the
    segmenter split off. Cones there are refitted on the thread axis (a real end chamfer) and kept when
    the coaxial fit is good; everything else in the band is absorbed into the thread region."""
    a, p0 = np.asarray(s["_axis"]), np.asarray(s["_point"])
    pitch = s["pitch"]
    r_min, r_max = s["minor_d"] / 2, s["major_d"] / 2
    depth = r_max - r_min
    zone = s["_zone_faces"]
    C = np.asarray(mesh.triangles_center)
    area = np.asarray(mesh.area_faces)
    d = C - p0
    h = d @ a
    rad = np.linalg.norm(d - h[:, None] * a, axis=1)
    z0, z1 = float(h[zone].min()), float(h[zone].max())
    inband = (h >= z0 - pitch) & (h <= z1 + pitch) & (rad >= r_min - 0.5 * depth) & (rad <= r_max + 0.5 * depth)
    host = next(r for r in regions if r.id == tid)
    absorbed, kept = [], []
    for r in regions:
        if r.id == tid:
            continue
        f = np.where(labels == r.id)[0]
        if len(f) == 0:
            continue
        frac = float(area[f][inband[f]].sum() / max(area[f].sum(), 1e-12))
        if frac < 0.9:
            continue
        q = r.primitive
        if q is not None and q.kind == "cone" and area[f].sum() > 4 * pitch * pitch:
            # coaxial refit: r = c0 + c1 h on the thread axis
            A = np.c_[np.ones(len(f)), h[f]]
            (c0, c1), *_ = np.linalg.lstsq(A, rad[f], rcond=None)
            res = rad[f] - (c0 + c1 * h[f])
            rms = float(np.sqrt(np.mean(res ** 2)))
            s.setdefault("_chamfer_fits", []).append((r.id, round(rms, 4), round(float(np.degrees(np.arctan(abs(c1)))), 2)))
            if abs(c1) > 1e-3 and rms <= 2.0 * max(s.get("_tol", 0.05), 0.05):
                al = float(np.arctan(abs(c1)))
                apex = p0 + a * (-c0 / c1)
                ax = a if c1 > 0 else -a
                r.primitive = F.Primitive("cone", {"apex": apex, "axis": ax, "half_angle": al})
                r.primitive.rms, r.primitive.max_err = rms, float(np.abs(res).max())
                r.primitive.normal_err_deg = getattr(q, "normal_err_deg", 0.0)
                r.flags = [x for x in r.flags if not x.startswith("coaxial_cone_with_")] + [f"coaxial_cone_with_{tid}", "thread_end_chamfer"]
                kept.append(r.id)
                continue
        labels[f] = tid
        absorbed.append(r.id)
    s["runout_regions_absorbed"] = len(absorbed)
    s["end_chamfers_kept"] = len(kept)
    host.faces = np.where(labels == tid)[0]
    if absorbed:   # absorbed debris is thread geometry too (excluded from the cylinder deviation)
        s["_zone_faces"] = np.union1d(zone, np.where(np.isin(labels, [tid]) & inband)[0])


def spec_json(s):
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in s.items() if not k.startswith("_")}
