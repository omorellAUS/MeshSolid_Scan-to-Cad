"""Snapping of fitted primitives: parallel/perpendicular directions, coplanar and
coaxial merging, tangency of cylinders to planes, and rounding of dimensions.

Every change is recorded (raw value -> snapped value) and only applied when the
snapped primitive still fits the region's points (rms growth bounded)."""
from __future__ import annotations

import numpy as np

from . import fit as F
from .segment import region_points


def _u(v):
    return v / np.linalg.norm(v)


def _ang(a, b):
    return np.degrees(np.arccos(np.clip(abs(a @ b), 0, 1)))


def _round(x, tol, grids=(1.0, 0.5)):
    for gsz in grids:
        r = round(x / gsz) * gsz
        if abs(r - x) <= tol:
            return float(r), gsz
    return float(x), None


def _dir_of(p):
    return p.params["normal"] if p.kind == "plane" else p.params.get("axis")


def refit_fixed_dir(kind, P, N, d, point_hint=None):
    d = _u(d)
    if kind == "plane":
        if N is not None and np.mean(N @ d) < 0:
            d = -d
        off = float(np.mean(P @ d))
        c = P.mean(0)
        pt = c - (c @ d - off) * d
        return F._finish(F.Primitive("plane", {"point": pt, "normal": d}), P, N)
    if kind == "cylinder":
        u, v = F._perp_basis(d)
        c0 = P.mean(0)
        xy = np.c_[(P - c0) @ u, (P - c0) @ v]
        cc, r = F._circle_fit_2d(xy)
        from scipy.optimize import least_squares
        sol = least_squares(lambda x: np.linalg.norm(xy - x[:2], axis=1) - x[2], [*cc, r], method="lm")
        pt = c0 + sol.x[0] * u + sol.x[1] * v
        return F._finish(F.Primitive("cylinder", {"point": pt, "axis": d, "radius": abs(sol.x[2])}), P, N)
    raise ValueError(kind)


def snap(mesh_graph, regions, tol=0.1, angle_deg=2.0, round_tol=0.03, do_round=True):
    g = mesh_graph
    log = []
    pts = {r.id: region_points(g, r.faces) for r in regions}
    # ---- 1. direction clusters (unsigned), area weighted
    items = [(r, _dir_of(r.primitive)) for r in regions if r.primitive.kind in ("plane", "cylinder", "cone", "torus")]
    items.sort(key=lambda t: -t[0].area)
    clusters = []  # [dir, weight, members]
    for r, d in items:
        for c in clusters:
            if _ang(c[0], d) <= angle_deg:
                s = 1 if c[0] @ d > 0 else -1
                c[0] = _u(c[0] * c[1] + s * d * r.area)
                c[1] += r.area
                c[2].append(r)
                break
        else:
            clusters.append([_u(d), r.area, [r]])
    clusters.sort(key=lambda c: -c[1])
    # ---- 2. orthogonal frame from the two heaviest near-perpendicular clusters
    frame = []
    if clusters:
        e1 = clusters[0][0]
        frame = [e1]
        for c in clusters[1:]:
            if abs(_ang(c[0], e1) - 90) <= angle_deg:
                e2 = _u(c[0] - (c[0] @ e1) * e1)
                frame += [e2, np.cross(e1, e2)]
                break
    for c in clusters:
        for e in frame:
            if _ang(c[0], e) <= angle_deg:
                log.append({"op": "snap_direction", "regions": [r.id for r in c[2]],
                            "angle_deg": float(_ang(c[0], e))})
                c[0] = e * (1 if c[0] @ e > 0 else -1)
                break
    # ---- 3. refit with fixed direction
    dir_of_region = {}
    for c in clusters:
        for r in c[2]:
            dir_of_region[r.id] = c[0]
            if r.primitive.kind in ("plane", "cylinder"):
                P, N = pts[r.id]
                old = r.primitive
                new = refit_fixed_dir(old.kind, P, N, c[0])
                if new.rms <= max(1.5 * old.rms, old.rms + 0.2 * tol):
                    r.primitive = new
                else:
                    log.append({"op": "reject_direction_snap", "region": r.id,
                                "rms_old": old.rms, "rms_new": new.rms})
            elif r.primitive.kind in ("cone", "torus"):
                q = r.primitive.params
                q["axis"] = c[0] * (1 if q["axis"] @ c[0] > 0 else -1)
    # ---- 4. coplanar merge & coaxial merge
    planes = [r for r in regions if r.primitive.kind == "plane"]
    for i, a in enumerate(planes):
        for b in planes[i + 1:]:
            na, nb = a.primitive.params["normal"], b.primitive.params["normal"]
            if na @ nb > np.cos(np.radians(angle_deg)):
                da, db = a.primitive.params["point"] @ na, b.primitive.params["point"] @ na
                if abs(da - db) < tol and b.primitive is not a.primitive:
                    P = np.vstack([pts[a.id][0], pts[b.id][0]])
                    N = np.vstack([pts[a.id][1], pts[b.id][1]])
                    p = refit_fixed_dir("plane", P, N, na)
                    a.primitive = b.primitive = p
                    log.append({"op": "coplanar", "regions": [a.id, b.id], "gap": float(abs(da - db))})
    cyls = [r for r in regions if r.primitive.kind in ("cylinder", "cone", "torus")]
    for i, a in enumerate(cyls):
        for b in cyls[i + 1:]:
            qa, qb = a.primitive.params, b.primitive.params
            if _ang(qa["axis"], qb["axis"]) > angle_deg:
                continue
            pa = qa.get("point", qa.get("apex", qa.get("center")))
            pb = qb.get("point", qb.get("apex", qb.get("center")))
            d = pb - pa
            off = np.linalg.norm(d - (d @ qa["axis"]) * qa["axis"])
            if off < tol:
                w = a.area / (a.area + b.area)
                ax = qa["axis"]
                ctr = pa + (1 - w) * (d - (d @ ax) * ax)   # common axis line point
                for r in (a, b):
                    q = r.primitive.params
                    key = "point" if "point" in q else ("apex" if "apex" in q else "center")
                    p0 = q[key]
                    q[key] = ctr + ((p0 - ctr) @ ax) * ax
                    q["axis"] = ax * (1 if q["axis"] @ ax > 0 else -1)
                log.append({"op": "coaxial", "regions": [a.id, b.id], "axis_offset": float(off)})
    # ---- 4b. identical cylinders (coaxial, same radius) share one primitive
    cyl_only = [r for r in regions if r.primitive.kind == "cylinder"]
    for i, a in enumerate(cyl_only):
        for b in cyl_only[i + 1:]:
            qa, qb = a.primitive.params, b.primitive.params
            if a.primitive is b.primitive or _ang(qa["axis"], qb["axis"]) > angle_deg:
                continue
            d = qb["point"] - qa["point"]
            if np.linalg.norm(d - (d @ qa["axis"]) * qa["axis"]) < 1e-9 and abs(qa["radius"] - qb["radius"]) < tol:
                P = np.vstack([pts[a.id][0], pts[b.id][0]])
                N = np.vstack([pts[a.id][1], pts[b.id][1]])
                p = refit_fixed_dir("cylinder", P, N, qa["axis"])
                p.params["point"] = qa["point"]  # keep the shared axis line
                for r in regions:
                    if r.primitive is a.primitive or r.primitive is b.primitive:
                        r.primitive = p
                log.append({"op": "same_cylinder", "regions": [a.id, b.id]})
    # ---- 5. tangency: cylinder parallel to plane with |dist(axis,plane) - R| < tol -> exact
    for c in [r for r in regions if r.primitive.kind == "cylinder"]:
        q = c.primitive.params
        tang = []
        for p in planes:
            n = p.primitive.params["normal"]
            if abs(n @ q["axis"]) > np.sin(np.radians(angle_deg)):
                continue
            dist = (q["point"] - p.primitive.params["point"]) @ n
            if abs(abs(dist) - q["radius"]) < tol and _adjacent(g, c, p):
                tang.append((p, n, dist))
        if 1 <= len(tang) <= 2:
            # solve for axis point (in plane perpendicular to axis) and R so that tangency holds,
            # least squares with the region's points
            P, N = pts[c.id]
            from scipy.optimize import least_squares
            ax = q["axis"]
            u, v = F._perp_basis(ax)

            def res(x):
                pt = q["point"] + x[0] * u + x[1] * v
                cyl = F.Primitive("cylinder", {"point": pt, "axis": ax, "radius": x[2]})
                r1 = F.signed_distance(cyl, P)
                cons = [((pt - p.primitive.params["point"]) @ n) - np.sign(dist) * x[2] for p, n, dist in tang]
                return np.r_[r1, 100 * np.array(cons)]
            sol = least_squares(res, [0, 0, q["radius"]])
            newp = F._finish(F.Primitive("cylinder", {"point": q["point"] + sol.x[0] * u + sol.x[1] * v,
                                                      "axis": ax, "radius": abs(sol.x[2])}), P, N)
            if newp.rms <= c.primitive.rms + 0.5 * tol and abs(newp.params["radius"] - q["radius"]) < tol:
                log.append({"op": "tangent", "region": c.id, "planes": [p.id for p, _, _ in tang],
                            "radius_old": float(q["radius"]), "radius_new": float(newp.params["radius"])})
                c.primitive = newp
    # ---- 6. rounding of dimensions (diameters; plane offsets relative to datum planes)
    if do_round:
        datums = {}
        for c in clusters:
            ps = [r for r in c[2] if r.primitive.kind == "plane"]
            if ps:
                datums[id(c)] = max(ps, key=lambda r: r.area)
        for c in clusters:
            e = c[0]
            dat = datums.get(id(c))
            done = set()
            for r in c[2]:
                pr = r.primitive
                if pr.kind == "plane" and dat is not None and r is not dat and id(pr) not in done \
                        and pr is not dat.primitive:
                    done.add(id(pr))
                    d0 = dat.primitive.params["point"] @ e
                    d = pr.params["point"] @ e
                    nd, gsz = _round(d - d0, round_tol)
                    if gsz is not None and nd != d - d0:
                        pr.params["point"] = pr.params["point"] + (nd - (d - d0)) * e
                        log.append({"op": "round_offset", "region": r.id, "datum": dat.id,
                                    "raw": float(d - d0), "snapped": nd})
            # cylinders: diameter, and axis position relative to datums of perpendicular clusters
            seen = set()
            for r in c[2]:
                pr = r.primitive
                if pr.kind != "cylinder" or id(pr) in seen:
                    continue
                seen.add(id(pr))
                tangent_locked = any(l.get("op") == "tangent" and l.get("region") == r.id for l in log)
                d_raw = 2 * pr.params["radius"]
                nd, gsz = _round(d_raw, 2 * round_tol)
                if gsz is not None and nd != d_raw:
                    pr.params["radius"] = nd / 2
                    log.append({"op": "round_diameter", "region": r.id, "raw": d_raw, "snapped": nd})
                if tangent_locked:
                    continue  # position governed by tangency
                for c2 in clusters:
                    if c2 is c or id(c2) not in datums or abs(c2[0] @ e) > 1e-6:
                        continue
                    e2 = c2[0]
                    d0 = datums[id(c2)].primitive.params["point"] @ e2
                    x = pr.params["point"] @ e2 - d0
                    nx, gsz = _round(x, round_tol)
                    if gsz is not None and nx != x:
                        pr.params["point"] = pr.params["point"] + (nx - x) * e2
                        log.append({"op": "round_axis_position", "region": r.id,
                                    "datum": datums[id(c2)].id, "raw": float(x), "snapped": nx})
    if do_round:
        seen = set()
        for r in regions:
            pr = r.primitive
            if pr.kind != "torus" or id(pr) in seen:
                continue
            seen.add(id(pr))
            for key in ("minor_radius", "major_radius"):
                nv, gsz = _round(pr.params[key], round_tol)
                if gsz is not None and nv != pr.params[key]:
                    log.append({"op": "round_" + key, "region": r.id, "raw": float(pr.params[key]), "snapped": nv})
                    pr.params[key] = nv
    # tangency after rounding: re-enforce plane-tangent fillets with rounded radius
    for l in [l for l in log if l.get("op") == "tangent"]:
        r = next(x for x in regions if x.id == l["region"])
        q = r.primitive.params
        for pid in l["planes"]:
            p = next(x for x in regions if x.id == pid).primitive.params
            n = p["normal"]
            dist = (q["point"] - p["point"]) @ n
            q["point"] = q["point"] + (np.sign(dist) * q["radius"] - dist) * n
    from .tangency import enforce, region_pairs
    enforce(regions, region_pairs(g, regions), tol, log, pts={k: v[0] for k, v in pts.items()},
            do_round=do_round, round_tol=round_tol)
    return {"frame": [e.tolist() for e in frame],
            "direction_clusters": [{"dir": c[0].tolist(), "regions": [r.id for r in c[2]]} for c in clusters],
            "log": log}


def _adjacent(g, a, b):
    fa = np.zeros(len(g.Fc), bool)
    fa[a.faces] = True
    fb = np.zeros(len(g.Fc), bool)
    fb[b.faces] = True
    x, y = g.adj[:, 0], g.adj[:, 1]
    return bool(np.any((fa[x] & fb[y]) | (fa[y] & fb[x])))
