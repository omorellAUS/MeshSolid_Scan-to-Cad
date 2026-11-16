"""Tangent (G1) contacts between fitted primitives: exact enforcement after snapping, and the
analytic contact curves (lines / circles) used as trimming tools, because Boolean splitting by a
tangent surface does not produce the contact edge."""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from . import fit as F


def _u(v):
    return v / np.linalg.norm(v)


def _perp(d, a):
    return d - (d @ a) * a


def region_pairs(g, regions):
    lab = -np.ones(len(g.Fc), int)
    for r in regions:
        lab[r.faces] = r.id
    a, b = lab[g.adj[:, 0]], lab[g.adj[:, 1]]
    m = (a >= 0) & (b >= 0) & (a != b)
    pairs = set()
    for x, y in zip(a[m], b[m]):
        pairs.add((min(x, y), max(x, y)))
    return pairs


def contact(pa, pb, tol, ang=1e-3):
    """Return (kind, data, gap) if pa and pb touch tangentially within tol, else None.
    Order-independent; data describes the contact curve."""
    ka, kb = pa.kind, pb.kind
    if ka > kb:   # canonical order by name
        pa, pb, ka, kb = pb, pa, kb, ka
    qa, qb = pa.params, pb.params
    if (ka, kb) == ("cone", "plane"):
        # a plane is tangent to a cone along a generator iff it passes through the apex and its
        # normal is perpendicular to that generator (|n.a| = sin(half_angle))
        a, n, al = qa["axis"], qb["normal"], qa["half_angle"]
        if abs(abs(a @ n) - np.sin(al)) > max(ang, np.radians(0.5)):
            return None
        gap = abs((qa["apex"] - qb["point"]) @ n)
        if gap > tol:
            return None
        g = a - (a @ n) * n
        g = g / np.linalg.norm(g)
        return "line", {"point": qa["apex"], "dir": g}, gap
    if (ka, kb) == ("cylinder", "plane"):
        a, n = qa["axis"], qb["normal"]
        if abs(a @ n) > ang:
            return None
        dist = (qa["point"] - qb["point"]) @ n
        gap = abs(abs(dist) - qa["radius"])
        if gap > tol:
            return None
        return "line", {"point": qa["point"] - dist * n, "dir": a}, gap
    if (ka, kb) == ("cylinder", "cylinder"):
        a1, a2 = qa["axis"], qb["axis"]
        if abs(abs(a1 @ a2) - 1) > ang:
            return None
        d = _perp(qb["point"] - qa["point"], a1)
        dd = np.linalg.norm(d)
        if dd < 1e-9:
            return None
        r1, r2 = qa["radius"], qb["radius"]
        for s, tgt in ((1, r1 + r2), (-1, abs(r1 - r2))):
            gap = abs(dd - tgt)
            if gap <= tol:
                if s == 1:
                    pt = qa["point"] + d / dd * r1
                else:   # internal: contact on the far side of the smaller one
                    pt = qa["point"] + (d / dd * r1 if r1 > r2 else -d / dd * r1)
                return "line", {"point": pt, "dir": a1}, gap
        return None
    if (ka, kb) == ("plane", "torus"):
        n, a = qa["normal"], qb["axis"]
        if abs(abs(n @ a) - 1) > ang:
            return None
        h = (qb["center"] - qa["point"]) @ n
        gap = abs(abs(h) - qb["minor_radius"])
        if gap > tol:
            return None
        return "circle", {"center": qb["center"] - h * n, "axis": a, "radius": qb["major_radius"]}, gap
    if (ka, kb) == ("cylinder", "torus"):
        a1, a2 = qa["axis"], qb["axis"]
        if abs(abs(a1 @ a2) - 1) > ang:
            return None
        if np.linalg.norm(_perp(qb["center"] - qa["point"], a1)) > tol:
            return None
        R, r, Rc = qb["major_radius"], qb["minor_radius"], qa["radius"]
        gap = min(abs(R + r - Rc), abs(R - r - Rc))
        if gap > tol:
            return None
        c = qa["point"] + ((qb["center"] - qa["point"]) @ a1) * a1
        return "circle", {"center": c, "axis": a1, "radius": Rc}, gap
    if (ka, kb) == ("cone", "torus"):
        a = qa["axis"]
        if abs(abs(a @ qb["axis"]) - 1) > ang:
            return None
        if np.linalg.norm(_perp(qb["center"] - qa["apex"], a)) > tol:
            return None
        # meridian half-plane: (rho, h) with h along the cone axis from its apex
        al = qa["half_angle"]
        hc = (qb["center"] - qa["apex"]) @ a
        C = np.array([qb["major_radius"], hc])
        u = np.array([np.sin(al), np.cos(al)])
        t = C @ u
        Fp = t * u
        gap = abs(np.linalg.norm(C - Fp) - qb["minor_radius"])
        if gap > tol or t <= 0:
            return None
        return "circle", {"center": qa["apex"] + Fp[1] * a, "axis": a, "radius": float(Fp[0])}, gap
    if (ka, kb) == ("sphere", "torus"):
        a = qb["axis"]
        d = qb["center"] - qa["center"]
        if np.linalg.norm(_perp(d, a)) > tol:
            return None
        C = np.array([qb["major_radius"], d @ a])
        dist = np.linalg.norm(C)
        Rs, r = qa["radius"], qb["minor_radius"]
        gap = min(abs(dist - (Rs + r)), abs(dist - abs(Rs - r)))
        if gap > tol or dist < 1e-9:
            return None
        Fp = C / dist * Rs
        return "circle", {"center": qa["center"] + Fp[1] * a, "axis": a, "radius": float(abs(Fp[0]))}, gap
    return None


def enforce(regions, pairs, tol, log, pts=None, do_round=True, round_tol=0.03):
    """Make near-tangent contacts exact. Blends (cylinders, tori) move; planes and larger
    surfaces stay. Returns number of adjusted primitives."""
    by_id = {r.id: r for r in regions}
    nbrs = {}
    for a, b in pairs:
        nbrs.setdefault(a, set()).add(b)
        nbrs.setdefault(b, set()).add(a)
    done = set()
    n_adj = 0
    # smaller-radius cylinders first (fillets), then tori
    cyls = sorted([r for r in regions if r.primitive.kind == "cylinder"], key=lambda r: r.primitive.params["radius"])
    for r in cyls:
        p = r.primitive
        if id(p) in done:
            continue
        q = p.params
        cons = []
        for j in nbrs.get(r.id, ()):
            o = by_id[j].primitive
            if o is p:
                continue
            if o.kind == "plane":
                n = o.params["normal"]
                if abs(n @ q["axis"]) < 0.02:
                    dist = (q["point"] - o.params["point"]) @ n
                    if abs(abs(dist) - q["radius"]) < tol:
                        cons.append(("plane", o, np.sign(dist)))
            elif o.kind == "cylinder" and o.params["radius"] > q["radius"]:
                if abs(abs(o.params["axis"] @ q["axis"]) - 1) < 2e-4:
                    d = np.linalg.norm(_perp(q["point"] - o.params["point"], o.params["axis"]))
                    R2 = o.params["radius"]
                    if abs(d - (R2 - q["radius"])) < tol:
                        cons.append(("cyl_in", o, R2 - q["radius"]))
                    elif abs(d - (R2 + q["radius"])) < tol:
                        cons.append(("cyl_out", o, R2 + q["radius"]))
        if not cons:
            continue
        planes = [c for c in cons if c[0] == "plane"]
        if len(planes) >= 2:   # axis must be parallel to both planes
            n1, n2 = planes[0][1].params["normal"], planes[1][1].params["normal"]
            ax = np.cross(n1, n2)
            if np.linalg.norm(ax) > 1e-3:
                ax = _u(ax) * np.sign(ax @ q["axis"] or 1)
                q["axis"] = ax
        u, v = F._perp_basis(q["axis"])
        p0 = q["point"].copy()

        def place(R):
            def res(x):
                pt = p0 + x[0] * u + x[1] * v
                out = []
                for kind, o, s_ in cons:
                    if kind == "plane":
                        out.append((pt - o.params["point"]) @ o.params["normal"] - s_ * R)
                    elif kind == "cyl_in":
                        out.append(np.linalg.norm(_perp(pt - o.params["point"], o.params["axis"]))
                                   - (o.params["radius"] - R))
                    else:
                        out.append(np.linalg.norm(_perp(pt - o.params["point"], o.params["axis"]))
                                   - (o.params["radius"] + R))
                return np.r_[1e3 * np.array(out), 1e-3 * x]
            sol = least_squares(res, [0.0, 0.0], xtol=1e-15, ftol=1e-15, gtol=1e-15)
            return p0 + sol.x[0] * u + sol.x[1] * v, float(np.abs(res(sol.x)[:-2]).max() / 1e3)

        R0 = q["radius"]
        if pts is not None and len(cons) >= 2 and r.id in pts:
            P = pts[r.id]

            def cost(R):
                pt, _ = place(R)
                c = F.Primitive("cylinder", {"point": pt, "axis": q["axis"], "radius": R})
                return float(np.mean(F.signed_distance(c, P) ** 2))
            from scipy.optimize import minimize_scalar
            opt = minimize_scalar(cost, bounds=(0.7 * R0, 1.3 * R0), method="bounded",
                                  options={"xatol": 1e-6})
            R1 = float(opt.x)
            if do_round:
                Rr = round(R1 * 2 / 0.5) * 0.5 / 2   # diameter on a 0.5 mm grid
                if abs(Rr - R1) <= round_tol and cost(Rr) <= 1.5 * cost(R1) + 1e-12:
                    R1 = Rr
            q["radius"] = R1
        q["point"], resid = place(q["radius"])
        log.append({"op": "tangent_exact", "region": r.id, "with": [o_id for o_id in nbrs[r.id]
                    if any(c[1] is by_id[o_id].primitive for c in cons)], "residual": resid,
                    "radius_before": float(R0), "radius_after": float(q["radius"])})
        done.add(id(p))
        n_adj += 1
    for r in [r for r in regions if r.primitive.kind == "torus"]:
        p = r.primitive
        if id(p) in done:
            continue
        q = p.params
        changed = []
        tc = []   # ("cyl", Rc, s) : R = Rc - s*r ; ("plane", point, n, s): centre = plane + s*r*n
        for j in nbrs.get(r.id, ()):
            o = by_id[j].primitive
            if o.kind == "cylinder" and abs(abs(o.params["axis"] @ q["axis"]) - 1) < 2e-4 and \
                    np.linalg.norm(_perp(q["center"] - o.params["point"], o.params["axis"])) < tol:
                Rc, r_ = o.params["radius"], q["minor_radius"]
                for s_ in (1, -1):
                    if abs(q["major_radius"] + s_ * r_ - Rc) < tol:
                        ax = o.params["axis"]
                        q["axis"] = ax * np.sign(ax @ q["axis"])
                        q["center"] = o.params["point"] + ((q["center"] - o.params["point"]) @ ax) * ax
                        tc.append(("cyl", Rc, s_))
                        changed.append(j)
                        break
        for j in nbrs.get(r.id, ()):
            o = by_id[j].primitive
            if o.kind == "plane" and abs(abs(o.params["normal"] @ q["axis"]) - 1) < 2e-4:
                n = o.params["normal"]
                h = (q["center"] - o.params["point"]) @ n
                if abs(abs(h) - q["minor_radius"]) < tol:
                    tc.append(("plane", o.params["point"], n, np.sign(h)))
                    changed.append(j)
        c0 = q["center"].copy()

        def apply(rr):
            c, R = c0.copy(), q["major_radius"]
            for t in tc:
                if t[0] == "cyl":
                    R = t[1] - t[2] * rr
                else:
                    _, pp, n, s_ = t
                    h = (c - pp) @ n
                    c = c + (s_ * rr - h) * n
            return c, R
        if tc:
            r0 = q["minor_radius"]
            if pts is not None and r.id in pts and len(tc) >= 2:
                P = pts[r.id]

                def cost(rr):
                    c, R = apply(rr)
                    t = F.Primitive("torus", {"center": c, "axis": q["axis"], "major_radius": R, "minor_radius": rr})
                    return float(np.mean(F.signed_distance(t, P) ** 2))
                from scipy.optimize import minimize_scalar
                rr = float(minimize_scalar(cost, bounds=(0.6 * r0, 1.4 * r0), method="bounded",
                                           options={"xatol": 1e-6}).x)
                if do_round:
                    rq = round(rr / 0.5) * 0.5
                    if rq > 0 and abs(rq - rr) <= round_tol and cost(rq) <= 1.5 * cost(rr) + 1e-12:
                        rr = rq
                q["minor_radius"] = rr
            q["center"], q["major_radius"] = apply(q["minor_radius"])
            log.append({"op": "tangent_exact", "region": r.id, "with": changed,
                        "minor_radius_before": float(r0), "minor_radius_after": float(q["minor_radius"])})
            done.add(id(p))
            n_adj += 1
            continue
        if changed:
            log.append({"op": "tangent_exact", "region": r.id, "with": changed})
            done.add(id(p))
            n_adj += 1
    return n_adj
