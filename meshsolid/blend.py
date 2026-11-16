"""Constant-radius rolling-ball blends that are not analytic primitives.

A fillet of radius r between a plane and a cylinder whose axis is oblique to the plane is a
pipe (canal) surface: the ball centre runs along the intersection of the plane offset by r and
the cylinder offset by r - an exact ellipse.  Such regions are first approximated by a torus
during segmentation; here they are recognised from their two support surfaces and replaced by
kind "pipe" (params: ellipse spine + radius + support ids), giving exact contact curves:
  * on the plane:    the spine ellipse translated by -s_p*r*n (an ellipse),
  * on the cylinder: the spine pushed radially by r (a smooth curve, interpolated).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

from . import fit as F


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def ellipse_spine(pl, cy, s_p, s_c, r):
    """Spine ellipse of a radius-r ball touching plane pl (offset side s_p) and cylinder cy
    (s_c=-1: ball inside the cylinder, +1 outside). Returns dict or None."""
    n = _unit(pl["normal"])
    a = _unit(cy["axis"])
    cs = float(n @ a)
    if abs(cs) < 0.05 or abs(cs) > 0.999:
        return None
    Rs = cy["radius"] + s_c * r
    if Rs <= 0:
        return None
    d = n @ pl["point"] + s_p * r
    t = (d - n @ cy["point"]) / cs
    c = cy["point"] + t * a
    u = _unit(a - cs * n)            # major direction (in the plane)
    v = np.cross(n, u)               # minor direction (perpendicular to the axis)
    return {"center": c, "normal": n, "u": u, "v": v, "A": Rs / abs(cs), "B": Rs}


def ellipse_pts(e, t):
    t = np.asarray(t, float)
    return e["center"] + np.outer(e["A"] * np.cos(t), e["u"]) + np.outer(e["B"] * np.sin(t), e["v"])


def ellipse_param(e, P, iters=8):
    """Parameter of the closest spine point for each point P (Newton from the planar angle)."""
    d = P - e["center"]
    A, B = e["A"], e["B"]
    t = np.arctan2((d @ e["v"]) / B, (d @ e["u"]) / A)
    du, dv = d @ e["u"], d @ e["v"]
    for _ in range(iters):
        c, s = np.cos(t), np.sin(t)
        # f = (E - P).E' in the ellipse plane coordinates (normal offset is constant in t)
        f = (A * c - du) * (-A * s) + (B * s - dv) * (B * c)
        fp = (A * s) ** 2 + (B * c) ** 2 + (A * c - du) * (-A * c) + (B * s - dv) * (-B * s)
        t = t - f / np.where(np.abs(fp) > 1e-12, fp, 1e-12)
    return t


def pipe_distance(q, P):
    e = q
    t = ellipse_param(e, P)
    S = ellipse_pts(e, t)
    return np.linalg.norm(P - S, axis=1) - q["radius"], S, t


def pipe_normal(q, P):
    _, S, _ = pipe_distance(q, P)
    d = P - S
    return d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)


def make_pipe(pl, cy, s_p, s_c, r, ids):
    e = ellipse_spine(pl, cy, s_p, s_c, r)
    if e is None:
        return None
    e = dict(e)
    e.update(radius=float(r), s_plane=float(s_p), s_cyl=float(s_c),
             support_plane=int(ids[0]), support_cyl=int(ids[1]))
    return e


def fit_pipe(P, N, pl, cy, ids, r0, tol):
    """Best rolling-ball blend for points P between plane pl and cylinder cy; None if no fit."""
    best = None
    for s_p in (1.0, -1.0):
        for s_c in (1.0, -1.0):
            def cost(r):
                q = make_pipe(pl, cy, s_p, s_c, r, ids)
                if q is None:
                    return 1e9
                d, _, _ = pipe_distance(q, P)
                return float(np.sqrt(np.mean(d * d)))
            lo, hi = 0.3 * r0, 2.5 * r0
            grid = np.linspace(lo, hi, 23)
            c = [cost(r) for r in grid]
            k = int(np.argmin(c))
            res = minimize_scalar(cost, bounds=(grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)]),
                                  method="bounded", options={"xatol": 1e-7})
            if best is None or res.fun < best[0]:
                best = (res.fun, s_p, s_c, float(res.x))
    if best is None or best[0] > tol:
        return None
    q = make_pipe(pl, cy, best[1], best[2], best[3], ids)
    p = F.Primitive("pipe", q)
    F._finish(p, P, N)
    return p


def recognise_blends(g, regions, pairs, tol, log, do_round=True, round_tol=0.03, region_points=None):
    """Replace approximate fits (torus/cylinder/...) of plane-cylinder ball blends by exact pipes."""
    by_id = {r.id: r for r in regions}
    nbrs = {}
    for a, b in pairs:
        nbrs.setdefault(a, set()).add(b)
        nbrs.setdefault(b, set()).add(a)
    n = 0
    for r in regions:
        k = r.primitive.kind
        if k not in ("torus", "cylinder", "cone", "sphere"):
            continue
        nb = [by_id[j] for j in nbrs.get(r.id, ()) if j in by_id]
        planes = [o for o in nb if o.primitive.kind == "plane"]
        cyls = [o for o in nb if o.primitive.kind == "cylinder" and o.primitive.params["radius"] > 3 * _r0(r.primitive)]
        if not planes or not cyls:
            continue
        P, N = region_points(g, r.faces)
        cur = r.primitive
        best = None
        for pl in planes:
            for cy in cyls:
                cs = abs(pl.primitive.params["normal"] @ cy.primitive.params["axis"])
                if cs < 0.05 or cs > 0.999:
                    continue  # parallel/perpendicular supports: the blend is a cylinder/torus
                p = fit_pipe(P, N, pl.primitive.params, cy.primitive.params, (pl.id, cy.id), _r0(cur), tol)
                if p is not None and (best is None or p.rms < best.rms):
                    best = p
        if best is None or best.max_err > tol or not (best.rms < 0.7 * cur.rms or cur.max_err > tol):
            continue
        r_fit = best.params["radius"]
        if do_round:
            rr = round(r_fit * 2) / 2
            if rr > 0 and abs(rr - r_fit) <= round_tol:
                q = make_pipe(by_id[best.params["support_plane"]].primitive.params,
                              by_id[best.params["support_cyl"]].primitive.params,
                              best.params["s_plane"], best.params["s_cyl"], rr,
                              (best.params["support_plane"], best.params["support_cyl"]))
                pr = F.Primitive("pipe", q)
                F._finish(pr, P, N)
                if pr.max_err <= tol:
                    best = pr
        log.append({"op": "blend_pipe", "region": r.id, "was": cur.kind, "was_rms": cur.rms,
                    "rms": best.rms, "max_err": best.max_err, "radius_fit": r_fit,
                    "radius": best.params["radius"],
                    "supports": [best.params["support_plane"], best.params["support_cyl"]]})
        r.primitive = best
        n += 1
    return n


def _r0(p):
    q = p.params
    if p.kind == "torus":
        return float(q["minor_radius"])
    if p.kind in ("cylinder", "sphere"):
        return float(q["radius"])
    return 1.0
