"""Coaxial model selection: in CAD, a chamfer or blend around a cylinder is a surface of revolution
coaxial with it (a chamfer around a vertical fillet is a cone on the fillet axis). On a small noisy patch a
free fit cannot tell a cone piece from an oblique cylinder, so we test the coaxial hypothesis explicitly:
with the axis fixed by the neighbouring cylinder the cone fit is linear (r = c0 + c1*h), and the slope can be
seeded/snapped from an adjacent chamfer plane's angle. The coaxial cone replaces the current fit only if it
explains the points at least as well (distance rms and normal error)."""
from __future__ import annotations

import numpy as np

from . import fit as F
from .segment import region_points


def _adjacent(g, labels):
    a, b = labels[g.adj[:, 0]], labels[g.adj[:, 1]]
    nb = {}
    for x, y in zip(a, b):
        if x != y and x >= 0 and y >= 0:
            nb.setdefault(int(x), set()).add(int(y))
            nb.setdefault(int(y), set()).add(int(x))
    return nb


def _coaxial_cone(P, N, axis, p0, slope=None):
    d = P - p0
    h = d @ axis
    r = np.linalg.norm(d - np.outer(h, axis), axis=1)
    if slope is None:
        A = np.c_[np.ones_like(h), h]
        c0, c1 = np.linalg.lstsq(A, r, rcond=None)[0]
    else:
        c1 = slope
        c0 = float(np.mean(r - c1 * h))
    if abs(c1) < 1e-9:
        return None
    alpha = float(np.arctan(abs(c1)))
    if not np.radians(2.0) < alpha < np.radians(88.0):
        return None
    ax = axis if c1 > 0 else -axis
    apex = p0 + axis * (-c0 / c1)
    return F._finish(F.Primitive("cone", {"apex": apex, "axis": ax, "half_angle": alpha}), P, N)


def coaxial_refit(g, labels, regions, tol, max_faces_frac=0.05):
    """Returns a log of replacements. Only small cylinder/cone regions are candidates."""
    by_id = {r.id: r for r in regions}
    nb = _adjacent(g, labels)
    total = sum(len(r.faces) for r in regions)
    log = []
    for r in regions:
        p = r.primitive
        if p is None or p.kind not in ("cylinder", "cone") or len(r.faces) > max_faces_frac * total:
            continue
        P, N = region_points(g, r.faces)
        cur = F._finish(F.Primitive(p.kind, dict(p.params)), P, N)
        best = None
        for j in nb.get(r.id, ()):
            q = by_id[j].primitive
            if q is None or q.kind != "cylinder" or j == r.id:
                continue
            axis = np.asarray(q.params["axis"], float)
            axis = axis / np.linalg.norm(axis)
            p0 = np.asarray(q.params["point"], float)
            if p.kind == "cone" and abs(np.asarray(p.params["axis"]) @ axis) > np.cos(np.radians(0.5)):
                continue  # already coaxial
            cand = _coaxial_cone(P, N, axis, p0)
            if cand is None:
                continue
            # seed/snap the half angle from an adjacent plane (tangent-continuous chamfer face)
            for k in nb.get(r.id, ()):
                pk = by_id[k].primitive
                if pk is None or pk.kind != "plane":
                    continue
                beta = np.degrees(np.arccos(np.clip(abs(np.asarray(pk.params["normal"]) @ axis), 0, 1)))
                a_pl = 90.0 - beta   # angle between the plane and the axis
                if abs(np.degrees(cand.params["half_angle"]) - a_pl) < 1.5:
                    s = np.tan(np.radians(a_pl)) * np.sign((cand.params["axis"] @ axis))
                    c2 = _coaxial_cone(P, N, axis, p0, slope=s)
                    if c2 is not None and c2.rms <= cand.rms * 1.15 + 0.1 * tol:
                        c2.params["half_angle_snapped_to_region"] = float(k)
                        cand = c2
                    break
            if best is None or cand.rms < best[0].rms:
                best = (cand, j)
        if best is None:
            continue
        cand, j = best
        # the coaxial cone has 2 free parameters against 5 for a free cylinder/cone, so near-equal
        # residuals favour it (BIC); on noisy patches of ~20 faces the normal error is noise-dominated
        better = cand.rms <= cur.rms and cand.normal_err_deg <= cur.normal_err_deg + 2.5
        similar = (cand.rms <= max(1.15 * cur.rms, cur.rms + 0.25 * tol)
                   and cand.normal_err_deg <= max(cur.normal_err_deg + 0.5, 1.2 * cur.normal_err_deg))
        ok = better or similar
        log.append({"region": r.id, "coaxial_with": j, "accepted": bool(ok),
                    "rms_current": cur.rms, "rms_coaxial_cone": cand.rms,
                    "normal_err_current": cur.normal_err_deg, "normal_err_coaxial": cand.normal_err_deg,
                    "half_angle_deg": float(np.degrees(cand.params["half_angle"]))})
        if ok:
            cand.params.pop("half_angle_snapped_to_region", None)
            r.primitive = cand
            r.flags = list(r.flags) + [f"coaxial_cone_with_{j}"]
    return log


def realign_coaxial(regions):
    """After snapping/tangency moved cylinders, put every coaxial cone back exactly on its partner's
    axis (apex projected onto the axis, axis direction copied), so the cone-cylinder intersection is an
    exact circle instead of an approximated BSpline edge."""
    by_id = {r.id: r for r in regions}
    log = []
    for r in regions:
        tag = next((f for f in r.flags if f.startswith("coaxial_cone_with_")), None)
        if tag is None or r.primitive is None or r.primitive.kind != "cone":
            continue
        j = int(tag.rsplit("_", 1)[1])
        if j not in by_id:          # partner region merged away / released by an earlier stage
            log.append({"region": r.id, "partner": j, "skipped": "partner region no longer exists"})
            continue
        q = by_id[j].primitive
        if q is None or q.kind != "cylinder":
            continue
        a = np.asarray(q.params["axis"], float)
        a = a / np.linalg.norm(a)
        p0 = np.asarray(q.params["point"], float)
        c = r.primitive.params
        old_apex = np.asarray(c["apex"], float)
        apex = p0 + ((old_apex - p0) @ a) * a
        c["axis"] = a if np.asarray(c["axis"]) @ a >= 0 else -a
        c["apex"] = apex
        log.append({"region": r.id, "with": j, "apex_moved": float(np.linalg.norm(apex - old_apex))})
    return log
