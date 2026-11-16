"""Turned-part builder: when (nearly) all surface area belongs to surfaces of revolution about one axis
(planes normal to it, coaxial cylinders, cones, tori, spheres centred on it), the solid is the revolve of a
closed 2D profile in the (r, z) half-plane. Each region becomes one or more profile segments (its data
intervals along its 2D curve); segments are chained end to end and joined at the intersection of their
curves. Where a surface was not scanned at all (a gap between two chained segments with no nearby
intersection) a connector is synthesised perpendicular to the neighbours (design intent of a turned
step) and reported as `synthesised`. Non-coaxial holes (concave cylinders) are then cut. Nothing is
reported as built unless the OCC solid is valid; acceptance is the pipeline's normal deviation gate."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

COS1 = math.cos(math.radians(1.0))


@dataclass
class RevolveResult:
    ok: bool = False
    method: str = "revolve_profile"
    shape: object = None
    reason: str = ""
    axis_point: list = None
    axis_dir: list = None
    coaxial_area_frac: float = 0.0
    profile: list = field(default_factory=list)       # [{"region", "kind", "from", "to", "synthesised"}]
    synthesised: list = field(default_factory=list)
    dropped_segments: list = field(default_factory=list)
    holes_cut: list = field(default_factory=list)
    unmodelled_regions: list = field(default_factory=list)
    profile_refits: list = field(default_factory=list)
    fill_artifacts: list = field(default_factory=list)   # hole caps that cannot be real surface
    fill_faces: object = None                            # mesh face indices of those caps (not in JSON)
    bosses: list = field(default_factory=list)
    gear_zone_regions: list = field(default_factory=list)  # scan regions modelled by the gear, not profiled
    gear: dict = None                                       # tooth-space cut result
    screw: dict = None                                      # helical rotor sweep result
    brepcheck_valid: bool = False
    n_solids: int = 0
    n_faces: int = 0
    volume: float = 0.0
    mesh_volume: float = 0.0
    seconds: float = 0.0

    def to_json(self):
        return {k: v for k, v in self.__dict__.items() if k not in ("shape", "fill_faces")}


# ------------------------------------------------------------------ 2D curves in (r, z)
def _curve2d(prim, a0, u, tol, coax_cos=None, off_lim=None):
    """Return ('line', q, d) or ('circle', c, rho) in (r, z) coordinates, or None if not coaxial.

    coax_cos / off_lim: optional looser gates (gear/screw blanks force the gear axis, which can
    disagree with hub-primitive axes by up to ~15 deg on scans — p01 hub tori were dropped).
    """
    q, k = prim.params, prim.kind
    lt = float(off_lim) if off_lim is not None else max(5 * tol, 0.2)
    cos_need = float(coax_cos) if coax_cos is not None else COS1

    def z_of(p):
        return float((np.asarray(p, float) - a0) @ u)

    def off_axis(p):
        d = np.asarray(p, float) - a0
        return float(np.linalg.norm(d - (d @ u) * u))

    def par(ax):
        ax = np.asarray(ax, float)
        return abs(ax @ u) / np.linalg.norm(ax) > cos_need
    if k == "plane":
        if not par(q["normal"]):
            return None
        return ("line", np.array([0.0, z_of(q["point"])]), np.array([1.0, 0.0]))
    if k == "cylinder":
        if not par(q["axis"]) or off_axis(q["point"]) > lt:
            return None
        return ("line", np.array([float(q["radius"]), 0.0]), np.array([0.0, 1.0]))
    if k == "cone":
        if not par(q["axis"]) or off_axis(q["apex"]) > lt:
            return None
        s = 1.0 if np.asarray(q["axis"], float) @ u > 0 else -1.0
        a = float(q["half_angle"])
        d = np.array([math.sin(a), s * math.cos(a)])      # r grows with h along the cone axis
        return ("line", np.array([0.0, z_of(q["apex"])]), d / np.linalg.norm(d))
    if k == "torus":
        if not par(q["axis"]) or off_axis(q["center"]) > lt:
            return None
        return ("circle", np.array([float(q["major_radius"]), z_of(q["center"])]), float(q["minor_radius"]))
    if k == "sphere":
        if off_axis(q["center"]) > lt:
            return None
        return ("circle", np.array([0.0, z_of(q["center"])]), float(q["radius"]))
    return None


def _param(c, X):
    if c[0] == "line":
        return (X - c[1]) @ c[2]
    d = X - c[1]
    return np.arctan2(d[:, 1], d[:, 0])


def _point(c, t):
    if c[0] == "line":
        return c[1] + t * c[2]
    return c[1] + c[2] * np.array([math.cos(t), math.sin(t)])


def _intersections(A, B, tol):
    """Intersection points of two 2D curves; near-tangent line/circle and circle/circle give the touch point."""
    out = []
    if A[0] == "line" and B[0] == "line":
        M = np.array([A[2], -B[2]]).T
        if abs(np.linalg.det(M)) < 1e-6:
            return []
        s = np.linalg.solve(M, B[1] - A[1])
        return [A[1] + s[0] * A[2]]
    if A[0] == "circle" and B[0] == "line":
        A, B = B, A
    if A[0] == "line" and B[0] == "circle":
        q, d = A[1], A[2]
        c, rho = B[1], B[2]
        f = q - c
        t0 = -(f @ d)
        foot = q + t0 * d
        h = float(np.linalg.norm(foot - c))
        if h > rho + 5 * tol:
            return []
        if h >= rho:
            return [foot]
        w = math.sqrt(rho * rho - h * h)
        return [foot + w * d, foot - w * d]
    c1, r1, c2, r2 = A[1], A[2], B[1], B[2]
    dd = float(np.linalg.norm(c2 - c1))
    if dd < 1e-9:
        return []
    if dd > r1 + r2 + 5 * tol or dd < abs(r1 - r2) - 5 * tol:
        return []
    a = (r1 * r1 - r2 * r2 + dd * dd) / (2 * dd)
    h = math.sqrt(max(r1 * r1 - a * a, 0.0))
    e = (c2 - c1) / dd
    m = c1 + a * e
    n = np.array([-e[1], e[0]])
    out = [m + h * n] if h < 1e-9 else [m + h * n, m - h * n]
    return out


def _segments(curve, X, rid, gap, Xall=None, small_gap=None, lim=None):
    """Split the data of one region along its curve into intervals (separated by > gap).  With Xall
    (all scan vertices in (r, z)), a shorter gap (> small_gap) also splits when the profile visibly
    leaves the curve inside it: scan points projecting into the gap all lie > lim off the curve (e.g.
    the tooth tips between the groove floors of a labyrinth seal, which one coaxial region spans)."""
    t = _param(curve, X)
    if curve[0] == "circle":                 # unwrap around the mean direction
        m = math.atan2(np.mean(np.sin(t)), np.mean(np.cos(t)))
        t = m + np.angle(np.exp(1j * (t - m)))
        g = gap / max(curve[2], 1e-6)
        gs = None if small_gap is None else small_gap / max(curve[2], 1e-6)
    else:
        g = gap
        gs = small_gap
    o = np.sort(t)
    dif = np.diff(o)
    cuts = np.where(dif > g)[0]
    if Xall is not None and gs is not None and curve[0] == "line":
        cand = np.where((dif > gs) & (dif <= g))[0]
        if len(cand):
            ta = _param(curve, Xall); da = _curve_dist(curve, Xall)
            near_band = da < max(4 * lim, 6.0)         # points of this neighbourhood of the profile (teeth, steps)
            add = []
            for c in cand:
                a, b = o[c], o[c + 1]
                mid = (ta > a + 0.25 * (b - a)) & (ta < b - 0.25 * (b - a)) & near_band
                if mid.sum() >= 5 and np.min(da[mid]) > lim:
                    add.append(c)
            if add:
                cuts = np.unique(np.r_[cuts, add]).astype(int)
    segs = []
    lo = 0
    for hi in list(cuts) + [len(o) - 1]:
        part = o[lo:hi + 1]
        if len(part) >= 8:
            t0, t1 = float(np.percentile(part, 0.5)), float(np.percentile(part, 99.5))
            segs.append({"region": rid, "curve": curve, "t0": t0, "t1": t1, "n": len(part),
                         "E0": _point(curve, t0), "E1": _point(curve, t1), "tm": float(np.median(part))})
        lo = hi + 1
    return segs


def _curve_dist(c, X):
    if c[0] == "line":
        d = X - c[1]
        return np.abs(d @ np.array([-c[2][1], c[2][0]]))
    return np.abs(np.linalg.norm(X - c[1], axis=1) - c[2])


def _suppress_overlaps(segs, tol, res):
    """Several regions fitted to the same physical surface (e.g. a badly scanned groove bottom fitted as a
    cylinder, a sphere and two tori) give overlapping segments that zig-zag. A segment whose sampled span
    lies within max(0.3 mm, 5 tol) of a better-supported segment's curve over that segment's own span
    is dropped."""
    lim = max(0.3, 5 * tol)
    order = sorted(range(len(segs)), key=lambda i: -segs[i]["n"])
    kept = []
    for i in order:
        s = segs[i]
        ts = np.linspace(s["t0"], s["t1"], 25)
        X = np.array([_point(s["curve"], t) for t in ts])
        dup = None
        for k in kept:
            o = segs[k]
            if s.get("fixed") and o.get("fixed"):
                continue
            # a small-radius arc next to a line is a fillet, not a duplicate of the line
            if s["curve"][0] == "circle" and o["curve"][0] == "line" and s["curve"][2] < 20.0:
                continue
            # two lines crossing at a steep angle are different surfaces meeting at a corner (a short
            # tooth tip land next to a tooth flank), however close the short one lies to the other
            if s["curve"][0] == "line" and o["curve"][0] == "line" and abs(float(s["curve"][2] @ o["curve"][2])) < math.cos(math.radians(30)):
                continue
            # o's own span: project X onto o and require it inside o's interval (with margin)
            to = _param(o["curve"], X)
            if o["curve"][0] == "circle":
                to = o["tm"] + np.angle(np.exp(1j * (to - o["tm"])))
            lo, hi = min(o["t0"], o["t1"]), max(o["t0"], o["t1"])
            mg = lim if o["curve"][0] == "line" else lim / max(o["curve"][2], 1e-6)
            inside = (to >= lo - mg) & (to <= hi + mg)
            if inside.mean() > 0.6 and np.median(_curve_dist(o["curve"], X[inside])) < lim:
                dup = o["region"]
                break
        if dup is None:
            kept.append(i)
        else:
            res.dropped_segments.append({"region": s["region"], "why": f"overlaps region {dup}"})
    return [segs[i] for i in sorted(kept)]


def _chain(segs):
    """Greedy end-to-end chaining into one loop, starting from the longest segment.  Step faces made by
    a previous call (tangent to a blend arc) are rebuilt, so they are removed first."""
    segs[:] = [x for x in segs if x.get("tangent_for") is None]
    L = [float(np.linalg.norm(s["E1"] - s["E0"])) for s in segs]
    order = [(int(np.argmax(L)), False)]      # (index, reversed?)
    used = {order[0][0]}
    start = segs[order[0][0]]["E0"]
    cur = segs[order[0][0]]["E1"]
    while True:
        best = None
        for i, s in enumerate(segs):
            if i in used:
                continue
            for rev, e in ((False, s["E0"]), (True, s["E1"])):
                d = float(np.linalg.norm(e - cur))
                if best is None or d < best[0]:
                    best = (d, i, rev)
        dclose = float(np.linalg.norm(start - cur))
        if best is None or (dclose <= best[0] and len(order) >= 3):
            break
        _, i, rev = best
        order.append((i, rev))
        used.add(i)
        s = segs[i]
        cur = s["E0"] if rev else s["E1"]
    # insert left-over segments (fillets the greedy walk stepped over) into the joint they bridge
    used = {o[0] for o in order}
    changed = True
    while changed:
        changed = False
        for i, s in enumerate(segs):
            if i in used:
                continue
            best = None
            for k in range(len(order)):
                a_i, a_rev = order[k]
                b_i, b_rev = order[(k + 1) % len(order)]
                ae = segs[a_i]["E0"] if a_rev else segs[a_i]["E1"]
                bs = segs[b_i]["E1"] if b_rev else segs[b_i]["E0"]
                g = float(np.linalg.norm(ae - bs))
                for rev, (e0, e1) in ((False, (s["E0"], s["E1"])), (True, (s["E1"], s["E0"]))):
                    c = float(np.linalg.norm(ae - e0) + np.linalg.norm(e1 - bs))
                    if c <= g + 1.0 and (best is None or c < best[0]):
                        best = (c, k, rev)
            if best is not None:
                order.insert(best[1] + 1, (i, best[2]))
                used.add(i)
                changed = True
    # a step the neighbouring face's data ran past: a straight segment roughly perpendicular to chain
    # segment A, one end within 1 mm of its intersection with A, inside A's span (>= 1 mm from its ends), heading towards the next
    # segment B.  Typical cause: scan-only material coplanar with a shoulder face (the internal cap of a
    # multi-capture scan) extends that face's region inwards past a short neck cylinder.  The joins then
    # trim A at the step (A x S intersection) and connect S to B.
    for i, s in enumerate(segs):
        if i in used or s["curve"][0] != "line":
            continue
        for k in range(len(order)):
            a_i, a_rev = order[k]
            b_i, b_rev = order[(k + 1) % len(order)]
            A = segs[a_i]
            if A["curve"][0] != "line" or abs(float(np.dot(A["curve"][2], s["curve"][2]))) > 0.2:
                continue
            a0_, ae = (A["E1"], A["E0"]) if a_rev else (A["E0"], A["E1"])
            bs = segs[b_i]["E1"] if b_rev else segs[b_i]["E0"]
            hit = None
            for rev, (e0, e1) in ((False, (s["E0"], s["E1"])), (True, (s["E1"], s["E0"]))):
                Xs = _intersections(A["curve"], s["curve"], 1e-3)
                if not Xs or np.linalg.norm(Xs[0] - e0) > 1.0:
                    continue
                t = float(_param(A["curve"], Xs[0][None])[0]); ta = float(_param(A["curve"], a0_[None])[0])
                tb = float(_param(A["curve"], ae[None])[0])
                if not (min(ta, tb) + 1.0 < t < max(ta, tb) - 1.0):
                    continue
                if np.linalg.norm(e1 - bs) < np.linalg.norm(e0 - bs) and np.linalg.norm(e1 - bs) < np.linalg.norm(e0 - ae) + 2.0:
                    hit = rev
                    break
            if hit is not None:
                order.insert(k + 1, (i, hit)); used.add(i)
                break
    # a blend arc in a turned step (A, B parallel lines, no intersection): the step face is the line
    # perpendicular to A tangent to the arc, the arc then runs into B.  Accepted when the tangent point is
    # within 2 mm of the arc's data end nearest A and 3 mm of A's end, and its other data end within 2 mm of
    # B's curve and 3 mm of B's start.
    for i, s in enumerate(list(segs)):
        if i in used or s["curve"][0] != "circle" or s["curve"][2] > 20.0 or s.get("tangent_for") is not None:
            continue
        cen, rho = s["curve"][1], s["curve"][2]
        for k in range(len(order)):
            a_i, a_rev = order[k]
            b_i, b_rev = order[(k + 1) % len(order)]
            A, B = segs[a_i], segs[b_i]
            if A["curve"][0] != "line" or B["curve"][0] != "line" or abs(float(A["curve"][2] @ B["curve"][2])) < 0.98:
                continue
            ae = A["E0"] if a_rev else A["E1"]
            dA = A["curve"][2]; nA = np.array([-dA[1], dA[0]])
            # tangent lines perpendicular to A: through cen +- rho * dA
            best = None
            for sg in (1.0, -1.0):
                T = cen + sg * rho * dA
                if best is None or np.linalg.norm(T - ae) < np.linalg.norm(best - ae):
                    best = T
            T = best
            ends = (s["E0"], s["E1"])
            near_end = min(ends, key=lambda e: np.linalg.norm(e - ae)); far_end = ends[1] if near_end is ends[0] else ends[0]
            bs = B["E1"] if b_rev else B["E0"]
            if (np.linalg.norm(T - near_end) > 2.0 or np.linalg.norm(T - ae) > 3.0
                    or _curve_dist(B["curve"], far_end[None])[0] > 2.0 or np.linalg.norm(far_end - bs) > 3.0):
                continue
            S0 = ae + ((T - ae) @ dA) * dA        # foot on A's line at the tangent level
            if np.linalg.norm(T - S0) < 2 * 0.05:
                continue
            C = {"region": f"step:{s['region']}", "curve": ("line", S0, (T - S0) / np.linalg.norm(T - S0)),
                 "t0": 0.0, "t1": float(np.linalg.norm(T - S0)), "n": 0, "E0": S0, "E1": T,
                 "tm": 0.5 * float(np.linalg.norm(T - S0)), "fixed": True, "tangent_for": s["region"]}
            segs.append(C)
            order.insert(k + 1, (len(segs) - 1, False))
            order.insert(k + 2, (i, bool(near_end is s["E1"])))
            used.add(i); used.add(len(segs) - 1)
            break
    # a blend arc whose ends lie on the two curves of a joint is inserted there even when one segment's
    # data runs past it (e.g. a face whose region also took scan faces beyond the blend)
    for i, s in enumerate(segs):
        if i in used or s["curve"][0] != "circle" or s["curve"][2] > 20.0:
            continue
        rho = s["curve"][2]
        for k in range(len(order)):
            a_i, _ = order[k]
            b_i, _ = order[(k + 1) % len(order)]
            A, B = segs[a_i]["curve"], segs[b_i]["curve"]
            if A[0] != "line" or B[0] != "line":
                continue
            X = _intersections(A, B, 1e-3)
            if not X or np.linalg.norm(X[0] - 0.5 * (s["E0"] + s["E1"])) > 2 * rho + 1.0:
                continue
            cen = s["curve"][1]
            # tangent (within 0.3 mm) to both curves, centre on the material side of neither ambiguity:
            # orientation from which end is nearer to which curve
            tt = max(0.3, 0.15 * rho)
            tA = abs(_curve_dist(A, cen[None])[0] - rho) <= tt
            tB = abs(_curve_dist(B, cen[None])[0] - rho) <= tt
            # tangent to both, or tangent to one with its free end within 2 mm of the other curve
            nearA = min(_curve_dist(A, s["E0"][None])[0], _curve_dist(A, s["E1"][None])[0]) <= 2.0
            nearB = min(_curve_dist(B, s["E0"][None])[0], _curve_dist(B, s["E1"][None])[0]) <= 2.0
            if not ((tA and tB) or (tA and nearB) or (tB and nearA)):
                continue
            rev = bool(_curve_dist(A, s["E0"][None])[0] > _curve_dist(A, s["E1"][None])[0])
            order.insert(k + 1, (i, rev)); used.add(i)
            if i in used:
                break
    return order


def _join(A, a_end, B, b_start, tol):
    """Vertex (or connector) between curve A leaving at a_end and curve B entering at b_start."""
    gap = float(np.linalg.norm(a_end - b_start))
    cands = _intersections(A, B, tol)
    lim = max(3 * gap, gap + 2.0, 1.0)
    best = None
    for X in cands:
        s = float(np.linalg.norm(X - a_end) + np.linalg.norm(X - b_start))
        if s <= lim and (best is None or s < best[0]):
            best = (s, X)
    if best is not None:
        return [best[1]], None
    # no usable intersection: synthesise a connector perpendicular to A (turned step), through the gap mid
    m = 0.5 * (a_end + b_start)
    if A[0] == "line":
        n = np.array([-A[2][1], A[2][0]])
    else:
        n = (b_start - a_end) / max(gap, 1e-9)
    C = ("line", m, n / np.linalg.norm(n))
    xa = _intersections(A, C, tol)
    xb = _intersections(C, B, tol)
    if not xa or not xb:
        return [a_end, b_start], ("line", a_end, (b_start - a_end) / max(gap, 1e-9))
    pa = min(xa, key=lambda X: np.linalg.norm(X - a_end))
    pb = min(xb, key=lambda X: np.linalg.norm(X - b_start))
    return [pa, pb], C


# ------------------------------------------------------------------ OCC
def _to3(p2, a0, u, e1):
    from OCP.gp import gp_Pnt
    P = a0 + p2[1] * u + p2[0] * e1
    return gp_Pnt(*map(float, P))



def _feature_area_credit(regions, coaxial_ids, tot, flat_frac_cap=0.25):
    """Cut features (non-coaxial cylinders/cones/tori/spheres) and a minority of side flats
    (planes not already accepted as end caps, capped at flat_frac_cap of tot) are dropped from
    the coaxial-area denominator.  They are modelled later as holes / milled flats, not as
    evidence that the body is not turned."""
    cut_area = flat_area = 0.0
    for r in regions:
        if r.id in coaxial_ids:
            continue
        kind = getattr(r.primitive, "kind", None) if r.primitive is not None else None
        if kind in ("cylinder", "cone", "torus", "sphere"):
            cut_area += r.area
        elif kind == "plane":
            flat_area += r.area
    flat_credit = min(flat_area, flat_frac_cap * tot)
    tot_adj = max(tot - cut_area - flat_credit, 0.0)
    return cut_area, flat_area, flat_credit, tot_adj


def build_revolve(mesh, regions, labels, tol, min_frac=0.85, gear=None, screw=None):
    """gear: optional (GearSpec, build-params dict) from gear.analyse.  The toothed zone then enters the
    profile as its blank (end planes, tip chamfer cones, tip cylinder), scan regions inside it are not
    profiled, and the exact tooth spaces are cut from the revolved solid before holes and bosses.
    screw: optional ScrewSpec from screw.analyse, handled the same way: the rotor zone enters the profile
    as a blank (end planes, tip cylinder) and everything of the blank outside the exact helical sweep of
    the fitted transverse profile is cut away."""
    t0 = time.time()
    res = RevolveResult(mesh_volume=float(mesh.volume) if mesh.is_watertight else 0.0)
    V = np.asarray(mesh.vertices)
    Fc = np.asarray(mesh.faces)
    gear_ids = set()
    if gear is not None:
        gs, gp = gear
        ga0, gu = np.asarray(gs.axis_point, float), np.asarray(gs.axis_dir, float)
        Cf = np.asarray(mesh.triangles_center) - ga0
        zf = Cf @ gu; rf = np.linalg.norm(Cf - np.outer(zf, gu), axis=1)
        za, zb = gs.chamfers["z_end0"], gs.chamfers["z_end1"]
        inz = (zf > za - 0.05) & (zf < zb + 0.05) & (rf > gs.r_root - 0.3)
        Af = np.asarray(mesh.area_faces)
        for r in regions:
            if Af[r.faces][inz[r.faces]].sum() > 0.5 * Af[r.faces].sum():
                gear_ids.add(r.id)
        res.gear_zone_regions = sorted(int(i) for i in gear_ids)
        res._blank_area = float(sum(r.area for r in regions if r.id in gear_ids))
        regions = [r for r in regions if r.id not in gear_ids]
    if screw is not None:
        sa0, su = np.asarray(screw.axis_point, float), np.asarray(screw.axis_dir, float)
        Cf = np.asarray(mesh.triangles_center) - sa0
        zf = Cf @ su; rf = np.linalg.norm(Cf - np.outer(zf, su), axis=1)
        inz = (zf > screw.z_a - 0.05) & (zf < screw.z_b + 0.05) & (rf > screw.r_root - 0.3)
        Af = np.asarray(mesh.area_faces)
        for r in regions:
            if Af[r.faces][inz[r.faces]].sum() > 0.5 * Af[r.faces].sum():
                gear_ids.add(r.id)
        res.gear_zone_regions = sorted(int(i) for i in gear_ids)
        res._blank_area = float(sum(r.area for r in regions if r.id in gear_ids)) + float(getattr(res, "_blank_area", 0.0) or 0.0)
        regions = [r for r in regions if r.id not in gear_ids]
    tot = float(sum(r.area for r in regions)) or 1.0
    # axis from the largest axial primitive
    axial = [r for r in regions if r.primitive is not None and r.primitive.kind in ("cylinder", "cone", "torus")]
    if not axial:
        res.reason = "no axial primitive"
        return res
    ref = max(axial, key=lambda r: r.area)
    q = ref.primitive.params
    u = np.asarray(q["axis"], float); u = u / np.linalg.norm(u)
    a0 = np.asarray(q.get("point", q.get("apex", q.get("center"))), float)
    if gear is not None:                  # the gear's axis (circle-centre fit of the round slices)
        u, a0 = gu / np.linalg.norm(gu), ga0
    if screw is not None:
        u, a0 = su / np.linalg.norm(su), sa0
    e1 = np.cross(u, [1.0, 0, 0])
    if np.linalg.norm(e1) < 0.1:
        e1 = np.cross(u, [0, 1.0, 0])
    e1 /= np.linalg.norm(e1)
    curves, coax_area = {}, 0.0
    cap = _joint_caps(mesh, a0, u, res)
    rf_ = (lambda r: r.faces[~cap[r.faces]]) if cap is not None else (lambda r: r.faces)
    axisym_ids = set()         # freeform regions refit as one line of revolution (robust, outliers allowed)
    band_curves = {}           # freeform regions that are several surfaces of revolution: [(id, line, X2)]
    # Gear/screw: hub primitives are fit to the scan axis; the blank forces the gear axis. Allow
    # 15 deg / 3 mm so hub tori/cylinders still enter the revolve profile (p01 fillets).
    _kw = {}
    if gear is not None or screw is not None:
        _kw = {"coax_cos": float(np.cos(np.deg2rad(15.0))), "off_lim": max(5 * tol, 3.0)}
    for r in regions:
        if not len(rf_(r)):
            continue
        c = _curve2d(r.primitive, a0, u, tol, **_kw) if r.primitive is not None else None
        if c is None and r.primitive is not None and r.primitive.kind in ("bspline", "freeform", "revolution"):
            c = _axisym_curve(V[np.unique(Fc[rf_(r)])], a0, u, tol, r.id, res)
            if c is not None:
                axisym_ids.add(r.id)
            else:
                Pf = V[np.unique(Fc[rf_(r)])]; df = Pf - a0; zf = df @ u
                Xf = np.c_[np.linalg.norm(df - np.outer(zf, u), axis=1), zf]
                bl = _axisym_bands(Pf, Xf, a0, u, tol, r.id, res)
                if bl:
                    band_curves[r.id] = bl
                    coax_area += r.area * sum(len(b_[2]) for b_ in bl) / len(Xf)
        if c is not None and c[0] == "circle":
            c = _flat_circle(c, V[np.unique(Fc[rf_(r)])], a0, u, tol, r.id, res)
        if c is not None:
            curves[r.id] = c
            coax_area += r.area
    # Credit the gear/screw blank (exact revolved solid) as coaxial so hub freeform on scans
    # does not false-fail the gate (p01 stripped remainder ~0.60–0.80 depending on segmentation).
    blank_area = float(getattr(res, "_blank_area", 0.0) or 0.0)
    # Cross-holes, offset bores and their fillets are cut later by _cut_holes; a minority
    # side-flat (capped at 25 % of area) is milled off the revolved body.  Neither should
    # fail the coaxial-area gate: turned_40x30x29_a is a Ø24/Ø12 Y-axis body with a Z-flat
    # and four Z-holes, raw coaxial 58.5 %, and fell through to a 25 mm³ cell sliver.
    coaxial_ids = set(curves) | set(axisym_ids) | set(band_curves)
    cut_area, flat_area, flat_credit, tot_adj = _feature_area_credit(regions, coaxial_ids, tot)
    tot_gate = max(tot_adj, coax_area) + blank_area
    coax_gate = coax_area + blank_area
    res.axis_point, res.axis_dir = a0.round(4).tolist(), u.round(6).tolist()
    res.coaxial_area_raw = round(coax_area / (tot or 1.0), 4)
    res.coaxial_area_frac = round(coax_gate / (tot_gate or 1.0), 4)
    res.gate_credit = {"cut_features_mm2": round(cut_area, 1),
                       "side_flats_mm2": round(flat_area, 1),
                       "side_flats_credited_mm2": round(flat_credit, 1)}
    # Gear/screw: 0.55 covers the observed hub-remainder band; plain turned parts keep 0.85.
    coax_need = 0.55 if gear is not None or screw is not None else min_frac
    if coax_gate / (tot_gate or 1.0) < coax_need:
        res.reason = (f"not a turned part: coaxial area {100 * coax_gate / (tot_gate or 1.0):.1f}% < "
                      f"{100 * coax_need:.0f}%")
        return res
    gap = max(1.0, 20 * tol)
    segs = []
    P0 = V - a0; z0_ = P0 @ u
    X0 = np.c_[np.linalg.norm(P0 - np.outer(z0_, u), axis=1), z0_]
    for rid_, bl in band_curves.items():
        for sid, c2, Xs in bl:
            segs += _segments(c2, Xs, sid, gap, Xall=X0, small_gap=max(0.3, 4 * tol), lim=max(0.3, 5 * tol))
    for r in regions:
        if r.id not in curves:
            continue
        P = V[np.unique(Fc[rf_(r)])] - a0
        z = P @ u
        rr = np.linalg.norm(P - np.outer(z, u), axis=1)
        X2 = np.c_[rr, z]
        if r.id in axisym_ids:
            # the robust refit tolerates outliers, but its segment ends came from ALL the region's points: a
            # smooth-blended freeform region spanning several diameters (pinion p01: the r 11.05 journal +
            # the r 9.21 neck behind a filleted step, 6.8 % off the line) was profiled as one r 11.05 line
            # over the neck (+1.85 mm of material the part does not have). Segments now come from the
            # inliers only; each axial band of outliers that is itself a line of revolution becomes its own
            # segment, the rest is left to the profile synthesis between neighbours.
            inl = _curve_dist(curves[r.id], X2) < max(0.3, 5 * tol)
            if (~inl).sum() >= 40:
                for sid, c2, Xs in _axisym_bands((P + a0)[~inl], X2[~inl], a0, u, tol, r.id, res):
                    segs += _segments(c2, Xs, sid, gap, Xall=X0, small_gap=max(0.3, 4 * tol), lim=max(0.3, 5 * tol))
            X2 = X2[inl]
        segs += _segments(curves[r.id], X2, r.id, gap, Xall=X0, small_gap=max(0.3, 4 * tol), lim=max(0.3, 5 * tol))
    segs += _unassigned_revolution_segments(mesh, labels, V, Fc, a0, u, tol, gap, X0, res, exclude=cap)
    if gear is not None:
        segs += _gear_blank_segments(gs, gp, V, a0, u)
    if screw is not None:
        segs += _screw_blank_segments(screw, V, a0, u)
    # solid shaft: the profile closes along the axis (scan reaches the axis region at both ends)
    Pz = V - a0; zz_ = Pz @ u; rr_ = np.linalg.norm(Pz - np.outer(zz_, u), axis=1)
    Rm = float(rr_.max()); zlo, zhi = float(zz_.min()), float(zz_.max())
    endlen = 0.02 * (zhi - zlo)
    near = rr_ < 0.25 * Rm
    if near[zz_ < zlo + endlen].any() and near[zz_ > zhi - endlen].any() and not _has_through_bore(curves, regions, V, Fc, a0, u, zlo, zhi):
        p_, q_ = np.array([0.0, zlo]), np.array([0.0, zhi])
        segs.append({"region": "axis", "curve": ("line", p_, np.array([0.0, 1.0])), "t0": zlo, "t1": zhi, "n": 1,
                     "E0": p_, "E1": q_, "tm": 0.5 * (zlo + zhi), "fixed": True})
    if len(segs) < 3:
        res.reason = "fewer than 3 profile segments"
        return res
    # drop tiny segments (shorter than 2 tol) - they cannot carry an edge
    keep = [s for s in segs if np.linalg.norm(s["E1"] - s["E0"]) > 2 * tol]
    res.dropped_segments = [{"region": s["region"], "why": "shorter than 2 tol"} for s in segs if s not in keep]
    segs = _suppress_overlaps(keep, tol, res)
    # refit each straight profile segment's offset on ALL scan vertices near it (incl. unassigned
    # ones and those of duplicate fits dropped above); direction kept (it carries the snapped intent)
    Pall = V - a0; zall = Pall @ u
    Xall = np.c_[np.linalg.norm(Pall - np.outer(zall, u), axis=1), zall]
    lim = max(0.3, 5 * tol)
    for sg in segs:
        c = sg["curve"]
        if c[0] != "line" or sg.get("fixed"):
            continue
        nrm = np.array([-c[2][1], c[2][0]])
        tt = _param(c, Xall); L = sg["t1"] - sg["t0"]; mg = min(0.5, 0.2 * L)
        dd = (Xall - c[1]) @ nrm
        k = (np.abs(dd) < lim) & (tt > sg["t0"] + mg) & (tt < sg["t1"] - mg)
        if k.sum() < 30:
            continue
        sh = float(np.median(dd[k]))
        if abs(sh) > 0.5 * tol:
            q = c[1] + sh * nrm
            sg["curve"] = ("line", q, c[2])
            sg["E0"], sg["E1"] = _point(sg["curve"], sg["t0"]), _point(sg["curve"], sg["t1"])
            res.profile_refits.append({"region": sg["region"], "offset_mm": round(sh, 4), "points": int(k.sum())})
    segs = _split_steps(segs, Xall, tol, res)
    for _it in range(6):
        order = _chain(segs)
        unused = [i for i in range(len(segs)) if i not in {o[0] for o in order}]
        chain = []
        for i, rev in order:
            s = dict(segs[i])
            if rev:
                s["E0"], s["E1"], s["t0"], s["t1"] = s["E1"], s["E0"], s["t1"], s["t0"]
            chain.append(s)
        # vertices / connectors
        joins = []
        for k in range(len(chain)):
            A, B = chain[k], chain[(k + 1) % len(chain)]
            pts, conn = _join(A["curve"], A["E1"], B["curve"], B["E0"], tol)
            joins.append((pts, conn))
        # validate each segment's trimmed extent: entering vertex -> leaving vertex must run in the data's
        # direction; a segment that flips is a spurious sliver and is removed, then the loop is rebuilt
        bad = []
        for k, s in enumerate(chain):
            vin = joins[k - 1][0][-1]
            vout = joins[k][0][0]
            c = s["curve"]
            ti = float(_param(c, vin[None])[0]); to = float(_param(c, vout[None])[0])
            if c[0] == "circle":
                ti = s["tm"] + math.remainder(ti - s["tm"], 2 * math.pi)
                to = s["tm"] + math.remainder(to - s["tm"], 2 * math.pi)
            dirn = np.sign(s["t1"] - s["t0"]) or 1.0
            if (to - ti) * dirn <= 0:
                bad.append(k)
        if not bad:
            break
        drop_ids = {id(segs[order[k][0]]) for k in bad}
        res.dropped_segments += [{"region": chain[k]["region"], "why": "reversed after trimming"} for k in bad]
        segs = [s for s in segs if id(s) not in drop_ids]
        if len(segs) < 3:
            res.reason = "profile collapsed"
            return res
    else:
        res.reason = "profile did not stabilise"
        return res
    res.dropped_segments += [{"region": segs[i]["region"], "why": "not on the profile loop"} for i in unused]
    # build the wire
    from OCP.GC import GC_MakeSegment, GC_MakeArcOfCircle
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeFace
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeRevol
    from OCP.gp import gp_Ax1, gp_Dir
    edges = []
    minlen = max(1e-4, 0.01 * tol)
    for k, s in enumerate(chain):
        vin = joins[k - 1][0][-1]
        vout = joins[k][0][0]
        if np.linalg.norm(vout - vin) > minlen:
            c = s["curve"]
            if c[0] == "line":
                g = GC_MakeSegment(_to3(vin, a0, u, e1), _to3(vout, a0, u, e1)).Value()
            else:
                ti = float(_param(c, vin[None])[0]); to = float(_param(c, vout[None])[0])
                ti = s["tm"] + math.remainder(ti - s["tm"], 2 * math.pi)
                to = s["tm"] + math.remainder(to - s["tm"], 2 * math.pi)
                g = GC_MakeArcOfCircle(_to3(vin, a0, u, e1), _to3(_point(c, 0.5 * (ti + to)), a0, u, e1),
                                       _to3(vout, a0, u, e1)).Value()
            edges.append(g)
            res.profile.append({"region": s["region"], "kind": c[0], "from": np.round(vin, 4).tolist(),
                                "to": np.round(vout, 4).tolist(), "synthesised": False})
        pts, conn = joins[k]
        if conn is not None and np.linalg.norm(pts[1] - pts[0]) > minlen:
            edges.append(GC_MakeSegment(_to3(pts[0], a0, u, e1), _to3(pts[1], a0, u, e1)).Value())
            item = {"region": None, "kind": "line", "from": np.round(pts[0], 4).tolist(),
                    "to": np.round(pts[1], 4).tolist(), "synthesised": True,
                    "between_regions": [s["region"], chain[(k + 1) % len(chain)]["region"]]}
            res.profile.append(item)
            res.synthesised.append(item)
    if min(min(p["from"][0], p["to"][0]) for p in res.profile) < -1e-6:
        res.reason = "profile crosses the axis"
        return res
    mw = BRepBuilderAPI_MakeWire()
    for g in edges:
        mw.Add(BRepBuilderAPI_MakeEdge(g).Edge())
    if not mw.IsDone():
        res.reason = "profile wire not closed"
        return res
    mf = BRepBuilderAPI_MakeFace(mw.Wire(), True)
    if not mf.IsDone():
        res.reason = "profile face failed"
        return res
    rv = BRepPrimAPI_MakeRevol(mf.Face(), gp_Ax1(_to3(np.zeros(2), a0, u, e1), gp_Dir(*map(float, u))), 2 * math.pi)
    rv.Build()
    if not rv.IsDone():
        res.reason = "revolve failed"
        return res
    from . import occ_util as U
    from OCP.TopAbs import TopAbs_SOLID, TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS
    shape = rv.Shape()
    ex = TopExp_Explorer(shape, TopAbs_SOLID)
    if not ex.More():
        res.reason = "revolve gave no solid"
        return res
    solid = TopoDS.Solid(ex.Current())
    if U.volume(solid) < 0:
        solid.Reverse()
    if gear is not None:
        solid = _cut_gear(solid, gs, gp, res)
        if solid is None:
            res.reason = "gear tooth-space cut failed"
            return res
    if screw is not None:
        solid = _cut_screw(solid, screw, res)
        if solid is None:
            res.reason = "screw rotor sweep failed"
            return res
    # non-coaxial holes: concave cylinders whose axis is not the part axis
    solid = _cut_holes(solid, regions, curves, V, Fc, mesh, tol, res)
    solid = _cluster_holes(solid, mesh, tol, res)
    solid = _bosses(solid, mesh, tol, res)
    res.shape = solid
    res.brepcheck_valid = bool(U.check(solid))
    res.n_solids = int(U.count(solid, TopAbs_SOLID))
    res.n_faces = int(U.count(solid, TopAbs_FACE))
    res.volume = float(U.volume_precise(solid) if screw is not None else U.volume(solid))
    res.unmodelled_regions += [{"region": r.id, "kind": r.primitive.kind if r.primitive else None,
                                "area": round(r.area, 2)} for r in regions
                               if r.id not in curves and r.id not in {h["region"] for h in res.holes_cut}]
    res.ok = res.brepcheck_valid and res.n_solids == 1 and res.volume > 0
    if not res.ok:
        res.reason = f"revolved solid invalid (valid={res.brepcheck_valid}, solids={res.n_solids})"
    res.seconds = time.time() - t0
    return res


def _flat_circle(c, P, a0, u, tol, rid, res):
    """A sphere or torus fitted to a short band (a narrow neck cylinder or a thin annular face) is, over
    that band, indistinguishable from a straight profile line, and as an arc it does not meet its
    neighbours the way the part does.  When the arc's sagitta over the band's chord is <= 0.5 tol, the
    band is profiled as its total-least-squares line instead (exactly radial / axial when within 0.5 deg)."""
    d = P - a0; z = d @ u; r = np.linalg.norm(d - np.outer(z, u), axis=1)
    X = np.c_[r, z]
    cen = X.mean(0); _, sv, Vt = np.linalg.svd(X - cen, full_matrices=False)
    t = (X - cen) @ Vt[0]
    chord = float(np.percentile(t, 99.5) - np.percentile(t, 0.5))
    rho = float(c[2])
    sag = rho - math.sqrt(max(rho * rho - 0.25 * chord * chord, 0.0))
    if chord <= 0 or sag > 0.5 * tol:
        return c
    dvec = Vt[0] / np.linalg.norm(Vt[0])
    for ax in (np.array([0.0, 1.0]), np.array([1.0, 0.0])):
        if abs(dvec @ ax) > math.cos(math.radians(0.5)):
            dvec = ax
    rms = float(np.sqrt(np.mean(((X - cen) @ np.array([-dvec[1], dvec[0]])) ** 2)))
    res.profile_refits.append({"region": int(rid), "short_band_as_line": [round(float(v), 4) for v in dvec],
                               "sagitta_mm": round(sag, 4), "chord_mm": round(chord, 3), "rms": round(rms, 4)})
    return ("line", cen, dvec)


def _rid(rid):
    return rid if isinstance(rid, str) else int(rid)


def _axisym_curve(P, a0, u, tol, rid, res):
    """A freeform region that is a surface of revolution about the part axis (e.g. a slightly dished
    collar face the freeform stage took over): robust 2D line or circle fit to its (r, z) points.
    Accepted when it covers >= 300 deg and the inlier rms <= max(tol, 0.03)."""
    d = P - a0; z = d @ u; rv = d - np.outer(z, u); r = np.linalg.norm(rv, axis=1)
    e1 = np.cross(u, [1.0, 0, 0]); e1 = e1 if np.linalg.norm(e1) > 0.1 else np.cross(u, [0, 1.0, 0])
    e1 /= np.linalg.norm(e1); e2 = np.cross(u, e1)
    th = np.degrees(np.arctan2(rv @ e2, rv @ e1))
    if len(np.unique((th // 10).astype(int))) < 30 or len(P) < 50:
        return None
    X = np.c_[r, z]
    keep = np.ones(len(X), bool)
    best = None
    for _ in range(3):
        c = X[keep].mean(0); _, _, Vt = np.linalg.svd(X[keep] - c, full_matrices=False); n = Vt[1]
        dl = (X - c) @ n
        mad = 1.4826 * np.median(np.abs(dl[keep])) + 1e-9
        keep = np.abs(dl) < max(3 * mad, 0.5 * tol)
    rms = float(np.sqrt(np.mean(dl[keep] ** 2)))
    if keep.mean() >= 0.8:
        dvec = Vt[0] / np.linalg.norm(Vt[0])
        best = ("line", c, dvec, rms)
    # 0.04 mm: scanned hub collars on pinions (p01) land at ~0.032–0.033 and were false-rejected at 0.03
    if best is None or best[3] > max(tol, 0.04):
        res.dropped_segments.append({"region": _rid(rid), "why": f"freeform, not a line of revolution (rms {rms:.3f})"})
        return None
    res.profile_refits.append({"region": _rid(rid), "freeform_as_revolution": "line", "rms": round(best[3], 4),
                               "inlier_frac": round(float(keep.mean()), 3)})
    return ("line", best[1], best[2])


def _joint_caps(mesh, a0, u, res, gap=4.0, min_comp_frac=0.02, face_cos=-0.7):
    """Faces at the joint of a scan assembled from separate captures that were never merged (pinion
    X2.B1872: the lower shaft and the gear are two watertight shells; the shaft shell's cut face at z -95.5
    and the gear shell's face at z -93 face each other across a 2-2.5 mm gap from r 0 to r ~15). Region 3
    (the gear face plane) carried those points down to r 0.8, and the profile synthesis closed it with a
    cone back out to the shaft fillet: a neck to r 0.8 where the real part is solid.
    Test (multi-shell scans only: >= 2 shells each >= 2 % of the faces): a face is a joint face when the ray
    along its outward normal hits a DIFFERENT shell within `gap` mm on a face that looks back at it
    (normals' dot < -0.7). A groove, fillet or shoulder of one capture faces its own shell, so it is never
    taken; a single-shell scan returns None."""
    try:
        import trimesh
        Fn = len(mesh.faces)
        comp = trimesh.graph.connected_component_labels(mesh.face_adjacency, node_count=Fn)
        cnt = np.bincount(comp)
        big = np.where(cnt >= min_comp_frac * Fn)[0]
        if len(big) < 2:
            return None
        C = np.asarray(mesh.triangles_center); N = np.asarray(mesh.face_normals)
        from scipy.spatial import cKDTree
        cand = []
        for c in big:
            q = np.where(comp == c)[0]
            other = np.where(np.isin(comp, big) & (comp != c))[0]
            dd, _ = cKDTree(C[other]).query(C[q], distance_upper_bound=gap + 2.0)
            cand.append(q[np.isfinite(dd)])
        cand = np.concatenate(cand)
        if not len(cand):
            return None
        lo, hi = C[cand].min(0) - 2 * gap, C[cand].max(0) + 2 * gap
        loc = np.where(np.all((C > lo) & (C < hi), axis=1))[0]
        sub = mesh.submesh([loc], append=True, repair=False)
        rmi = trimesh.ray.ray_triangle.RayMeshIntersector(sub)
        O = C[cand] + 0.01 * N[cand]
        keep = np.zeros(Fn, bool)
        for b0 in range(0, len(cand), 5000):
            sl = slice(b0, b0 + 5000)
            hl, ri, ti = rmi.intersects_location(O[sl], N[cand][sl], multiple_hits=False)
            if not len(ri):
                continue
            f_hit = loc[ti]; me = cand[sl][ri]
            ok = (np.linalg.norm(hl - O[sl][ri], axis=1) <= gap) & (comp[f_hit] != comp[me]) & \
                 (np.einsum("ij,ij->i", N[f_hit], N[me]) < face_cos)
            keep[me[ok]] = True
        if not keep.any():
            return None
        res.profile_refits.append({"region": "joint_caps", "faces": int(keep.sum()), "candidates": int(len(cand)),
                                   "area_mm2": round(float(np.asarray(mesh.area_faces)[keep].sum()), 2), "shells": int(len(big)),
                                   "why": "faces of two scan shells facing each other across <= 4 mm left out of the profile"})
        return keep
    except Exception as e:  # noqa: BLE001
        res.profile_refits.append({"region": "joint_caps", "error": repr(e)[:200]})
        return None


def _axisym_bands(P3, X2, a0, u, tol, rid, res, min_pts=40, min_len=1.0, max_bands=6):
    """Axial cylinders and radial planes inside a set of (r, z) points that is a MIX of surfaces of
    revolution (a smooth-blended freeform region): mode-seek the radius (then the z) histogram, take the
    points within max(0.3, 5 tol) of the mode, split them at axial gaps > 1 mm, and keep each band that is a
    line of revolution on its own (>= 300 deg, rms <= 0.04: _axisym_curve). Returns [(f"{rid}.{k}", line, X2_band)]."""
    lim = max(0.3, 5 * tol)
    used = np.zeros(len(X2), bool)
    out, recs = [], []
    for col, other in ((0, 1), (1, 0)):              # 0: cylinders (constant r), 1: radial faces (constant z)
        left = ~used
        for _ in range(max_bands):
            if left.sum() < min_pts or len(out) >= max_bands:
                break
            v = X2[left, col]
            edges = np.arange(v.min() - 0.05, v.max() + 0.1, 0.05)
            if len(edges) < 2:
                break
            h, e = np.histogram(v, bins=edges)
            pk = 0.5 * (e[np.argmax(h)] + e[np.argmax(h) + 1])
            cand = np.where(left & (np.abs(X2[:, col] - pk) < lim))[0]
            left[cand] = False                       # tried once, whatever the outcome
            if len(cand) < min_pts:
                continue
            o = np.argsort(X2[cand, other]); ts = X2[cand[o], other]
            lo = 0
            for hi in list(np.where(np.diff(ts) > 1.0)[0]) + [len(ts) - 1]:
                band = cand[o[lo:hi + 1]]; lo = hi + 1
                ext = float(X2[band, other].max() - X2[band, other].min())
                if len(band) < min_pts or ext < min_len:
                    continue
                sub = _SubRes()
                sid = f"{rid}.{len(out) + 1}"
                c2 = _axisym_curve(P3[band], a0, u, tol, sid, sub)
                rec = {"kind": "cylinder" if col == 0 else "radial", "at": round(float(pk), 3), "n": int(len(band)),
                       "span": [round(float(X2[band, other].min()), 3), round(float(X2[band, other].max()), 3)]}
                if c2 is None:
                    rec["fit"] = "not a line of revolution"
                else:
                    rec.update(fit="line", segment_region=sid, rms=sub.profile_refits[-1]["rms"])
                    out.append((sid, c2, X2[band]))
                    used[band] = True
                recs.append(rec)
    if recs:
        res.profile_refits.append({"region": _rid(rid), "revolution_bands": recs})
    return out


def _unassigned_revolution_segments(mesh, labels, V, Fc, a0, u, tol, gap, X0, res, min_area=2.0, max_rho=6.0, exclude=None):
    """Faces no region took (labels -1) that form a full ring about the axis: a rounded groove / small
    radius (pinion p01: the circlip groove r 10.0-10.9 at z 48-50, 1,940 unassigned faces, was bridged by a
    straight r 11.07 line; up to 1 mm of material the part does not have). Circle (Kasa) or line fit in
    (r, z); accepted at >= 300 deg coverage and rms <= max(tol, 0.04)."""
    lab = np.asarray(labels)
    un = np.where((lab < 0) & (~exclude if exclude is not None else True))[0]
    if len(un) < 30:
        return []
    import trimesh
    adj = np.asarray(mesh.face_adjacency)
    keep = (lab[adj[:, 0]] < 0) & (lab[adj[:, 1]] < 0)
    comps = trimesh.graph.connected_components(adj[keep], nodes=un, min_len=30)
    area = np.asarray(mesh.area_faces)
    e1 = np.cross(u, [1.0, 0, 0]); e1 = e1 if np.linalg.norm(e1) > 0.1 else np.cross(u, [0, 1.0, 0])
    e1 /= np.linalg.norm(e1); e2 = np.cross(u, e1)
    out, recs = [], []
    for k, fc in enumerate(sorted(comps, key=len, reverse=True)[:20]):
        if area[fc].sum() < min_area:
            continue
        vi = np.unique(Fc[fc]); d = V[vi] - a0; z = d @ u; rv = d - np.outer(z, u); r = np.linalg.norm(rv, axis=1)
        th = np.degrees(np.arctan2(rv @ e2, rv @ e1))
        if len(np.unique((th // 10).astype(int))) < 30:
            continue
        X = np.c_[r, z]
        rec0 = {"faces": int(len(fc)), "area": round(float(area[fc].sum()), 2), "z": [round(float(z.min()), 3), round(float(z.max()), 3)],
                "r": [round(float(r.min()), 3), round(float(r.max()), 3)], "fits": []}
        left = np.ones(len(X), bool)
        for _ in range(3):                      # up to 3 rings per component (groove arc + its corner rounds)
            if left.sum() < 100:
                break
            fit = _ransac_circle(X[left], lim=max(0.05, 2 * tol), max_rho=max_rho)
            if fit is None:
                break
            cen, rho, inl_l = fit
            idx = np.where(left)[0][inl_l]
            if len(np.unique((th[idx] // 10).astype(int))) < 30 or len(idx) < 100:
                rec0["fits"].append({"rho": round(rho, 3), "n": int(len(idx)), "fit": "partial ring"})
                left[idx] = False
                continue
            rms_c = float(np.sqrt(np.mean((np.linalg.norm(X[idx] - cen, axis=1) - rho) ** 2)))
            left[idx] = False
            if rms_c > max(tol, 0.04):
                rec0["fits"].append({"rho": round(rho, 3), "n": int(len(idx)), "fit": f"rms {rms_c:.3f}"})
                continue
            sid = f"unassigned.{len(out) + 1}"
            seg_c = _segments(("circle", cen, rho), X[idx], sid, gap, Xall=X0, small_gap=max(0.3, 4 * tol), lim=max(0.3, 5 * tol))
            out += seg_c
            rec0["fits"].append({"rho": round(rho, 3), "centre_rz": np.round(cen, 3).tolist(), "n": int(len(idx)),
                                 "rms": round(rms_c, 4), "segment_region": sid, "segments": len(seg_c)})
        recs.append(rec0)
    if recs:
        res.profile_refits.append({"region": "unassigned", "revolution_rings": recs})
    return out


def _ransac_circle(X, lim, max_rho, tries=300, seed=0):
    """Dominant circle in 2D points (radius <= max_rho): RANSAC over point triples, Kasa refit on the
    inliers. Returns (centre, rho, inlier mask) or None when fewer than 100 points agree."""
    rng = np.random.default_rng(seed)
    n = len(X)
    best = None
    for _ in range(tries):
        a, b, c = X[rng.choice(n, 3, replace=False)]
        M_ = np.array([[b[0] - a[0], b[1] - a[1]], [c[0] - a[0], c[1] - a[1]]]) * 2
        if abs(np.linalg.det(M_)) < 1e-9:
            continue
        rhs = np.array([b @ b - a @ a, c @ c - a @ a])
        cen = np.linalg.solve(M_, rhs); rho = float(np.linalg.norm(a - cen))
        if rho > max_rho or rho < 0.05:
            continue
        inl = np.abs(np.linalg.norm(X - cen, axis=1) - rho) < lim
        if best is None or inl.sum() > best[2].sum():
            best = (cen, rho, inl)
    if best is None or best[2].sum() < 100:
        return None
    for _ in range(2):
        inl = best[2]
        A_ = np.c_[2 * X[inl], np.ones(inl.sum())]; bb = (X[inl] ** 2).sum(1)
        sol, *_ = np.linalg.lstsq(A_, bb, rcond=None)
        cen = sol[:2]; rho = float(np.sqrt(max(sol[2] + cen @ cen, 0.0)))
        best = (cen, rho, np.abs(np.linalg.norm(X - cen, axis=1) - rho) < lim)
    return best if best[1] <= max_rho else None


class _SubRes:
    def __init__(self):
        self.profile_refits, self.dropped_segments = [], []


def _split_steps(segs, Xall, tol, res, min_step=0.02, min_len=4.0):
    """A long cylinder segment that carries a small diameter step (e.g. a bearing seat 0.1 mm under the
    journal) is split at the change point of the per-mm median radius; each piece gets its own radius.
    Accepted when the step is >= min_step and >= 4x the scatter of the bin medians, both pieces >= min_len."""
    out = []
    lim = max(0.3, 5 * tol)
    for sg in segs:
        c = sg["curve"]
        if sg.get("fixed") or c[0] != "line" or abs(c[2][1]) < 0.999 or abs(sg["t1"] - sg["t0"]) < 2 * min_len + 2:
            out.append(sg); continue
        pieces = [sg]
        for _depth in range(2):
            nxt = []
            for pc in pieces:
                sp = _best_split(pc, Xall, lim, min_step, min_len)
                if sp is None:
                    nxt.append(pc); continue
                parts, info = sp
                res.profile_refits.append({"region": pc["region"], "split_at_z": [round(v, 3) for v in info["z"]],
                                           "radius_steps_mm": [round(-v, 4) for v in info["step"]], "bins": info["bins"]})
                nxt += parts
            if len(nxt) == len(pieces):
                break
            pieces = nxt
        out += pieces
    return out


def _best_split(sg, Xall, lim, min_step, min_len):
    c = sg["curve"]
    nrm = np.array([-c[2][1], c[2][0]])
    t = _param(c, Xall); dd = (Xall - c[1]) @ nrm
    lo, hi = min(sg["t0"], sg["t1"]), max(sg["t0"], sg["t1"])
    k = (np.abs(dd) < lim) & (t > lo + 0.5) & (t < hi - 0.5)
    if k.sum() < 100:
        return None
    edges = np.arange(lo + 0.5, hi - 0.5 + 1e-9, 1.0)
    if len(edges) < 2 * min_len + 1:
        return None
    idx = np.digitize(t[k], edges) - 1
    med = np.array([np.median(dd[k][idx == i]) if np.sum(idx == i) >= 5 else np.nan for i in range(len(edges) - 1)])
    good = ~np.isnan(med)
    if good.sum() < 2 * min_len:
        return None
    zc = 0.5 * (edges[:-1] + edges[1:])
    m, zc = med[good], zc[good]
    best = None
    for j in range(int(min_len), len(m) - int(min_len) + 1):
        a, b = m[:j], m[j:]
        sse = float(((a - np.median(a)) ** 2).sum() + ((b - np.median(b)) ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, j)
    _, j = best
    L = int(min_len)
    # three pieces (a recessed / raised band in the middle, e.g. a seat between journal lengths)
    best3 = None
    for j1 in range(L, len(m) - 2 * L + 1):
        for j2 in range(j1 + L, len(m) - L + 1):
            pa, pb, pc = m[:j1], m[j1:j2], m[j2:]
            sse = float(((pa - np.median(pa)) ** 2).sum() + ((pb - np.median(pb)) ** 2).sum() + ((pc - np.median(pc)) ** 2).sum())
            if best3 is None or sse < best3[0]:
                best3 = (sse, j1, j2)

    def ok(pieces):
        meds = [float(np.median(q)) for q in pieces]
        sc = 1.4826 * float(np.median(np.abs(np.concatenate([q - np.median(q) for q in pieces]))))
        steps = np.diff(meds)
        return all(abs(st) >= min_step and abs(st) >= 4 * sc for st in steps), meds, steps
    cut = None
    ok3, meds3, steps3 = ok([m[:best3[1]], m[best3[1]:best3[2]], m[best3[2]:]]) if best3 else (False, None, None)
    ok2, meds2, steps2 = ok([m[:j], m[j:]])
    if ok3 and (not ok2 or best3[0] < 0.5 * best[0]):
        cut = ([best3[1], best3[2]], meds3, steps3)
    elif ok2:
        cut = ([j], meds2, steps2)
    if cut is None:
        return None
    js, meds, steps = cut
    zs = [0.5 * (zc[q - 1] + zc[q]) for q in js]
    bounds = [lo] + zs + [hi]
    sgn = 1.0 if sg["t1"] >= sg["t0"] else -1.0
    parts = []
    for (t0, t1), off in zip(zip(bounds[:-1], bounds[1:]), meds):
        q = c[1] + off * nrm
        cc = ("line", q, c[2])
        ta, tb = (t0, t1) if sgn > 0 else (t1, t0)
        parts.append({**sg, "curve": cc, "t0": ta, "t1": tb, "E0": _point(cc, ta), "E1": _point(cc, tb),
                      "tm": 0.5 * (t0 + t1), "n": max(8, sg["n"] // len(meds)), "fixed": True})
    if sgn < 0:
        parts = parts[::-1]
    return parts, {"z": [float(v) for v in zs], "step": [float(v) for v in steps], "bins": int(len(m))}


def _has_through_bore(curves, regions, V, Fc, a0, u, zlo, zhi):
    """A coaxial cylinder whose data spans > 80% of the part length near the axis side is a through bore."""
    for r in regions:
        c = curves.get(r.id)
        if c is None or c[0] != "line" or abs(c[2][1]) < 0.999:
            continue
        P = V[np.unique(Fc[r.faces])] - a0
        z = P @ u
        if np.ptp(z) > 0.8 * (zhi - zlo) and c[1][0] < 0.5 * float(np.linalg.norm(P - np.outer(z, u), axis=1).max() + 1e-9):
            return True
    return False


def _gear_blank_segments(g, gp, V, a0, u):
    """Profile of the gear blank in (r, z): end planes, tip chamfer cones, tip cylinder (fixed: not refitted
    to the rounded scan tips).  Each end plane runs from the radius of the scan just outside that end
    (neck / shoulder) to the chamfer."""
    za, zb = g.chamfers["z_end0"], g.chamfers["z_end1"]
    rt = gp["r_tip"]
    P = V - a0; z = P @ u; r = np.linalg.norm(P - np.outer(z, u), axis=1)

    def r_out(zend, sgn):
        k = (z * sgn > zend * sgn + 0.05) & (z * sgn < zend * sgn + 0.4) & (r > 0.5 * g.r_root)
        return float(np.median(r[k])) if k.sum() >= 20 else g.r_root
    r0, r1 = r_out(za, -1), r_out(zb, 1)
    ch = g.chamfers.get("chamfers") or {}
    c0, c1 = ch.get("end0"), ch.get("end1")
    out = []

    def seg(p, q, name):
        p, q = np.asarray(p, float), np.asarray(q, float)
        d = (q - p) / np.linalg.norm(q - p)
        c = ("line", p, d)
        out.append({"region": name, "curve": c, "t0": 0.0, "t1": float(np.linalg.norm(q - p)), "n": 10 ** 7, "gear": True,
                    "E0": p, "E1": q, "tm": 0.5 * float(np.linalg.norm(q - p)), "fixed": True})
    a0_ = (c0["axial_mm"], c0["radial_mm"]) if c0 else (0.0, 0.0)
    a1_ = (c1["axial_mm"], c1["radial_mm"]) if c1 else (0.0, 0.0)
    seg((r0, za), (rt - a0_[1], za), "gear:end0")
    if c0:
        seg((rt - a0_[1], za), (rt, za + a0_[0]), "gear:chamfer0")
    seg((rt, za + a0_[0]), (rt, zb - a1_[0]), "gear:tip")
    if c1:
        seg((rt, zb - a1_[0]), (rt - a1_[1], zb), "gear:chamfer1")
    seg((rt - a1_[1], zb), (r1, zb), "gear:end1")
    return out


def _screw_blank_segments(s, V, a0, u):
    """Rotor zone blank in (r, z): end planes from the journal radius just outside each end up to a tip
    cylinder slightly above the fitted tip (the sweep then defines the whole outer surface)."""
    za, zb = s.z_a, s.z_b
    rt = s.r_tip + 0.5
    P = V - a0; z = P @ u; r = np.linalg.norm(P - np.outer(z, u), axis=1)

    def r_out(zend, sgn):
        k = (z * sgn > zend * sgn + 0.1) & (z * sgn < zend * sgn + 0.6)
        return float(np.median(r[k])) if k.sum() >= 20 else 0.5 * s.r_root
    r0, r1 = min(r_out(za, -1), s.r_root - 0.2), min(r_out(zb, 1), s.r_root - 0.2)
    out = []

    def seg(p, q, name):
        p, q = np.asarray(p, float), np.asarray(q, float)
        d = (q - p) / np.linalg.norm(q - p)
        out.append({"region": name, "curve": ("line", p, d), "t0": 0.0, "t1": float(np.linalg.norm(q - p)), "n": 10 ** 7,
                    "gear": True, "E0": p, "E1": q, "tm": 0.5 * float(np.linalg.norm(q - p)), "fixed": True})
    seg((r0, za), (rt, za), "screw:end0")
    seg((rt, za), (rt, zb), "screw:tip_blank")
    seg((rt, zb), (r1, zb), "screw:end1")
    return out


def _cut_screw(solid, s, res):
    """solid minus (rotor-zone blank cylinder minus the helical rotor sweep)."""
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder
    from OCP.gp import gp_Ax2, gp_Pnt, gp_Dir
    from OCP.TopAbs import TopAbs_SOLID
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS
    from . import screw as SC
    from . import occ_util as U
    t = time.time()
    try:
        rotor = SC.build_rotor(s)
    except Exception as ex:  # noqa: BLE001
        res.screw = {"ok": False, "reason": f"rotor sweep: {ex}"}
        return None
    a0, u = np.asarray(s.axis_point, float), np.asarray(s.axis_dir, float)
    base = a0 + s.z_a * u
    cylr = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(*map(float, base)), gp_Dir(*map(float, u))), s.r_tip + 2.0,
                                    float(s.z_b - s.z_a)).Shape()
    op = BRepAlgoAPI_Cut(cylr, rotor); op.Build()
    if not op.IsDone():
        res.screw = {"ok": False, "reason": "blank minus rotor failed"}
        return None
    op2 = BRepAlgoAPI_Cut(solid, op.Shape()); op2.Build()
    if not op2.IsDone():
        res.screw = {"ok": False, "reason": "boolean cut failed"}
        return None
    ex = TopExp_Explorer(op2.Shape(), TopAbs_SOLID)
    sols = []
    while ex.More():
        sols.append(TopoDS.Solid(ex.Current())); ex.Next()
    if len(sols) != 1:
        res.screw = {"ok": False, "reason": f"cut gave {len(sols)} solids"}
        return None
    res.screw = {"ok": True, "lobes": s.lobes, "volume_removed_mm3": round(U.volume_precise(solid) - U.volume_precise(sols[0]), 3),
                 "seconds": round(time.time() - t, 2)}
    return sols[0]


def _cut_gear(solid, g, gp, res):
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCP.TopAbs import TopAbs_SOLID
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS
    from . import gear as G
    from . import occ_util as U
    t = time.time()
    try:
        tools = G.build_gear_cut(g.teeth, gp["rb"], gp["psi_b"], gp["r_root"], gp["rho"], gp["r_tip"], gp["w"],
                                 g.fit["clocking_rad"], g.chamfers["z_end0"], g.chamfers["z_end1"],
                                 np.asarray(g.axis_point), np.asarray(g.axis_dir))
    except Exception as ex:  # noqa: BLE001
        res.gear = {"ok": False, "reason": f"tooth space build: {ex}"}
        return None
    op = BRepAlgoAPI_Cut(solid, tools)
    op.Build()
    if not op.IsDone():
        res.gear = {"ok": False, "reason": "boolean cut failed"}
        return None
    ex = TopExp_Explorer(op.Shape(), TopAbs_SOLID)
    sols = []
    while ex.More():
        sols.append(TopoDS.Solid(ex.Current())); ex.Next()
    if len(sols) != 1:
        res.gear = {"ok": False, "reason": f"cut gave {len(sols)} solids"}
        return None
    out = sols[0]
    res.gear = {"ok": True, "spaces_cut": g.teeth, "volume_removed_mm3": round(U.volume(solid) - U.volume(out), 3),
                "seconds": round(time.time() - t, 2)}
    return out


def _cut_holes(solid, regions, curves, V, Fc, mesh, tol, res):
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCP.BRepClass3d import BRepClass3d_SolidClassifier
    from OCP.TopAbs import TopAbs_IN
    from OCP.gp import gp_Ax2, gp_Pnt, gp_Dir
    from . import occ_util as U
    for r in regions:
        if r.id in curves or r.primitive is None or r.primitive.kind != "cylinder":
            continue
        q = r.primitive.params
        a = np.asarray(q["axis"], float); a /= np.linalg.norm(a)
        p0 = np.asarray(q["point"], float); R = float(q["radius"])
        # design intent: a hole within 3 deg of the part axis is drilled parallel to it
        u_ax = np.asarray(res.axis_dir, float)
        if abs(a @ u_ax) > np.cos(np.radians(3.0)):
            a = u_ax * np.sign(a @ u_ax)
            P_ = V[np.unique(Fc[r.faces])] - p0
            rad = P_ - np.outer(P_ @ a, a)
            # least-squares circle centre in the plane normal to a
            e1_ = np.cross(a, [1.0, 0, 0]); e1_ = e1_ if np.linalg.norm(e1_) > 0.1 else np.cross(a, [0, 1.0, 0])
            e1_ /= np.linalg.norm(e1_); e2_ = np.cross(a, e1_)
            X2 = np.c_[rad @ e1_, rad @ e2_]
            cc = np.linalg.lstsq(np.c_[2 * X2, np.ones(len(X2))], (X2 ** 2).sum(1), rcond=None)[0]
            p0 = p0 + cc[0] * e1_ + cc[1] * e2_
        if R <= 0 or R > 0.05 * float(np.linalg.norm(np.ptp(V, axis=0))):
            continue      # large non-coaxial "cylinders" on a turned part are mis-fits, not holes
        F = r.faces
        C = np.asarray(mesh.triangles_center)[F]; N = np.asarray(mesh.face_normals)[F]
        d = C - p0
        radial = d - np.outer(d @ a, a)
        concave = float(np.average(np.einsum("ij,ij->i", radial, N) < 0, weights=mesh.area_faces[F]))
        if concave < 0.8:
            continue
        t = (V[np.unique(Fc[F])] - p0) @ a
        t0, t1 = float(t.min()), float(t.max())
        # extend an end that opens to the outside of the solid
        for _ in range(2):
            pass
        ext0 = ext1 = 0.0
        for sgn in (-1, 1):
            tt = (t0 - 1.0) if sgn < 0 else (t1 + 1.0)
            P = p0 + tt * a
            cl = BRepClass3d_SolidClassifier(solid, gp_Pnt(*map(float, P)), 1e-6)
            if cl.State() != TopAbs_IN:
                if sgn < 0:
                    ext0 = 3.0
                else:
                    ext1 = 3.0
        if ext0 == 0.0 and ext1 == 0.0:
            continue      # closed both ends: not a drilled hole from the outside
        base = p0 + (t0 - ext0) * a
        s2 = _cut_cyl(solid, base, a, R, (t1 + ext1) - (t0 - ext0))
        if s2 is None:
            continue
        solid = s2
        res.holes_cut.append({"region": r.id, "diameter": round(2 * R, 4), "depth": round(t1 - t0, 3),
                              "open_ends": [bool(ext0), bool(ext1)], "pattern_angle_deg": 0.0,
                              "axis_parallel_to_part_axis": bool(abs(a @ u_ax) > 0.9999)})
        _fill_caps(mesh, p0, a, R, t0, t1, ext0 > 0, tol, r.id, 0.0, res)
        # copies of the same hole around the part axis that segmentation did not fit (data-verified)
        for phi in _pattern_angles(V[np.unique(Fc[F])], mesh, tol, res):
            Rm = _rot(np.asarray(res.axis_dir, float), np.radians(phi))
            o = np.asarray(res.axis_point, float)
            b2 = o + Rm @ (base - o); a2 = Rm @ a
            s3 = _cut_cyl(solid, b2, a2, R, (t1 + ext1) - (t0 - ext0))
            if s3 is not None:
                solid = s3
                res.holes_cut.append({"region": r.id, "diameter": round(2 * R, 4), "depth": round(t1 - t0, 3),
                                      "open_ends": [bool(ext0), bool(ext1)], "pattern_angle_deg": round(phi, 1)})
                _fill_caps(mesh, o + Rm @ (p0 - o), a2, R, t0, t1, ext0 > 0, tol, r.id, round(phi, 1), res)
    return solid


def _fill_caps(mesh, p0, a, R, t0, t1, open_low, tol, rid, phi, res):
    """Mesh surface spanning the inside of a cut hole while the hole's wall data continues past it
    deeper into the part.  A real hole floor cannot have wall beyond it, so such a cap is a hole-fill
    artefact of the scan (the scanner could not see the bottom and the mesh was closed across it).
    Recorded (never silently dropped) so the deviation gate can report it as excluded."""
    C = np.asarray(mesh.triangles_center); d = C - p0
    t = d @ a; rr = np.linalg.norm(d - np.outer(t, a), axis=1)
    inside = np.where((rr < 0.85 * R) & (t > t0 - 0.5) & (t < t1 + 0.5))[0]
    if len(inside) == 0:
        return
    deep = t1 if open_low else t0                 # blind end of the wall data
    tc = t[inside]
    beyond = (deep - tc) if open_low else (tc - deep)
    margin = max(0.5, 5 * tol)
    cap = inside[beyond > margin]
    if len(cap) == 0:
        return
    A = mesh.area_faces[cap]
    res.fill_artifacts.append({"hole_region": rid, "pattern_angle_deg": phi, "faces": int(len(cap)),
                               "area_mm2": round(float(A.sum()), 2),
                               "hole_cross_section_mm2": round(float(np.pi * R * R), 2),
                               "wall_beyond_cap_mm": round(float(np.median(beyond[beyond > margin])), 2),
                               "why": "surface across the hole bore with hole wall continuing past it"})
    res.fill_faces = cap if res.fill_faces is None else np.union1d(res.fill_faces, cap)


def _bosses(solid, mesh, tol, res, min_area=5.0):
    """Compact protrusions of the scan outside the solid (pins, lugs): fuse a cylinder boss along the
    local surface normal, hollow if the scan shows an inner bore.  Kept only if it brings the
    protrusion's own scan points closer to the solid."""
    from .deviation import _on_surface
    from .occ_util import shape_to_trimesh
    from . import occ_util as U
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse
    from OCP.gp import gp_Ax2, gp_Pnt, gp_Dir
    from OCP.TopAbs import TopAbs_SOLID
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS
    import trimesh
    sm = shape_to_trimesh(solid, lin=0.01, ang=0.1)
    C = np.asarray(mesh.triangles_center)
    cp, dist, tri = _on_surface(sm, C)
    sgn = np.einsum("ij,ij->i", C - cp, sm.face_normals[tri])
    out = (sgn > 0) & (dist > max(0.5, 5 * tol))
    if not out.any():
        return solid
    idx = np.where(out)[0]
    adj = mesh.face_adjacency
    keep = out[adj[:, 0]] & out[adj[:, 1]]
    g = trimesh.graph.connected_components(adj[keep], nodes=idx, min_len=1)
    N = mesh.face_normals
    for comp in g:
        comp = np.asarray(comp)
        area = float(mesh.area_faces[comp].sum())
        if area < min_area:
            continue
        # boss axis = solid normal at the protrusion's foot
        foot = cp[comp].mean(0)
        ax = sm.face_normals[tri[comp]].mean(0); ax /= np.linalg.norm(ax)
        # design intent on a turned part: a boss stands radially or axially (within 5 deg); snap it
        u_ax = np.asarray(res.axis_dir, float); o_ax = np.asarray(res.axis_point, float)
        rv_ = (foot - o_ax) - ((foot - o_ax) @ u_ax) * u_ax; rv_ /= max(np.linalg.norm(rv_), 1e-9)
        c5 = np.cos(np.radians(5.0))
        if abs(ax @ u_ax) > c5:
            ax = u_ax * np.sign(ax @ u_ax)
        elif ax @ rv_ > c5:
            ax = rv_
        else:
            continue
        e1 = np.cross(ax, [1.0, 0, 0]); e1 = e1 if np.linalg.norm(e1) > 0.1 else np.cross(ax, [0, 1.0, 0])
        e1 /= np.linalg.norm(e1); e2 = np.cross(ax, e1)
        D = C[comp] - foot; h = D @ ax; X2 = np.c_[D @ e1, D @ e2]
        wall = np.abs(N[comp] @ ax) < 0.5
        if wall.sum() < 10:
            continue
        cc = np.linalg.lstsq(np.c_[2 * X2[wall], np.ones(wall.sum())], (X2[wall] ** 2).sum(1), rcond=None)[0][:2]
        rho = np.linalg.norm(X2 - cc, axis=1)
        Ro = float(np.percentile(rho[wall], 90))
        inner_w = wall & (rho < 0.75 * Ro)
        Ri = float(np.median(rho[inner_w])) if inner_w.sum() >= 10 else 0.0
        top = float(np.percentile(h, 97))
        if top < 1.0 or Ro < 0.5:
            continue
        centre = foot + cc[0] * e1 + cc[1] * e2
        base = centre - 1.0 * ax
        cyl = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(*map(float, base)), gp_Dir(*map(float, ax))), Ro, top + 1.0).Shape()
        fu = BRepAlgoAPI_Fuse(solid, cyl); fu.Build()
        if not fu.IsDone():
            continue
        ex = TopExp_Explorer(fu.Shape(), TopAbs_SOLID)
        if not ex.More() or U.count(fu.Shape(), TopAbs_SOLID) != 1:
            continue
        s2 = TopoDS.Solid(ex.Current())
        floor = None
        if Ri > 0:
            fl = (rho < 0.8 * Ri) & (np.abs(N[comp] @ ax) > 0.7)
            floor = float(np.median(h[fl])) if fl.sum() >= 3 else None
            if floor is not None and floor < top - 0.3:
                s3 = _cut_cyl(s2, centre + floor * ax, ax, Ri, top - floor + 1.0)
                if s3 is not None:
                    s2 = s3
        if not U.check(s2):
            continue
        # verify on the protrusion's own scan points
        m2 = shape_to_trimesh(s2, lin=0.01, ang=0.1)
        before = float(np.median(dist[comp]))
        after = float(np.median(_on_surface(m2, C[comp])[1]))
        if after >= 0.5 * before:
            continue
        solid = s2
        res.bosses.append({"faces": int(len(comp)), "area_mm2": round(area, 1), "outer_diameter": round(2 * Ro, 3),
                           "inner_diameter": round(2 * Ri, 3) if Ri and floor is not None else None,
                           "height_above_surface": round(top, 3),
                           "inner_depth": round(top - floor, 3) if Ri and floor is not None else None,
                           "axis": ax.round(4).tolist(), "foot": foot.round(3).tolist(),
                           "median_dev_before": round(before, 3), "median_dev_after": round(after, 3)})
    return solid


def _cluster_holes(solid, mesh, tol, res, min_area=5.0):
    """Holes segmentation did not fit: compact clusters of scan surface INSIDE the solid.  A hole axis
    on a turned part is axial or radial (design intent); the wall points are fitted with a circle in
    the plane normal to each candidate axis and the better fit is cut, kept only if it brings the
    cluster's own scan points closer.  Hole-fill caps are checked as for fitted holes."""
    from .deviation import _on_surface
    from .occ_util import shape_to_trimesh
    from OCP.BRepClass3d import BRepClass3d_SolidClassifier
    from OCP.TopAbs import TopAbs_IN
    from OCP.gp import gp_Pnt
    import trimesh
    sm = shape_to_trimesh(solid, lin=0.01, ang=0.1)
    C = np.asarray(mesh.triangles_center); N = mesh.face_normals
    cp, dist, tri = _on_surface(sm, C)
    sgn = np.einsum("ij,ij->i", C - cp, sm.face_normals[tri])
    inside = (sgn < 0) & (dist > max(0.5, 5 * tol))
    if not inside.any():
        return solid
    idx = np.where(inside)[0]
    adj = mesh.face_adjacency
    keep = inside[adj[:, 0]] & inside[adj[:, 1]]
    u = np.asarray(res.axis_dir, float); o = np.asarray(res.axis_point, float)
    diag = float(np.linalg.norm(np.ptp(mesh.vertices, axis=0)))
    for comp in trimesh.graph.connected_components(adj[keep], nodes=idx, min_len=1):
        comp = np.asarray(comp)
        area = float(mesh.area_faces[comp].sum())
        if area < min_area:
            continue
        c0 = C[comp].mean(0)
        rv_ = (c0 - o) - ((c0 - o) @ u) * u; rv_ /= max(np.linalg.norm(rv_), 1e-9)
        best = None
        for name, a in (("axial", u), ("radial", rv_)):
            wall = np.abs(N[comp] @ a) < 0.5
            if wall.sum() < 10:
                continue
            e1 = np.cross(a, [1.0, 0, 0]); e1 = e1 if np.linalg.norm(e1) > 0.1 else np.cross(a, [0, 1.0, 0])
            e1 /= np.linalg.norm(e1); e2 = np.cross(a, e1)
            W = C[comp][wall] - c0; X2 = np.c_[W @ e1, W @ e2]
            cc = np.linalg.lstsq(np.c_[2 * X2, np.ones(len(X2))], (X2 ** 2).sum(1), rcond=None)[0]
            R = float(np.sqrt(max(cc[2] + cc[:2] @ cc[:2], 0)))
            resid = float(np.median(np.abs(np.linalg.norm(X2 - cc[:2], axis=1) - R)))
            # concave: wall normals point to the axis
            p0 = c0 + cc[0] * e1 + cc[1] * e2
            d = C[comp][wall] - p0; radial = d - np.outer(d @ a, a)
            concave = float(np.mean(np.einsum("ij,ij->i", radial, N[comp][wall]) < 0))
            if concave < 0.8 or R <= 0 or R > 0.05 * diag:
                continue
            if best is None or resid < best[0]:
                best = (resid, name, a, p0, R)
        if best is None or best[0] > max(0.15, 3 * tol):
            continue
        _, name, a, p0, R = best
        t = (np.asarray(mesh.vertices)[np.unique(mesh.faces[comp])] - p0) @ a
        t0, t1 = float(t.min()), float(t.max())
        ext = []
        for tt in (t0 - 1.0, t1 + 1.0):
            P = p0 + tt * a
            ext.append(BRepClass3d_SolidClassifier(solid, gp_Pnt(*map(float, P)), 1e-6).State() != TopAbs_IN)
        if not any(ext):
            continue
        e0, e1_ = (3.0 if ext[0] else 0.0), (3.0 if ext[1] else 0.0)
        # blind depth: hole wall data on the closed side (caps across the bore do not count)
        s2 = _cut_cyl(solid, p0 + (t0 - e0) * a, a, R, (t1 + e1_) - (t0 - e0))
        if s2 is None:
            continue
        m2 = shape_to_trimesh(s2, lin=0.01, ang=0.1)
        before = float(np.median(dist[comp])); after = float(np.median(_on_surface(m2, C[comp])[1]))
        if after >= 0.5 * before:
            continue
        solid = s2
        res.holes_cut.append({"region": None, "source": "unfitted scan cluster", "axis_kind": name,
                              "diameter": round(2 * R, 4), "depth": round(t1 - t0, 3), "open_ends": [bool(e0), bool(e1_)],
                              "centre_radius": round(float(np.linalg.norm((p0 - o) - ((p0 - o) @ u) * u)), 3),
                              "angle_deg": None, "median_dev_before": round(before, 3), "median_dev_after": round(after, 3)})
        _fill_caps(mesh, p0, a, R, t0, t1, bool(e0), tol, None, None, res)
    return solid


def _rot(u, ang):
    u = u / np.linalg.norm(u)
    K = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


_KD = {}


def _pattern_angles(P, mesh, tol, res):
    """Angles (deg) about the part axis where a rotated copy of the hole wall points P lies on the scan."""
    from scipy.spatial import cKDTree
    key = id(mesh)
    if key not in _KD:
        S, _ = __import__("trimesh").sample.sample_surface_even(mesh, 300000, seed=1)
        _KD.clear(); _KD[key] = cKDTree(S)
    kd = _KD[key]
    u = np.asarray(res.axis_dir, float); o = np.asarray(res.axis_point, float)
    def med(phi):
        return float(np.median(kd.query((P - o) @ _rot(u, np.radians(phi)).T + o)[0]))
    self_d = med(0.0)                       # sampling floor: the hole matched against itself
    lim = max(1.5 * self_d, self_d + 2.0 * tol)
    angs = np.arange(0.0, 360.0, 0.25)
    m = np.array([med(a) for a in angs])
    out = []
    for i, phi in enumerate(angs):
        if 5.0 <= phi <= 355.0 and m[i] < lim and m[i] <= m[i - 1] and m[i] <= m[(i + 1) % len(angs)]:
            fine = np.arange(phi - 0.25, phi + 0.26, 0.05)
            out.append(float(fine[int(np.argmin([med(f) for f in fine]))]))
    return out


def _cut_cyl(solid, base, a, R, L):
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCP.gp import gp_Ax2, gp_Pnt, gp_Dir
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_SOLID
    from OCP.TopoDS import TopoDS
    from . import occ_util as U
    cyl = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(*map(float, base)), gp_Dir(*map(float, a))), float(R), float(L)).Shape()
    cut = BRepAlgoAPI_Cut(solid, cyl)
    cut.Build()
    if not cut.IsDone():
        return None
    ex = TopExp_Explorer(cut.Shape(), TopAbs_SOLID)
    if not ex.More():
        return None
    s2 = TopoDS.Solid(ex.Current())
    if U.check(s2) and U.count(cut.Shape(), TopAbs_SOLID) == 1:
        return s2
    return None
