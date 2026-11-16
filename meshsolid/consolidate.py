"""Pre-build region consolidation for real scans.

On real scans many regions get no trimmed face because the segmentation left (a) the same surface split
into several regions that are not adjacent (e.g. a bore interrupted by a groove), (b) small slivers
along edges fitted with a degenerate primitive (a near-flat strip as a 1000 mm sphere or a 200 mm
cylinder), (c) small fragments of a badly scanned surface. Each such region needs its own trimmed piece,
and the piece test fails (its bounded surface covers mostly other regions' faces).

This stage, run after snapping/blends and before the build:
  1. merges regions that lie on the same surface (same kind and parameters within tol, and each one's
     points fit the other's primitive at rms <= tol), adjacent or not;
  2. replaces a degenerate primitive (sphere/cylinder radius > half the part diagonal, torus with
     major < minor) by a plane when a plane fits at rms <= 1.5 tol;
  3. absorbs a region into the neighbour whose primitive explains its points within the pipeline's
     acceptance limits (rms <= 1.5 tol, p95 <= 3 tol); large regions only at rms <= tol;
  4. un-assigns small regions (< small_frac of the area) whose primitive is still degenerate and that no
     neighbour explains: their faces become unassigned, the neighbours' trimmed faces extend over them,
     and the deviation check still measures them.
Nothing here can make a pass out of a bad fit: every move keeps the moved faces within the acceptance
limits of the surface they end up on, or leaves them unassigned and measured."""
import numpy as np

from . import fit as F


def _pts(g, region_points, faces):
    P, _N = region_points(g, faces)
    return P


def _fits(prim, P, rms_lim, p95_lim):
    d = np.abs(F.signed_distance(prim, P))
    return float(np.sqrt(np.mean(d * d))) <= rms_lim and float(np.percentile(d, 95)) <= p95_lim


def _degenerate(prim, diag):
    q, k = prim.params, prim.kind
    if k in ("sphere", "cylinder"):
        return float(q.get("radius", 0.0)) > 0.5 * diag
    if k == "torus":
        return float(q["major_radius"]) < float(q["minor_radius"])
    return False


def _same_surface(a, b, tol):
    if a.kind != b.kind:
        return False
    p, q = a.params, b.params
    if a.kind == "plane":
        na, nb = np.asarray(p["normal"]), np.asarray(q["normal"])
        if abs(na @ nb) < np.cos(np.radians(0.5)):
            return False
        return abs((np.asarray(q["point"]) - np.asarray(p["point"])) @ na) < tol
    if a.kind == "cylinder":
        da, db = np.asarray(p["axis"], float), np.asarray(q["axis"], float)
        da, db = da / np.linalg.norm(da), db / np.linalg.norm(db)
        if abs(da @ db) < np.cos(np.radians(0.5)) or abs(p["radius"] - q["radius"]) > tol:
            return False
        w = np.asarray(q["point"]) - np.asarray(p["point"])
        return float(np.linalg.norm(w - (w @ da) * da)) < tol
    if a.kind == "sphere":
        return abs(p["radius"] - q["radius"]) < tol and np.linalg.norm(np.asarray(p["center"]) - np.asarray(q["center"])) < tol
    return False


def consolidate(g, regions, labels, tol, region_points, small_frac=0.005, protected=()):
    """protected: region ids never moved or removed (thread zones, blend supports)."""
    log = []
    protected = set(protected)
    total = float(sum(r.area for r in regions)) or 1.0
    diag = float(np.linalg.norm(np.ptp(g.V, axis=0)))
    nb_faces = g.adj
    by_id = {r.id: r for r in regions}

    def neighbours(rid):
        m = labels[nb_faces[:, 0]] == rid
        s = set(labels[nb_faces[m, 1]].tolist())
        m = labels[nb_faces[:, 1]] == rid
        s |= set(labels[nb_faces[m, 0]].tolist())
        s.discard(rid); s.discard(-1)
        return s

    def merge(src, dst, why):
        labels[src.faces] = dst.id
        dst.faces = np.concatenate([dst.faces, src.faces])
        dst.area += src.area
        del by_id[src.id]
        log.append({"op": why, "region": src.id, "into": dst.id, "area": round(src.area, 2)})

    # 2. degenerate primitives -> plane
    for r in list(by_id.values()):
        if r.id not in protected and r.primitive is not None and _degenerate(r.primitive, diag):
            P, N = region_points(g, r.faces)
            q = F.fit_plane(P, N)
            if _fits(q, P, 1.5 * tol, 3.0 * tol):
                log.append({"op": "degenerate_to_plane", "region": r.id, "was": r.primitive.kind,
                            "radius": round(float(r.primitive.params.get("radius", r.primitive.params.get("major_radius", 0))), 3)})
                r.primitive = q
    # 1. same surface (adjacent or not), larger keeps its primitive
    changed = True
    while changed:
        changed = False
        rs = sorted(by_id.values(), key=lambda r: -r.area)
        for i, a in enumerate(rs):
            if a.id not in by_id or a.id in protected or a.primitive is None or a.primitive.kind not in ("plane", "cylinder", "sphere"):
                continue
            for b in rs[i + 1:]:
                if b.id not in by_id or b.id in protected or b.primitive is None or b.primitive is a.primitive:
                    continue
                if _same_surface(a.primitive, b.primitive, tol) and _fits(a.primitive, _pts(g, region_points, b.faces), tol, 2.5 * tol):
                    merge(b, a, "same_surface")
                    changed = True
    # 3. absorb into a neighbour that explains the region within the acceptance limits
    changed = True
    while changed:
        changed = False
        for r in sorted(by_id.values(), key=lambda r: r.area):
            if r.id not in by_id or r.id in protected or r.primitive is None or r.primitive.kind == "pipe":
                continue
            small = r.area < small_frac * total
            P = _pts(g, region_points, r.faces)
            cands = []
            for j in neighbours(r.id):
                o = by_id.get(j)
                if o is None or o.id in protected or o.primitive is None or o.area <= r.area or o.primitive.kind == "pipe":
                    continue
                lim = (1.5 * tol, 3.0 * tol) if small else (tol, 2.5 * tol)
                if _fits(o.primitive, P, *lim):
                    d = np.abs(F.signed_distance(o.primitive, P))
                    cands.append((float(np.sqrt(np.mean(d * d))), o))
            if cands:
                merge(r, min(cands, key=lambda t: t[0])[1], "absorbed_small" if small else "absorbed")
                changed = True
    # 4. small + degenerate + unexplained -> unassigned (measured by the deviation check, no face needed)
    for r in list(by_id.values()):
        if r.id not in protected and r.area < small_frac * total and r.primitive is not None and _degenerate(r.primitive, diag):
            labels[r.faces] = -1
            del by_id[r.id]
            log.append({"op": "unassigned", "region": r.id, "kind": r.primitive.kind, "area": round(r.area, 2)})
    out = [r for r in regions if r.id in by_id]
    return out, labels, log
