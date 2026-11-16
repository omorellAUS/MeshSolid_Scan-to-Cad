"""Keyway recognition (M9): shaft keyseats and hub keyways from the fitted regions, snapped to
DIN 6885-1 / ANSI B17.1 (meshsolid.standards).

A keyway on a cylinder (axis u, radius R) is two parallel planes with opposite outward normals n and -n,
both parallel to u and symmetric about a plane through the axis (the side walls, spacing = width b),
and a floor plane parallel to u whose normal is along e = the radial direction between the walls:
  shaft keyseat  convex cylinder, floor at x0 < R from the axis, depth t1 = R - x0 (measured at the
                 keyway centre line from the circle, the DIN 6885 t1 convention)
  hub keyway     concave cylinder (bore), floor at x0 > R, depth t2 = x0 - R
Length from the side walls' extent along the axis; ends: 'round' when cylinders of radius b/2 with radial
axes close the keyseat (end-milled), 'through' when the walls reach the part's end faces, else 'square'.
Geometry is built by the regular builders (the walls and floor are planes); this module recognises
and reports the design values."""
from __future__ import annotations

import numpy as np


def _outward(mesh, r):
    """Mean face normal of a region (scan normals point out of the material)."""
    n = np.asarray(mesh.face_normals)[r.faces] * np.asarray(mesh.area_faces)[r.faces][:, None]
    v = n.sum(0)
    return v / (np.linalg.norm(v) or 1.0)


def detect_keyways(mesh, regions, tol):
    planes = [r for r in regions if r.primitive is not None and r.primitive.kind == "plane"]
    cyls = [r for r in regions if r.primitive is not None and r.primitive.kind == "cylinder"]
    C = np.asarray(mesh.triangles_center)
    zmin_all, out, used = None, [], set()
    for cy in sorted(cyls, key=lambda r: -r.area):
        q = cy.primitive.params
        u = np.asarray(q["axis"], float); u /= np.linalg.norm(u)
        a0 = np.asarray(q["point"], float); R = float(q["radius"])
        nc = _outward(mesh, cy)
        dc = C[cy.faces] - a0; dc -= np.outer(dc @ u, u)
        convex = float(np.mean(np.einsum("ij,j->i", dc, nc))) > 0      # outward normal points away from axis
        nr = [] if not convex else None
        zc = (C[cy.faces] - a0) @ u
        z_lo, z_hi = float(zc.min()), float(zc.max())
        side = []
        for p in planes:
            n = _outward(mesh, p)
            if abs(n @ u) > 0.05:
                continue
            P = C[p.faces] - a0
            rad = np.linalg.norm(P - np.outer(P @ u, u), axis=1)
            if np.median(rad) > R + max(0.5, 0.5 * R) or np.median(rad) < 0.2 * R:
                continue
            side.append((p, n, P))
        for i, (p1, n1, P1) in enumerate(side):
            for p2, n2, P2 in side[i + 1:]:
                if p1.id in used or p2.id in used or n1 @ n2 > -0.995:
                    continue
                o1, o2 = float(np.median(P1 @ n1)), float(np.median(P2 @ n1))
                b = abs(o1 - o2)
                if not (0.05 * R < b < 1.2 * R) or abs(0.5 * (o1 + o2)) > max(0.15 * b, 3 * tol):
                    continue
                e = np.cross(u, n1); e /= np.linalg.norm(e)
                if float(np.median(np.r_[P1 @ e, P2 @ e])) < 0:
                    e = -e
                floor, x0 = None, None
                for f in planes:
                    if f.id in (p1.id, p2.id):
                        continue
                    nf = _outward(mesh, f)
                    if abs(nf @ e) < 0.995:
                        continue
                    Pf = C[f.faces] - a0
                    if np.abs(Pf @ n1).max() > 0.6 * b + 3 * tol:
                        continue
                    xf = float(np.median(Pf @ e))
                    if convex and 0.3 * R < xf < R and nf @ e > 0:
                        floor, x0 = f, xf
                    if not convex and R < xf < 2 * R and nf @ e < 0:
                        floor, x0 = f, xf
                    if floor is not None:
                        break
                if floor is None:
                    continue
                V = np.asarray(mesh.vertices) - a0
                zw = np.r_[V[np.unique(mesh.faces[p1.faces])] @ u, V[np.unique(mesh.faces[p2.faces])] @ u]
                za, zb = float(zw.min()), float(zw.max())
                # square ends: end-wall planes (normal along the axis) inside the slot
                for f in planes:
                    nf = _outward(mesh, f)
                    if abs(nf @ u) < 0.995:
                        continue
                    Pf = C[f.faces] - a0
                    xs, ys = Pf @ e, Pf @ n1
                    lo_x, hi_x = (x0, R) if convex else (R, x0)
                    if np.median(np.abs(ys)) < 0.5 * b and lo_x - 3 * tol < np.median(xs) < hi_x + 3 * tol:
                        zf = float(np.median(Pf @ u))
                        if abs(zf - za) < 1.0:
                            za = zf
                        elif abs(zf - zb) < 1.0:
                            zb = zf
                depth = (R - x0) if convex else (x0 - R)
                ends = "square"
                rounds = []
                for c2 in cyls:
                    if c2 is cy:
                        continue
                    q2 = c2.primitive.params
                    if abs(abs(np.asarray(q2["axis"]) @ e) - 1) < 0.01 and abs(float(q2["radius"]) - b / 2) < max(0.1 * b, 3 * tol):
                        zc2 = float((np.asarray(q2["point"]) - a0) @ u)
                        rounds.append(zc2)
                if len(rounds) >= 1:
                    ends = "round"
                    za = min(za, min(rounds) - b / 2); zb = max(zb, max(rounds) + b / 2)
                Pall = np.asarray(mesh.vertices) - a0
                zv = Pall @ u
                lim = max(3 * tol, 0.05 * b)                      # the scan rounds the wall/end-face edges
                if za < zv.min() + lim or zb > zv.max() - lim:
                    ends = "through" if (za < zv.min() + lim and zb > zv.max() - lim) else "open one end"
                    za, zb = (max(za, float(zv.min())) if za >= zv.min() + lim else float(zv.min()),
                              min(zb, float(zv.max())) if zb <= zv.max() - lim else float(zv.max()))
                from .standards import key_for_shaft
                cands = key_for_shaft(2 * R, b)
                best = cands[0] if cands and cands[0]["width_misfit_mm"] <= max(0.1, 2 * tol) else None
                kw = {"kind": "shaft keyseat" if convex else "hub keyway", "regions": [int(p1.id), int(p2.id), int(floor.id)],
                      "cylinder_region": int(cy.id), "diameter": round(2 * R, 4), "width_b": round(b, 4),
                      ("depth_t1" if convex else "depth_t2"): round(depth, 4), "length": round(zb - za, 4), "ends": ends,
                      "z_range_axis_frame": [round(za, 3), round(zb, 3)], "axis_point": a0.round(4).tolist(),
                      "axis_dir": u.round(6).tolist(), "radial_dir": e.round(6).tolist()}
                if best:
                    nom = best["t1"] if convex else best["t2"]
                    kw["standard"] = {"standard": best["standard"], "designation": best["designation"], "b": best["b"],
                                      "h": best["h"], "depth_nominal": nom,
                                      "depth_misfit_mm": None if nom is None else round(abs(nom - depth), 4),
                                      **({"note": best["note"]} if "note" in best else {})}
                else:
                    kw["standard"] = None
                    kw["standard_candidates"] = cands[:3]
                out.append(kw)
                used |= {p1.id, p2.id}
    return out
