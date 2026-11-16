"""Named whole-region auto-exclusion for connected scan areas more than 2 mm from the solid.

Applies to any part.  Each region is listed with its area, % of scan points and reason, is
greyed on the Metrology map with its boundary drawn, and is kept out of alignment refinement
and the deviation gate.  The total excluded % is reported, and flagged when it exceeds 5 %.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import ConvexHull, QhullError, cKDTree

THRESHOLD_MM = 2.0
OVER_5PCT = 5.0


@dataclass
class Config:
    threshold_mm: float = THRESHOLD_MM
    link: float = 2.0            # mm, neighbour distance when growing a region
    min_points: int = 25
    min_area: float = 4.0        # mm^2
    over_pct: float = OVER_5PCT


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


def _area(P, faces=None, face_areas=None, vmask=None, cell=0.7):
    """Surface area of a region: sum of mesh faces that touch it, else a planar grid estimate."""
    if faces is not None and face_areas is not None and vmask is not None and len(faces):
        touched = np.asarray(vmask, bool)[np.asarray(faces)]
        return float(np.asarray(face_areas)[touched.any(1)].sum())
    xy = np.asarray(P)[:, :2]
    g = np.unique(np.floor(xy / cell).astype(int), axis=0)
    return float(len(g) * cell * cell)


def _hull_points(P, max_pts=48):
    """World-frame 3D convex-hull vertices (or the AABB corners if the hull is degenerate)."""
    P = np.asarray(P, float)
    if len(P) < 3:
        return P.tolist()
    try:
        # hull in the plane of greatest extent if the set is nearly flat
        if np.linalg.matrix_rank(P - P.mean(0), tol=1e-6) < 3:
            # 2D hull on the two largest axes
            span = np.ptp(P, axis=0)
            axes = np.argsort(span)[::-1][:2]
            H = ConvexHull(P[:, axes])
            pts = P[H.vertices]
        else:
            H = ConvexHull(P)
            pts = P[H.vertices]
    except (QhullError, ValueError):
        lo, hi = P.min(0), P.max(0)
        corners = np.array([[lo[0], lo[1], lo[2]], [lo[0], lo[1], hi[2]],
                            [lo[0], hi[1], lo[2]], [lo[0], hi[1], hi[2]],
                            [hi[0], lo[1], lo[2]], [hi[0], lo[1], hi[2]],
                            [hi[0], hi[1], lo[2]], [hi[0], hi[1], hi[2]]])
        return corners.tolist()
    if len(pts) > max_pts:
        # keep an even subsample around the hull
        step = int(np.ceil(len(pts) / max_pts))
        pts = pts[::step]
    return np.round(pts, 4).tolist()


def detect(P, dv, *, mesh=None, already_excluded=None, cfg=Config()):
    """Connected components of |dv| > threshold, not already excluded.

    Returns (report_dict, vertex_mask).  `P` and `dv` are in the same vertex order.
    """
    cfg = cfg if isinstance(cfg, Config) else Config(**(cfg or {}))
    P = np.asarray(P, float)
    dv = np.asarray(dv, float)
    n = len(P)
    already = np.zeros(n, bool) if already_excluded is None else np.asarray(already_excluded, bool)
    cand = np.isfinite(dv) & (np.abs(dv) > cfg.threshold_mm) & ~already
    mask = np.zeros(n, bool)
    regions = []
    faces = face_areas = None
    if mesh is not None and len(getattr(mesh, "faces", [])):
        faces = np.asarray(mesh.faces)
        fa = getattr(mesh, "area_faces", None)
        face_areas = np.asarray(fa) if fa is not None else np.zeros(0)
        if face_areas.size != len(faces):
            face_areas = None
            try:
                face_areas = trimesh_face_areas(mesh)
            except Exception:  # noqa: BLE001
                face_areas = None
    if cand.sum() >= cfg.min_points:
        idx = np.where(cand)[0]
        lab, nc = _components(P[idx], cfg.link)
        k = 0
        for c in range(nc):
            q = idx[lab == c]
            if len(q) < cfg.min_points:
                continue
            vmask = np.zeros(n, bool); vmask[q] = True
            area = _area(P[q], faces, face_areas, vmask)
            if area < cfg.min_area:
                continue
            k += 1
            pct = 100.0 * len(q) / max(n, 1)
            lo, hi = P[q].min(0), P[q].max(0)
            depth = float(np.max(np.abs(dv[q])))
            sign = "outside" if float(np.median(dv[q])) > 0 else "inside"
            hull = _hull_points(P[q])
            regions.append({
                "id": f"d{cfg.threshold_mm:g}_{k}",
                "label": (f"region {k}: |d| > {cfg.threshold_mm:g} mm "
                          f"({area:.1f} mm², {pct:.2f} % of the scan, max |d| {depth:.2f} mm, scan {sign} the solid)"),
                "reason": (f"connected scan region deviating more than {cfg.threshold_mm:g} mm from the solid; "
                           "not modelled and not inspected"),
                "n_points": int(len(q)),
                "pct_of_scan": round(float(pct), 3),
                "area_mm2": round(float(area), 1),
                "depth_max_mm": round(depth, 3),
                "sign": sign,
                "boundary": {"type": "box",
                             "min": [round(float(x), 3) for x in lo],
                             "max": [round(float(x), 3) for x in hi],
                             "hull": hull},
                "outline": hull,
                "_ix": q.tolist(),
            })
            mask[q] = True
    pct_total = 100.0 * float(mask.sum()) / max(n, 1)
    report = {
        "threshold_mm": cfg.threshold_mm,
        "regions": regions,
        "n_regions": len(regions),
        "n_points": int(mask.sum()),
        "pct_of_scan": round(float(pct_total), 3),
        "over_5pct": bool(pct_total > cfg.over_pct),
        "over_5pct_flag": (f"excluded {pct_total:.2f} % of the scan, above the {cfg.over_pct:g} % limit"
                           if pct_total > cfg.over_pct else None),
        "reason": f"connected scan regions with |deviation from the solid| > {cfg.threshold_mm:g} mm",
    }
    return report, mask


def trimesh_face_areas(mesh):
    v = np.asarray(mesh.vertices)
    f = np.asarray(mesh.faces)
    e1 = v[f[:, 1]] - v[f[:, 0]]
    e2 = v[f[:, 2]] - v[f[:, 0]]
    return 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)


def outlines(report):
    """List of Nx3 hulls for drawing region boundaries on the deviation map."""
    return [np.asarray(r["outline"], float) for r in (report or {}).get("regions") or [] if r.get("outline")]


def public_report(report):
    """Drop per-vertex index lists so the report is safe to write to JSON."""
    out = dict(report or {})
    regs = []
    for r in out.get("regions") or []:
        regs.append({k: v for k, v in r.items() if not k.startswith("_")})
    out["regions"] = regs
    return out


def apply_to_build(rep, mesh, dv, v_excluded, dev=None):
    """Detect 2 mm regions, merge into the vertex mask, update conversion-report stats (cheap: no
    re-query of the solid).  Returns (new_vertex_mask, region_mask, public_report)."""
    ev = np.asarray(v_excluded, bool).copy()
    report, m2 = detect(np.asarray(mesh.vertices), np.asarray(dv), mesh=mesh, already_excluded=ev)
    if m2.any():
        ev = ev | m2
        if dev is not None:
            from .deviation import _stats
            keep = ~ev
            if keep.sum() > 10:
                dev["mesh_vertices_to_solid_excl"] = _stats(np.asarray(dv)[keep])
                gate = dev["mesh_vertices_to_solid_excl"]
                sm = dev.setdefault("summary", {})
                sm["vertex_to_solid_rms_gate_mm"] = round(gate["rms"], 5)
                sm["vertex_to_solid_p95_gate_mm"] = round(gate["p95"], 5)
                sm["gate_field"] = "mesh_vertices_to_solid_excl"
    pub = public_report(report)
    pct_all = 100.0 * float(ev.sum()) / max(len(ev), 1)
    pub["pct_excluded_total"] = round(pct_all, 3)
    pub["over_5pct_total"] = bool(pct_all > OVER_5PCT)
    if pct_all > OVER_5PCT and not pub.get("over_5pct_flag"):
        pub["over_5pct_flag"] = f"excluded {pct_all:.2f} % of the scan, above the {OVER_5PCT:g} % limit"
    if rep is not None:
        rep["auto_exclusions_2mm"] = pub
    return ev, m2, pub
