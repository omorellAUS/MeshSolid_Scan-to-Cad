"""Balance-cut detection on rotor parts (impellers, wheels).

Balance cuts are local material-removal scallops ground into one of the two balance planes of a
rotor: (1) the outer band of the BACK face, (2) the inner band of the FRONT (hub/eye) face around the
bore.  Seen from the scan they are compact patches where the scan lies inside the nominal surface.
They are not design geometry: we mask them from metrology and build to the nominal surface,
listing them separately.

Frame: rotor axis = z through the origin; points/normals in that frame.
"""
from dataclasses import dataclass, asdict
import numpy as np
from scipy.spatial import cKDTree


@dataclass
class BalanceConfig:
    back_band: tuple = (0.70, 1.00)     # fraction of R_out searched on the back face
    front_band: tuple = (None, 1.00)    # (inner r, fraction of the front-face outer radius); None = bore + 1 mm
    plane_tol: float = 2.0              # mm, axial distance from the end-face plane
    axial_normal: float = 0.85          # |n.z| of the nominal surface
    depth_min: float = 0.15             # mm, scan inside the nominal by more than this (vs ring baseline)
    link: float = 1.2                   # mm, neighbour distance when growing patches
    area_min: float = 8.0               # mm^2
    max_span_deg: float = 60.0          # compact: angular extent
    min_width: float = 1.0              # mm, radial width (rejects thin rings/strips along a face edge)
    max_aspect: float = 12.0            # arc length / radial width
    max_radial_aspect: float = 4.0      # radial width / arc length (rejects radial blade-root streaks)
    dilate: float = 1.0                 # mm, mask margin around a cut


@dataclass
class Cut:
    plane: str
    angle_deg: float
    radius_mm: float
    z_mm: float
    depth_mm: float
    depth_max_mm: float
    area_mm2: float
    span_deg: float
    radial_width_mm: float
    n_points: int


def _ring_residual(r, ds, step=1.0):
    """Deviation minus the median of its 1-mm radial ring (removes a uniform face offset)."""
    b = np.floor(r / step).astype(int)
    res = np.empty_like(ds)
    for k in np.unique(b):
        m = b == k
        res[m] = ds[m] - np.median(ds[m])
    return res


def _area(xy, cell=0.7):
    g = np.unique(np.floor(xy / cell).astype(int), axis=0)
    return float(len(g) * cell * cell)


def end_faces(Q, N, r_bore, R_out):
    """Back plane = axial end whose outer annulus carries axial-normal area; front = the other end's
    axial-normal face nearest the front end outside the bore."""
    r = np.hypot(Q[:, 0], Q[:, 1])
    ax = np.abs(N[:, 2]) > 0.9
    rim = (r > 0.8 * R_out) & ax
    lo, hi = Q[:, 2].min(), Q[:, 2].max()
    zr = Q[rim, 2]
    back_hi = np.median(zr) > (lo + hi) / 2
    z_back = float(np.percentile(zr, 90 if back_hi else 10))
    sgn = 1.0 if back_hi else -1.0          # +1: back face at high z
    fr = ax & (r > r_bore + 1.0) & ((Q[:, 2] - (lo + hi) / 2) * sgn < 0)
    zf = Q[fr, 2]
    # front hub face: the axial-normal area within 3 mm of the front-most axial face
    zend = float(zf.min() if back_hi else zf.max())
    near = fr & (np.abs(Q[:, 2] - zend) < 3.0)
    z_front = float(np.median(Q[near, 2]))
    r_front_out = float(np.percentile(r[near], 99))
    nearb = ax & (np.abs(Q[:, 2] - z_back) < 1.0)
    r_back_in = float(np.percentile(r[nearb], 1))
    return dict(z_back=z_back, z_front=z_front, back_sign=sgn, r_front_out=r_front_out, r_back_in=r_back_in)


def detect(P, ds, Nn, faces, r_bore, R_out, cfg=BalanceConfig(), blades=None):
    """P scan points (model frame), ds signed deviation (+ = scan outside), Nn nominal normal at the
    nearest model point.  Returns (cuts, mask of points belonging to cuts incl. margin)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    r = np.hypot(P[:, 0], P[:, 1]); z = P[:, 2]
    axn = np.abs(Nn[:, 2]) > cfg.axial_normal
    sgn = faces["back_sign"]
    zones = {
        "back": axn & (np.abs(z - faces["z_back"]) < cfg.plane_tol) & (Nn[:, 2] * sgn > 0)
        & (r >= cfg.back_band[0] * R_out) & (r <= cfg.back_band[1] * R_out + 1.0),
        "front": axn & (np.abs(z - faces["z_front"]) < cfg.plane_tol) & (Nn[:, 2] * sgn < 0)
        & (r >= (cfg.front_band[0] if cfg.front_band[0] is not None else r_bore + 1.0))
        & (r <= cfg.front_band[1] * faces["r_front_out"]),
    }
    cuts, mask = [], np.zeros(len(P), bool)
    diag = {}
    for plane, zm in zones.items():
        idx = np.where(zm)[0]
        diag[plane] = {"points": int(len(idx))}
        if len(idx) < 50:
            continue
        # baseline = per-ring offset + a face tilt (runout); cuts are what is left
        res = _ring_residual(r[idx], ds[idx])
        X = np.c_[P[idx, 0], P[idx, 1], np.ones(len(idx))]
        for _ in range(3):
            w = np.abs(res) < max(0.1, 2.5 * np.median(np.abs(res)))
            coef = np.linalg.lstsq(X[w], res[w], rcond=None)[0]
            res = _ring_residual(r[idx], ds[idx] - X @ coef)
        diag[plane].update({"r_range": [round(float(r[idx].min()), 2), round(float(r[idx].max()), 2)],
                            "residual_p1": round(float(np.percentile(res, 1)), 3),
                            "candidates": int((res < -cfg.depth_min).sum())})
        sel = res < -cfg.depth_min
        cand = idx[sel]; resc = res[sel]
        if len(cand) == 0:
            continue
        pairs = cKDTree(P[cand]).query_pairs(cfg.link, output_type="ndarray")
        n = len(cand)
        A = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n)) if len(pairs) else coo_matrix((n, n))
        nc, lab = connected_components(A, directed=False)
        found = []
        for c in range(nc):
            k = lab == c
            if k.sum() < 5:
                continue
            Pc = P[cand[k]]
            a = _area(Pc[:, :2])
            if a < cfg.area_min:
                continue
            th = np.degrees(np.arctan2(Pc[:, 1], Pc[:, 0]))
            mth = np.degrees(np.arctan2(np.sin(np.radians(th)).mean(), np.cos(np.radians(th)).mean()))
            span = float(np.ptp((th - mth + 180) % 360 - 180))
            rr = np.hypot(Pc[:, 0], Pc[:, 1])
            width = float(np.percentile(rr, 98) - np.percentile(rr, 2))
            arc = np.radians(span) * float(np.median(rr))
            if span > cfg.max_span_deg or width < cfg.min_width or arc > cfg.max_aspect * max(width, 1e-6) \
                    or width > cfg.max_radial_aspect * max(arc, 1e-6):
                continue
            found.append((k, Cut(plane, round(float(mth), 1), round(float(np.median(np.hypot(Pc[:, 0], Pc[:, 1]))), 2),
                                 round(float(np.median(Pc[:, 2])), 2), round(float(-np.percentile(resc[k], 10)), 3),
                                 round(float(-resc[k].min()), 3), round(a, 1), round(span, 1), round(width, 2), int(k.sum()))))
        # many equal patches = an N-fold feature (blade roots etc.), not balance cuts
        if len(found) >= 6:
            ang = np.sort([f[1].angle_deg for f in found])
            gaps = np.diff(np.r_[ang, ang[0] + 360])
            areas = np.array([f[1].area_mm2 for f in found])
            if gaps.std() < 0.2 * gaps.mean() and areas.std() < 0.3 * areas.mean():
                found = []
        for k, cut in found:
            cuts.append(cut)
            near = cKDTree(P[cand[k]]).query(P[idx], distance_upper_bound=cfg.dilate)[0] < np.inf
            mask[idx[near]] = True
    detect.last_diag = diag
    return cuts, mask


def scan_only(P, ds, faces, zlo, zhi, r_bore, margin=0.5, far=1.0):
    """Scan geometry the part does not have: material beyond the model's axial extent (stub/nut),
    inside the bore, or a flat cap across the back plane inside the back-face annulus."""
    r = np.hypot(P[:, 0], P[:, 1]); z = P[:, 2]
    stub = (z < zlo - margin) | (z > zhi + margin) | ((r < r_bore - margin) & (np.abs(ds) > far))
    cap = (np.abs(z - faces["z_back"]) < 1.5) & (r < faces["r_back_in"] - 1.0) & (np.abs(ds) > far)
    return stub, cap


def cuts_json(cuts):
    out = {"count": len(cuts), "by_plane": {}}
    for pl in ("back", "front"):
        cs = [asdict(c) for c in cuts if c.plane == pl]
        out["by_plane"][pl] = {"count": len(cs), "cuts": sorted(cs, key=lambda c: c["angle_deg"])}
    return out


def _frame(u):
    u = np.asarray(u, float); u = u / np.linalg.norm(u)
    e1 = np.cross(u, [1.0, 0, 0])
    if np.linalg.norm(e1) < 0.1:
        e1 = np.cross(u, [0, 1.0, 0])
    e1 /= np.linalg.norm(e1)
    return np.stack([e1, np.cross(u, e1), u])          # rows: x, y, z(axis)


def detect_on_build(mesh, solid_mesh, dv, axis_point, axis_dir, cfg=BalanceConfig(), n_model=400000):
    """Balance cuts of a scanned rotor against the BUILT solid (the solid is fitted to the nominal
    surfaces; cuts are too small to move a robust fit).  Returns (json, mesh-face mask of the cuts)."""
    import trimesh
    Rm = _frame(axis_dir); o = np.asarray(axis_point, float)
    Q, fi = trimesh.sample.sample_surface(solid_mesh, n_model, seed=4)
    Q = (Q - o) @ Rm.T; N = solid_mesh.face_normals[fi] @ Rm.T
    P = (np.asarray(mesh.vertices) - o) @ Rm.T
    Mr = np.hypot(Q[:, 0], Q[:, 1])
    nin = -np.einsum("ij,ij->i", N, np.c_[Q[:, 0], Q[:, 1], 0 * Mr] / np.maximum(Mr, 1e-9)[:, None])
    r_bore = float(np.percentile(Mr[nin > 0.9], 5)) if (nin > 0.9).sum() > 50 else 0.0
    R_out = float(np.percentile(Mr, 99.9))
    faces = end_faces(Q, N, r_bore, R_out)
    j = cKDTree(Q).query(P)[1]
    cuts, vmask = detect(P, np.asarray(dv, float), N[j], faces, r_bore, R_out, cfg)
    fmask = vmask[np.asarray(mesh.faces)].all(1)
    out = cuts_json(cuts)
    out.update({"search_zones": getattr(detect, "last_diag", {}), "config": asdict(cfg),
                "faces_axis_frame": {k: round(float(v), 3) for k, v in faces.items()},
                "frame": "angles about the part axis from the axis-frame x direction",
                "excluded_mesh_faces": int(fmask.sum()),
                "modelled_as": "not modelled: the solid is built to the nominal surfaces; cuts are listed here"})
    return out, fmask
