"""Rim slots: flat-floored pockets in an end face of a rotor, open to the rim (STG1: 3-4 slots, floors about
2.85 mm into the back face, r 76-83 mm, over 196-320 deg).  Unlike balance scallops they are design geometry
(flat floor parallel to the end face, straight parallel walls), so they are detected from the scan and CUT
into the solid.

Frame: rotor axis = z.  Points P and scan normals N in that frame.  An end face is the plane z = z0 with the
material on the side s (s = +1: material at z > z0).  A floor point lies parallel to the end face, inside the
material by depth d = s (z - z0) in [depth_min, depth_max], with its normal facing out of the material like
the end face (s n_z < -cos), in the rim band.

Each slot: centre angle, width between parallel walls (tangential extent of the floor), inner end radius,
depth, floor rms.  A slot is confident when it has a flat floor (rms <= floor_rms_max), enough points,
is open to the rim and is at least min_width wide.  The 'auto' flag cuts each confident slot
and leaves non-confident patches listed but uncut; 'on' cuts every detected slot."""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class RimSlotConfig:
    band: tuple = (0.80, 1.02)         # fraction of R searched
    depth_min: float = 0.4             # mm into the material
    depth_max: float = 8.0
    normal_cos: float = 0.95
    link: float = 1.5                  # mm, neighbour distance when growing floor patches
    min_points: int = 30
    min_area: float = 4.0              # mm^2 (from the floor points' tangential x radial extent)
    min_width: float = 1.0
    floor_rms_max: float = 0.05        # mm, flat floor
    rim_open_tol: float = 1.5          # mm: floor reaches within this of R -> open to the rim
    end_plane_band: float = 0.3        # mm: end-face points used to confirm the plane z0
    max_span_deg: float = 45.0         # a floor patch wider than this is a rebate/step ring, not a slot


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


def detect_end(P, N, R, z0, s, cfg=RimSlotConfig()):
    """Slots in the end face z = z0 (material side s).  Returns (slots, mask of floor points)."""
    P = np.asarray(P, float); N = np.asarray(N, float)
    r = np.hypot(P[:, 0], P[:, 1]); d = s * (P[:, 2] - z0)
    k = ((r >= cfg.band[0] * R) & (r <= cfg.band[1] * R) & (d >= cfg.depth_min) & (d <= cfg.depth_max)
         & (s * N[:, 2] < -cfg.normal_cos))
    idx = np.where(k)[0]
    slots, mask = [], np.zeros(len(P), bool)
    if len(idx) < cfg.min_points:
        return slots, mask
    # patches: grow in the plane, then split by floor depth (two pockets of different depth stay apart)
    X = np.c_[P[idx, 0], P[idx, 1], 10.0 * d[idx]]
    lab, n = _components(X, cfg.link)
    for c in range(n):
        q = idx[lab == c]
        if len(q) < cfg.min_points:
            continue
        th = np.arctan2(P[q, 1], P[q, 0])
        thc = float(np.angle(np.mean(np.exp(1j * th))))
        dth = np.degrees(np.angle(np.exp(1j * (th - thc))))
        span = float(np.percentile(dth, 99.5) - np.percentile(dth, 0.5))
        if span > cfg.max_span_deg:            # annular rebate / step at the rim: turned geometry, not a slot
            continue
        xr = P[q, 0] * math.cos(thc) + P[q, 1] * math.sin(thc)          # radial along the slot centre line
        yt = -P[q, 0] * math.sin(thc) + P[q, 1] * math.cos(thc)         # tangential
        y1, y2 = np.percentile(yt, [0.5, 99.5]); x1, x2 = np.percentile(xr, [0.5, 99.5])
        width = float(y2 - y1); yc = float(0.5 * (y1 + y2))
        dep = float(np.median(d[q])); frms = float(np.sqrt(np.mean((d[q] - dep) ** 2)))
        # walls and inner end from the scan's wall faces between the end face and the floor (the floor
        # points stop short of the walls where corner normals are blended)
        cth, sth = math.cos(thc), math.sin(thc)
        xa = P[:, 0] * cth + P[:, 1] * sth; ya = -P[:, 0] * sth + P[:, 1] * cth
        nx = N[:, 0] * cth + N[:, 1] * sth; ny = -N[:, 0] * sth + N[:, 1] * cth
        band = (d > 0.15 * dep) & (d < 0.85 * dep) & (xa > x1 - 2.0) & (xa < R + 1.0) & (np.abs(ya - yc) < 0.5 * width + 3.0)
        wl = band & (ny > cfg.normal_cos) & (ya < yc); wr = band & (ny < -cfg.normal_cos) & (ya > yc)
        wall_fit = False
        if wl.sum() >= 5 and wr.sum() >= 5:
            yl, yr = float(np.median(ya[wl])), float(np.median(ya[wr]))
            width, yc, wall_fit = yr - yl, 0.5 * (yl + yr), True
        ie = band & (nx > cfg.normal_cos) & (xa < x1 + 1.0)
        if ie.sum() >= 5:
            x1 = float(np.median(xa[ie]))
        area = width * float(x2 - x1)
        open_rim = bool(x2 >= R - cfg.rim_open_tol)
        reasons = []
        if frms > cfg.floor_rms_max:
            reasons.append(f"floor_rms {frms:.4f} > {cfg.floor_rms_max}")
        if width < cfg.min_width:
            reasons.append(f"width {width:.3f} < {cfg.min_width}")
        if area < cfg.min_area:
            reasons.append(f"area {area:.1f} < {cfg.min_area}")
        if not open_rim:
            reasons.append("not open to the rim")
        ok = not reasons
        # the slot centre line is offset by yc from the ray at thc: fold it into the angle at the floor's mid radius
        rm = float(0.5 * (x1 + x2)); ang = thc + math.atan2(yc, rm)
        slots.append({"centre_deg": round(math.degrees(ang) % 360.0, 3), "width_mm": round(width, 3),
                      "r_inner": round(float(x1), 3), "r_outer_floor": round(float(x2), 3), "depth_mm": round(dep, 3),
                      "z_floor": round(float(z0 + s * dep), 3), "floor_rms": round(frms, 4), "n_points": int(len(q)),
                      "area_mm2": round(area, 1), "open_to_rim": open_rim, "walls_from_wall_faces": wall_fit, "confident": bool(ok),
                      "reject_reason": "; ".join(reasons),
                      "z_end": round(float(z0), 3), "material_side": int(s)})
        mask[q] = True
    return slots, mask


def end_planes(P, N, cfg=RimSlotConfig()):
    """The two axial end faces from the scan: (z0, s) at the low end (material above) and the high end."""
    z = P[:, 2]; ax = np.abs(N[:, 2]) > cfg.normal_cos
    out = []
    lo = ax & (N[:, 2] < 0) & (z < np.percentile(z, 2) + cfg.end_plane_band)
    hi = ax & (N[:, 2] > 0) & (z > np.percentile(z, 98) - cfg.end_plane_band)
    if lo.sum() > 20:
        out.append((float(np.median(z[lo])), +1))
    if hi.sum() > 20:
        out.append((float(np.median(z[hi])), -1))
    return out


def detect(P, N, R, planes=None, cfg=RimSlotConfig()):
    """All rim slots on the given end planes (default: both scan end faces).  Returns a report dict with
    'slots', 'confident' (every slot confident and >= 1 slot; auto-mode cutting is per-slot, see apply)
    and the floor-point mask."""
    planes = planes if planes is not None else end_planes(P, N, cfg)
    slots, mask = [], np.zeros(len(P), bool)
    for z0, s in planes:
        sl, m = detect_end(P, N, R, z0, s, cfg)
        slots += sl; mask |= m
    conf = bool(slots) and all(x["confident"] for x in slots)
    return {"slots": slots, "confident": conf, "planes": [{"z0": round(z, 3), "material_side": s} for z, s in planes],
            "config": asdict(cfg)}, mask


def cut(shape, axis_point, axis_dir, e1, slots, R, over=1.0):
    """Cut each slot (flat floor, parallel walls, flat inner end, open to the rim) from the solid.
    axis_point/axis_dir/e1: the detection frame (z = axis_dir, x = e1)."""
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCP.gp import gp_Pnt, gp_Trsf, gp_Ax1, gp_Ax3, gp_Dir
    a0 = np.asarray(axis_point, float); u = np.asarray(axis_dir, float); u = u / np.linalg.norm(u)
    e1 = np.asarray(e1, float); e1 = e1 - (e1 @ u) * u; e1 /= np.linalg.norm(e1)
    place = gp_Trsf(); place.SetTransformation(gp_Ax3(gp_Pnt(*map(float, a0)), gp_Dir(*map(float, u)), gp_Dir(*map(float, e1))))
    place.Invert()                                        # local (detection frame) -> world
    out = shape
    for sl in slots:
        z_a, z_b = sl["z_floor"], sl["z_end"] - sl["material_side"] * over
        z1, z2 = min(z_a, z_b), max(z_a, z_b)
        w = sl["width_mm"]
        box = BRepPrimAPI_MakeBox(gp_Pnt(sl["r_inner"], -w / 2, z1), gp_Pnt(R + 2.0 + over, w / 2, z2)).Shape()
        rot = gp_Trsf(); rot.SetRotation(gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), math.radians(sl["centre_deg"]))
        box = BRepBuilderAPI_Transform(box, rot, True).Shape()
        box = BRepBuilderAPI_Transform(box, place, True).Shape()
        out = BRepAlgoAPI_Cut(out, box).Shape()
    return out


def to_frame(V, Nv, axis_point, axis_dir, e1):
    a0 = np.asarray(axis_point, float); u = np.asarray(axis_dir, float); u = u / np.linalg.norm(u)
    e1 = np.asarray(e1, float); e1 = e1 - (e1 @ u) * u; e1 /= np.linalg.norm(e1); e2 = np.cross(u, e1)
    D = np.asarray(V, float) - a0
    return np.c_[D @ e1, D @ e2, D @ u], np.c_[Nv @ e1, Nv @ e2, Nv @ u]


def apply(mesh, shape, axis_point, axis_dir, e1, R, flag="auto", planes=None, cfg=RimSlotConfig(), log=print):
    """Detect and cut rim slots.  flag 'on' cuts every detected slot; 'auto' cuts each confident
    slot even when other patches fail the flat-floor (or width/area/open-to-rim) test.  Returns (shape, report)."""
    from OCP.BRepCheck import BRepCheck_Analyzer
    if flag == "off":
        return shape, {"skipped": "rim_slots=off"}
    P, Nl = to_frame(mesh.vertices, np.asarray(mesh.vertex_normals), axis_point, axis_dir, e1)
    rep, _ = detect(P, Nl, R, planes, cfg)
    rep["flag"] = flag
    sl = [x for x in rep["slots"] if x["confident"] or flag == "on"]
    do = bool(sl)
    rep["modelled"] = False
    if do:
        try:
            new = cut(shape, axis_point, axis_dir, e1, sl, R)
            from . import occ_util as U
            from OCP.TopAbs import TopAbs_SOLID
            if BRepCheck_Analyzer(new).IsValid() and U.count(new, TopAbs_SOLID) == 1 and U.volume(new) < U.volume(shape):
                rep["modelled"] = True; rep["n_cut"] = len(sl)
                log(f"[rim_slots] cut {len(sl)} slot(s)")
                return new, rep
            rep["cut_rejected"] = "result not one valid solid or no material removed"
        except Exception as ex:  # noqa: BLE001
            rep["cut_error"] = f"{type(ex).__name__}: {ex}"
    elif rep["slots"]:
        rep["not_modelled_reason"] = (
            "no confident slot (see per-slot 'confident' / 'reject_reason'); "
            "rerun with --rim-slots on to force"
        )
    return shape, rep
