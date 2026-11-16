"""Regions of the scan that metrology does not inspect, and why.

Three sources, all reported in report.json["not_inspected"] and in the PDF:
  * hand-entered features: a feature built from a typed value (e.g. the impeller bore from
    --measure bore_d=14.275 when the scan has no bore). The scan cannot confirm it, and the scan's
    closed ends over the bore opening would otherwise read as large "material the solid lacks"
    deviations (the s07 worst points). Excluded by default; --no-auto-exclude keeps them.
  * user face selection: scan vertices whose nearest solid point lies on the given faces
    (the Fusion add-in exports the selected BRep faces' tessellation as an STL in mm).
  * typed zones: box(xmin,ymin,zmin, xmax,ymax,zmax) and cyl(px,py,pz, dx,dy,dz, r[, t0,t1]),
    several separated by ';'. For cyl, t0..t1 is the axial range measured from (px,py,pz) along the
    unit direction; omitted = unbounded.
  * impeller end-face features (balance cuts, drill marks, uncertain depressions): sector zones
    written into the conversion report by face_features, auto-excluded with a 2 mm boundary.
    A sector is axis + e1 + r0..r1 + theta0..theta1 deg (CCW about the axis) + optional axial t0..t1.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

_NUM = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def parse_zones(text):
    """'box(...); cyl(...)' -> list of zone dicts. Raises ValueError with a readable message."""
    out = []
    for part in (text or "").split(";"):
        part = part.strip()
        if not part:
            continue
        m = re.match(r"\s*([A-Za-z]+)", part)
        if not m:
            raise ValueError(f"zone '{part}' must start with box or cyl")
        kind = m.group(1).lower()
        nums = [float(x) for x in _NUM.findall(part[m.end():])]
        if kind == "box":
            if len(nums) != 6:
                raise ValueError(f"box needs 6 numbers (xmin,ymin,zmin, xmax,ymax,zmax), got {len(nums)} in '{part}'")
            lo, hi = np.minimum(nums[:3], nums[3:]), np.maximum(nums[:3], nums[3:])
            out.append({"type": "box", "min": [float(x) for x in lo], "max": [float(x) for x in hi]})
        elif kind in ("cyl", "cylinder"):
            if len(nums) not in (7, 9):
                raise ValueError(f"cyl needs 7 or 9 numbers (px,py,pz, dx,dy,dz, r[, t0,t1]), got {len(nums)} in '{part}'")
            d = np.asarray(nums[3:6], float)
            if np.linalg.norm(d) < 1e-12:
                raise ValueError(f"cyl direction is zero in '{part}'")
            if nums[6] <= 0:
                raise ValueError(f"cyl radius must be > 0 in '{part}'")
            z = {"type": "cyl", "point": nums[0:3], "dir": (d / np.linalg.norm(d)).tolist(), "r": nums[6],
                 "t": [min(nums[7], nums[8]), max(nums[7], nums[8])] if len(nums) == 9 else None}
            out.append(z)
        elif kind == "sector":
            # px,py,pz, dx,dy,dz, e1x,e1y,e1z, r0,r1, a0,a1[, t0,t1]  (13 or 15 numbers)
            if len(nums) not in (13, 15):
                raise ValueError(
                    f"sector needs 13 or 15 numbers "
                    f"(px,py,pz, dx,dy,dz, e1x,e1y,e1z, r0,r1, a0,a1[, t0,t1]), "
                    f"got {len(nums)} in '{part}'")
            d = np.asarray(nums[3:6], float)
            if np.linalg.norm(d) < 1e-12:
                raise ValueError(f"sector direction is zero in '{part}'")
            e1 = np.asarray(nums[6:9], float)
            z = {"type": "sector", "point": nums[0:3], "dir": (d / np.linalg.norm(d)).tolist(),
                 "e1": e1.tolist(), "r": [min(nums[9], nums[10]), max(nums[9], nums[10])],
                 "theta_deg": [nums[11], nums[12]],
                 "t": [min(nums[13], nums[14]), max(nums[13], nums[14])] if len(nums) == 15 else None,
                 "margin_mm": 2.0}
            out.append(z)
        else:
            raise ValueError(f"unknown zone type '{kind}' (use box, cyl or sector)")
    return out


def zone_text(z):
    if z["type"] == "box":
        return "box min ({:.3f}, {:.3f}, {:.3f}) max ({:.3f}, {:.3f}, {:.3f}) mm".format(*z["min"], *z["max"])
    if z["type"] == "sector":
        t = "" if not z.get("t") else ", axial {:.3f}..{:.3f} mm".format(*z["t"])
        m = z.get("margin_mm")
        mg = f", {m:g} mm boundary" if m else ""
        return ("sector r {:.3f}..{:.3f} mm, {:.3f}..{:.3f} deg about the axis, "
                "axis point ({:.3f}, {:.3f}, {:.3f}), dir ({:.3f}, {:.3f}, {:.3f}){}{}"
                .format(z["r"][0], z["r"][1], z["theta_deg"][0], z["theta_deg"][1],
                        *z["point"], *z["dir"], t, mg))
    t = "" if not z.get("t") else ", axial {:.3f}..{:.3f} mm".format(*z["t"])
    return ("cylinder r {:.3f} mm, axis point ({:.3f}, {:.3f}, {:.3f}), dir ({:.3f}, {:.3f}, {:.3f}){}"
            .format(z["r"], *z["point"], *z["dir"], t))


def _in_arc(th, t0, t1):
    """True where angle th (rad) lies on the CCW arc t0 -> t1 (wraps).  A full circle is all True."""
    import math
    d = (t1 - t0) % (2 * math.pi)
    if d < 1e-12 or d > 2 * math.pi - 1e-12:
        return np.ones(np.shape(th), bool)
    return ((np.asarray(th) - t0) % (2 * math.pi)) <= d + 1e-12


def zone_mask(P, z):
    P = np.asarray(P, float)
    if z["type"] == "box":
        return np.all((P >= np.asarray(z["min"])) & (P <= np.asarray(z["max"])), axis=1)
    if z["type"] == "sector":
        import math
        p = np.asarray(z["point"], float)
        d = np.asarray(z["dir"], float); nrm = np.linalg.norm(d)
        d = d / nrm if nrm else d
        e1 = np.asarray(z["e1"], float); e1 = e1 - (e1 @ d) * d
        e1 = e1 / (np.linalg.norm(e1) if np.linalg.norm(e1) else 1.0)
        e2 = np.cross(d, e1)
        rel = P - p
        t = rel @ d
        x, y = rel @ e1, rel @ e2
        rad = np.hypot(x, y)
        th = np.arctan2(y, x)
        r0, r1 = map(float, z["r"])
        a0, a1 = np.radians(float(z["theta_deg"][0])), np.radians(float(z["theta_deg"][1]))
        margin = float(z.get("margin_mm") or 0.0)
        r0 = r0 - margin; r1 = r1 + margin
        rmid = max(0.5 * (max(r0, 0.0) + max(r1, 0.0)), 1e-3)
        da = margin / rmid
        m = (rad >= r0) & (rad <= r1) & _in_arc(th, a0 - da, a1 + da)
        if z.get("t"):
            m &= (t >= float(z["t"][0]) - margin) & (t <= float(z["t"][1]) + margin)
        return m
    p, d = np.asarray(z["point"], float), np.asarray(z["dir"], float)
    t = (P - p) @ d
    rad = np.linalg.norm((P - p) - t[:, None] * d, axis=1)
    m = rad <= float(z["r"])
    if z.get("t"):
        m &= (t >= z["t"][0]) & (t <= z["t"][1])
    return m


def faces_mask(closest_points, faces_mesh, tol=0.03):
    """Vertices whose nearest solid point lies on the given faces (within tol mm of their tessellation)."""
    from .deviation import _on_surface
    if faces_mesh is None or len(faces_mesh.faces) == 0 or len(closest_points) == 0:
        return np.zeros(len(closest_points), bool)
    _cp, dist, _t = _on_surface(faces_mesh, np.asarray(closest_points, float))
    return np.asarray(dist) <= float(tol)


# ---- hand-entered features -------------------------------------------------------------------------
def find_convert_report(solid_path):
    """The MeshSolid conversion report that built this solid: <stem>.report.json or report.json beside it."""
    p = Path(solid_path)
    for cand in (p.with_name(p.stem + ".report.json"), p.parent / "report.json"):
        if cand.is_file():
            try:
                rep = json.loads(cand.read_text())
            except (OSError, ValueError):
                continue
            if rep.get("tool") == "meshsolid" or "impeller" in rep or "step" in rep or "stages" in rep:
                return str(cand)
    return None


def _sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def find_convert_report_for_scan(scan_path, jobs_root=None, max_reports=400):
    """A MeshSolid conversion report whose input mesh is byte-identical to this scan, searched in the jobs
    folder (default: two levels above the scan, i.e. <jobs>/<job>/scan.stl). Used when the solid carries no
    link to its conversion (a Fusion body that lost the attribute, e.g. Stage 3 'Body4'), so hand-entered
    features (a typed bore the scan lacks) are still excluded. Returns (path, job) or (None, None)."""
    sp = Path(scan_path)
    root = Path(jobs_root) if jobs_root else sp.parent.parent
    try:
        size = sp.stat().st_size
        cands = sorted(root.glob("*/*.report.json"), key=lambda q: q.stat().st_mtime, reverse=True)[:max_reports]
    except OSError:
        return None, None
    sha = None
    for cand in cands:
        if cand.parent == sp.parent and cand.name == "report.json":
            continue
        try:
            rep = json.loads(cand.read_text())
        except (OSError, ValueError):
            continue
        if rep.get("tool") != "meshsolid" or not rep.get("input"):
            continue
        inp = Path(rep["input"])
        try:
            if not inp.is_file() or inp.stat().st_size != size:
                continue
            sha = sha or _sha256(sp)
            if _sha256(inp) == sha:
                return str(cand), cand.parent.name
        except OSError:
            continue
    return None, None


def hand_entered_features(convert_report):
    """[{kind, d_mm, source, label}] for features built from typed values rather than the scan."""
    if not convert_report:
        return []
    try:
        rep = json.loads(Path(convert_report).read_text()) if not isinstance(convert_report, dict) else convert_report
    except (OSError, ValueError):
        return []
    out = []
    bore = ((rep.get("impeller") or {}).get("bore") or {})
    if bore.get("d") and str(bore.get("source", "")).startswith("measure"):
        out.append({"kind": "bore", "d_mm": float(bore["d"]), "source": str(bore["source"]),
                    "label": f"bore Ø{float(bore['d']):g} mm (hand-entered: {bore['source']}; the scan has no bore)"})
    return out


def _step_cylinders(step_path, radius, tol=0.02):
    """Axes of cylindrical faces of the given radius in a STEP: [(point, unit_dir)]."""
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopoDS import TopoDS
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.GeomAbs import GeomAbs_Cylinder
    from .occ_util import read_step
    face = getattr(TopoDS, "Face_s", None) or TopoDS.Face
    out = []
    ex = TopExp_Explorer(read_step(str(step_path)), TopAbs_FACE)
    while ex.More():
        s = BRepAdaptor_Surface(face(ex.Current()))
        if s.GetType() == GeomAbs_Cylinder:
            c = s.Cylinder()
            if abs(c.Radius() - radius) <= tol:
                a = c.Axis(); lo, dr = a.Location(), a.Direction()
                out.append(([lo.X(), lo.Y(), lo.Z()], [dr.X(), dr.Y(), dr.Z()]))
        ex.Next()
    return out


def hand_entered_zones(features, solid_path, solid_tm, margin=0.1):
    """Exclusion zones for hand-entered features. A hand-entered bore excludes the scan inside a coaxial
    cylinder of the bore radius + margin over the whole part length (the scan's closed ends cover it)."""
    zones = []
    for f in features:
        if f["kind"] != "bore":
            continue
        R = 0.5 * f["d_mm"]
        axes = []
        if str(solid_path).lower().endswith((".step", ".stp")):
            try:
                axes = _step_cylinders(solid_path, R)
            except Exception:  # noqa: BLE001  (OCC missing or unreadable: fall back to the mesh axis)
                axes = []
        if not axes:
            from .metrology import rotation_axis
            c, u = rotation_axis(solid_tm)
            axes = [(c.tolist(), u.tolist())]
        p, d = axes[0]
        d = (np.asarray(d, float) / np.linalg.norm(d)).tolist()
        zones.append({"type": "cyl", "point": [float(x) for x in p], "dir": d, "r": R + margin, "t": None,
                      "reason": "hand-entered feature", "label": f["label"], "kind": f["kind"], "d_mm": f["d_mm"]})
    return zones


def face_label(path, k=0):
    """'exclude_face_02_cone.stl' -> 'face 2 (cone)' (the add-in writes one STL per selected face)."""
    import re as _re
    stem = Path(str(path)).stem
    m = _re.match(r"exclude_face_(\d+)(?:_(.+))?$", stem)
    if m:
        kind = (m.group(2) or "").replace("_", " ").strip()
        return f"face {int(m.group(1))}" + (f" ({kind})" if kind else "")
    return f"selected faces {k + 1}" if stem in ("exclude_faces", "faces") else stem


def face_feature_zones(convert_report):
    """Named exclusion zones from impeller end-face classification (balance cuts, drill marks,
    uncertain features).  Each zone carries a 2 mm boundary, a label, class and reason so Metrology
    can grey the region and list it on the Not inspected page.

    Modelled features (periodic couplings, bores, registers) are not returned.
    """
    if not convert_report:
        return []
    try:
        rep = json.loads(Path(convert_report).read_text()) if not isinstance(convert_report, dict) else convert_report
    except (OSError, ValueError):
        return []
    ff = rep.get("face_features") or {}
    out = []
    for f in ff.get("exclusions") or ff.get("features") or []:
        if not (f.get("exclude") or f.get("class") in ("a", "uncertain")):
            continue
        z = dict(f.get("boundary") or {})
        if z.get("type") not in ("sector", "cyl", "box"):
            continue
        z["reason"] = "face feature"
        z["label"] = f.get("label") or f.get("kind") or "end-face feature"
        detail_bits = [f.get("reason") or "irregular or uncertain end-face feature, not modelled"]
        if f.get("class"):
            detail_bits.append(f"class {f['class']}" + (f" ({f.get('kind')})" if f.get("kind") else ""))
        if z.get("type") == "sector":
            detail_bits.append(zone_text(z))
        z["detail"] = "; ".join(detail_bits)
        z["feature_class"] = f.get("class")
        z["kind"] = f.get("kind")
        z.setdefault("margin_mm", 2.0)
        out.append(z)
    return out

