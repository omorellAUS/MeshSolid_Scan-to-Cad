"""Scan-to-solid metrology: signed deviation map, zones, probes, section, histogram.

    python -m meshsolid.metrology SCAN.stl SOLID.step --out DIR
        [--range MIN MAX] [--symmetric] [--pass TOL] [--bands N]
        [--band-json FILE] [--exclude TAG,TAG] [--max-dist D]
        [--align] [--device auto|cpu|cuda] [--lin MM]
        [--probes N] [--probe X,Y,Z] [--section ORIGIN:NORMAL]
        [--no-map] [--no-zones] [--no-histogram] [--no-section-plot]

Look matches the impeller / pinion inspection maps: shaded 3-D deviation views, a banded
blue-cyan-green-yellow-red legend (green = in tolerance), histogram and stats on one report PNG.
Distances are signed (positive = scan outside the solid). The scan is never modified on disk.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import trimesh

STEP_EXT = (".step", ".stp")
MESH_EXT = (".stl", ".obj", ".ply")

# ---- band colours: every default band clearly distinct; red is reserved for "beyond outlier cap" ----
# Side palettes run from next-to-tolerance outwards (plus: yellow, orange, magenta, purple, brown;
# minus: cyan, teal, light blue, blue, navy). Grey = not inspected (face/zone exclusions).
GREEN = (36, 170, 65)
PLUS_COLORS = ((245, 215, 35), (245, 140, 25), (215, 40, 190), (110, 30, 180), (110, 60, 25))
MINUS_COLORS = ((30, 210, 225), (0, 120, 120), (60, 140, 255), (25, 60, 210), (15, 20, 105))
_SIDE_SETS = {1: (2,), 2: (0, 2), 3: (0, 1, 2), 4: (0, 1, 2, 3), 5: (0, 1, 2, 3, 4)}
_MINUS_SETS = {1: (3,), 2: (0, 4), 3: (0, 3, 4), 4: (0, 2, 3, 4), 5: (0, 1, 2, 3, 4)}
PLUS_EXTRA = ((250, 150, 200), (128, 128, 0), (60, 15, 80), (210, 180, 120), (190, 160, 230))
MINUS_EXTRA = ((170, 235, 240), (75, 40, 160), (90, 110, 130), (25, 25, 30), (70, 130, 180))
CAP_RGB = (255, 0, 0)              # beyond the outlier cap (default; selectable). Pure red is >= 90 from every
                                   # palette colour; the old (220, 30, 30) sat 63 from brown, which forced that band near-black
GRAY = (150, 150, 154)             # not inspected (face / zone / hand-entered exclusions)
MIN_COLOR_DIST = 45.0
RESERVED_DIST = 70.0               # band colours keep well away from beyond-cap red and not-inspected grey


def side_colors(n, plus=True):
    """n distinct colours for one side of the pass band, next-to-tolerance first."""
    pal = PLUS_COLORS if plus else MINUS_COLORS
    sets = _SIDE_SETS if plus else _MINUS_SETS
    if n <= 0:
        return []
    if n in sets:
        return [pal[i] for i in sets[n]]
    ext = list(pal) + list(PLUS_EXTRA if plus else MINUS_EXTRA)     # more than 5 on one side
    out = [ext[k % len(ext)] for k in range(n)]
    return make_distinct(out)[0]


def color_dist(a, b):
    """Rough perceptual RGB distance ('redmean')."""
    r = 0.5 * (a[0] + b[0])
    dr, dg, db = a[0] - b[0], a[1] - b[1], a[2] - b[2]
    return ((2 + r / 256.0) * dr * dr + 4 * dg * dg + (2 + (255 - r) / 256.0) * db * db) ** 0.5 / 3.0


def make_distinct(rgbs, reserved=(CAP_RGB, GRAY), min_dist=MIN_COLOR_DIST, reserved_dist=RESERVED_DIST):
    """Adjust colours that are too close to their neighbour or to a reserved colour (beyond-cap red,
    not-inspected grey) by lightening or darkening them. Returns (new_rgbs, changed_indices)."""
    out = [tuple(int(c) for c in x[:3]) for x in rgbs]
    changed = []
    for i in range(len(out)):
        def worst(c):
            # margin to the requirement: neighbours need min_dist, reserved colours reserved_dist
            m = [color_dist(c, out[j]) - min_dist for j in (i - 1, i + 1) if 0 <= j < len(out)]
            m += [color_dist(c, tuple(r)) - reserved_dist for r in reserved]
            return min(m or [1e9])
        if worst(out[i]) >= 0:
            continue
        best, best_d = out[i], worst(out[i])
        for t in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
            for target in ((255, 255, 255), (0, 0, 0)):
                c = tuple(int(round(out[i][k] + (target[k] - out[i][k]) * t)) for k in range(3))
                d = worst(c)
                if d > best_d + 1e-9:
                    best, best_d = c, d
            if best_d >= 0:
                break
        if best != out[i]:
            out[i] = best
            changed.append(i)
    return out, changed


def _js(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def _stats(d):
    d = np.asarray(d, float)
    if d.size == 0:
        return None
    a = np.abs(d)
    return {"n": int(d.size), "rms": round(float(np.sqrt(np.mean(d * d))), 5),
            "mean": round(float(np.mean(d)), 5), "p95": round(float(np.percentile(a, 95)), 5),
            "max_plus": round(float(d.max()), 5), "max_minus": round(float(d.min()), 5),
            "max_abs": round(float(a.max()), 5)}


def default_bands(vmin, vmax, pass_tol, n_bands):
    """Split [vmin, vmax] into n_bands. The pass interval is green; every side band gets its own clearly
    different colour (minus: cyan, light blue, blue, indigo, navy; plus: yellow, orange, magenta, purple).
    Red is not used: it marks points beyond the outlier cap."""
    vmin, vmax = float(vmin), float(vmax)
    if vmax <= vmin:
        vmax = vmin + 1e-6
    p = abs(float(pass_tol))
    lo, hi = -p, p
    if lo < vmin:
        lo = vmin
    if hi > vmax:
        hi = vmax
    n_bands = max(3, int(n_bands))
    # Allocate at least one band to the pass interval when it has width, rest to the sides.
    has_pass = hi > lo + 1e-12
    n_pass = 1 if has_pass else 0
    rest = n_bands - n_pass
    w_lo, w_hi = max(0.0, lo - vmin), max(0.0, vmax - hi)
    if rest <= 0:
        n_pass, n_lo, n_hi = n_bands, 0, 0
    elif w_lo + w_hi <= 0:
        n_pass, n_lo, n_hi = n_bands, 0, 0
        lo, hi = vmin, vmax
    else:
        n_lo = int(round(rest * w_lo / (w_lo + w_hi)))
        n_lo = min(max(n_lo, 1 if w_lo > 0 else 0), rest - (1 if w_hi > 0 else 0))
        n_hi = rest - n_lo

    def _edges(a, b, n):
        if n <= 0 or b <= a:
            return []
        return list(np.linspace(a, b, n + 1))

    edges = []
    if n_lo:
        edges += _edges(vmin, lo, n_lo)[:-1]
    if n_pass:
        edges += _edges(lo, hi, n_pass)[:-1]
    else:
        edges.append(lo if has_pass else vmin)
    if n_hi:
        edges += _edges(hi, vmax, n_hi)
    else:
        edges.append(vmax)
    if edges[-1] != vmax:
        edges.append(vmax)
    # unique, monotonic
    ev = [edges[0]]
    for x in edges[1:]:
        if x - ev[-1] > 1e-12:
            ev.append(float(x))
    if ev[-1] < vmax:
        ev.append(vmax)
    bands = []
    n = len(ev) - 1
    i_lo = i_hi = 0
    minus, plus = side_colors(n_lo, plus=False), side_colors(n_hi, plus=True)
    for i in range(n):
        a, b = float(ev[i]), float(ev[i + 1])
        mid = 0.5 * (a + b)
        if mid < lo - 1e-12 and n_lo:
            col = minus[n_lo - 1 - i_lo]              # far (navy) .. near (cyan) going up the scale
            i_lo += 1
        elif mid > hi + 1e-12 and n_hi:
            col = plus[i_hi]                          # near (yellow) .. far (magenta / purple)
            i_hi += 1
        else:
            col = GREEN
        bands.append({"lo": round(a, 6), "hi": round(b, 6),
                      "color": "#%02x%02x%02x" % col, "rgb": list(col),
                      "in_tol": bool(lo - 1e-12 <= mid <= hi + 1e-12)})
    return bands


def load_bands(path_or_list, vmin, vmax, pass_tol, n_bands):
    if not path_or_list:
        return default_bands(vmin, vmax, pass_tol, n_bands)
    data = path_or_list
    if isinstance(path_or_list, (str, Path)):
        data = json.loads(Path(path_or_list).read_text())
    out = []
    for b in data:
        col = b.get("color") or b.get("rgb")
        if isinstance(col, str) and col.startswith("#"):
            rgb = [int(col[i:i + 2], 16) for i in (1, 3, 5)]
        else:
            rgb = [int(x) for x in col]
        out.append({"lo": float(b["lo"]), "hi": float(b["hi"]), "color": "#%02x%02x%02x" % tuple(rgb),
                    "rgb": rgb, "in_tol": bool(b.get("in_tol", abs(0.5 * (b["lo"] + b["hi"])) <= abs(pass_tol)))})
    return out


def colour_clashes(bands, cap_rgb=CAP_RGB, gray=GRAY):
    """User-picked colours are NEVER changed (1.0.1 lightened/darkened them silently, which pushed bands near
    black). Instead every clash is returned as a visible warning: neighbouring bands that look alike, or a band
    that looks like the beyond-cap or not-inspected colour."""
    out = []
    for i, b in enumerate(bands):
        c = tuple(b["rgb"])
        if i and color_dist(c, tuple(bands[i - 1]["rgb"])) < MIN_COLOR_DIST:
            out.append({"band": i, "with": f"band {i}", "text": f"bands {i} and {i + 1} ({bands[i - 1]['color']} / "
                        f"{b['color']}) look alike"})
        for name, r in (("the beyond-cap colour", cap_rgb), ("the not-inspected grey", gray)):
            if r is not None and color_dist(c, tuple(r)) < RESERVED_DIST:
                out.append({"band": i, "with": name, "text": f"band {i + 1} ({b['color']}, {b['lo']:+.3f}..{b['hi']:+.3f} mm) "
                            f"looks like {name} (#%02x%02x%02x)" % tuple(r)})
    return out


def assign_bands(dv, bands):
    idx = np.full(len(dv), -1, int)
    for i, b in enumerate(bands):
        k = (dv >= b["lo"]) & (dv <= b["hi"] if i == len(bands) - 1 else dv < b["hi"])
        idx[k] = i
    # anything outside the scale still gets the end band
    idx[dv < bands[0]["lo"]] = 0
    idx[dv > bands[-1]["hi"]] = len(bands) - 1
    return idx


def vertex_colors(dv, bands, excluded=None, beyond_cap=None, cap_rgb=CAP_RGB):
    """Per-vertex colours and band index: -1 = not inspected (grey), -2 = beyond the outlier cap (cap colour)."""
    idx = assign_bands(dv, bands)
    col = np.array([list(bands[max(i, 0)]["rgb"]) + [255] for i in idx], dtype=np.uint8)
    if beyond_cap is not None and np.any(beyond_cap):
        col[beyond_cap] = list(cap_rgb)[:3] + [255]
        idx[beyond_cap] = -2
    if excluded is not None and np.any(excluded):
        col[excluded] = list(GRAY) + [255]
        idx[excluded] = -1
    return col, idx


def find_job_files(folder):
    """Pick a scan mesh and a STEP from a MeshSolid job folder (report.json if present)."""
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    out = {"folder": str(folder), "scan": None, "solid": None, "report": None}
    reps = list(folder.glob("*.report.json")) + list(folder.glob("report.json"))
    if reps:
        out["report"] = str(reps[0])
        try:
            rep = json.loads(Path(reps[0]).read_text())
            for key in ("step", "output", "solid"):
                p = Path(rep[key]) if rep.get(key) else None
                if p and p.is_file():
                    out["solid"] = str(p)
                    break
            mesh = (rep.get("mesh") or {}).get("path") or rep.get("input") or rep.get("mesh_path")
            if mesh and Path(mesh).is_file():
                out["scan"] = str(mesh)
        except (OSError, ValueError):
            pass
    if out["solid"] is None:
        steps = sorted(p for p in folder.iterdir() if p.suffix.lower() in STEP_EXT)
        if steps:
            out["solid"] = str(steps[0])
    if out["scan"] is None:
        preferred = list(folder.glob("mesh_body.stl")) + list(folder.glob("*.stl"))
        meshes = [p for p in preferred if p.suffix.lower() in MESH_EXT] or \
            [p for p in folder.iterdir() if p.suffix.lower() in MESH_EXT]
        if meshes:
            out["scan"] = str(meshes[0])
    return out


def load_scan(path):
    m = trimesh.load(str(path), force="mesh", process=True)
    if not isinstance(m, trimesh.Trimesh) or len(m.faces) == 0:
        raise ValueError(f"{path}: no triangles")
    return m


def load_solid(path, lin=0.01, ang=0.1):
    path = Path(path)
    if path.suffix.lower() in STEP_EXT:
        from .occ_util import read_step, shape_to_trimesh
        shape = read_step(str(path))
        tm = shape_to_trimesh(shape, lin=lin, ang=ang)
        tm.merge_vertices()
        try:
            from .occ_util import triangle_face_ids
            ids, types = triangle_face_ids(shape, tm, lin=lin, ang=ang)
            tm.metadata["brep_face"] = np.asarray(ids, int)
            tm.metadata["brep_face_types"] = list(types)
        except Exception as e:  # noqa: BLE001  (face ids are extra information; never lose the measurement)
            tm.metadata["brep_face_error"] = f"{type(e).__name__}: {e}"
        return tm, "step"
    return load_scan(path), "mesh"


def signed_distances(scan, solid_tm):
    """Per-vertex signed distance of the scan to the solid tessellation (positive = outside)."""
    from .deviation import _on_surface
    P = np.asarray(scan.vertices, float)
    cp, dist, tri = _on_surface(solid_tm, P)
    n = np.asarray(solid_tm.face_normals, float)[np.asarray(tri, int)]
    sgn = np.sign(np.einsum("ij,ij->i", P - cp, n))
    dv = np.asarray(dist, float) * np.where(sgn == 0, 1.0, sgn)
    return dv, cp, np.asarray(tri, int)


def align_scan(scan, solid_tm, samples=25000, seed=0):
    """Rigid ICP taking a copy of the scan onto the solid. Returns (mesh_copy, 4x4, median residual)."""
    from .compare import _icp
    rng = np.random.default_rng(seed)
    def _sample(m, n):
        # area-uniform surface samples (1.0.1 capped n at the vertex count, so a coarse tessellation such as
        # a box's 8 vertices was aligned from 8 points; the cap now applies only to the vertex fallback)
        if len(m.faces) and m.area > 0:
            return np.asarray(trimesh.sample.sample_surface(m, int(n), seed=seed)[0])
        n = min(int(n), max(len(m.vertices), 1))
        idx = rng.choice(len(m.vertices), size=n, replace=len(m.vertices) < n)
        return np.asarray(m.vertices[idx], float)
    T, med, _mir = _icp(_sample(solid_tm, samples), _sample(scan, samples))
    out = scan.copy()
    out.apply_transform(T)
    return out, T, med


def pose_of(T):
    """Rotation angle (deg), its unit axis and the translation (mm) of a 4x4 rigid transform."""
    if T is None:
        return {"rotation_deg": 0.0, "rotation_axis": [0.0, 0.0, 1.0], "translation_mm": [0.0, 0.0, 0.0]}
    T = np.asarray(T, float)
    R = T[:3, :3]
    c = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    ang = float(np.degrees(np.arccos(c)))
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    nw = float(np.linalg.norm(w))
    if nw > 1e-12:
        ax = w / nw
    else:                                   # 0 or 180 deg: axis from the symmetric part
        vals, vecs = np.linalg.eigh(R + R.T)
        ax = vecs[:, int(np.argmax(vals))]
    return {"rotation_deg": round(ang, 6), "rotation_axis": [round(float(v), 6) for v in ax],
            "translation_mm": [round(float(v), 6) for v in T[:3, 3]]}


def rotation_axis(solid_tm):
    """(centre, unit axis) of a rotational part: the PCA axis whose variance differs most from the other
    two (an impeller's axis is its SMALLEST extent, a shaft's its largest; the two radial ones match)."""
    V = np.asarray(solid_tm.vertices, float)
    c = solid_tm.bounds.mean(0)
    w, ev = np.linalg.eigh(np.cov((V - c).T))
    w = np.maximum(w, 1e-12)
    # how far each eigenvalue is from the mean of the other two (log ratio)
    odd = [abs(np.log(w[i] / (0.5 * (w[(i + 1) % 3] + w[(i + 2) % 3])))) for i in range(3)]
    u = ev[:, int(np.argmax(odd))]
    return c, u / (np.linalg.norm(u) + 1e-15)


def part_type_from_report(crep):
    """Zone rule set from the conversion report's recognised part type: (kind, source) with kind in
    impeller / gear / none, or (None, reason) when the report does not say (then the geometric
    auto-detect decides).  The pinion (gear detected, built as a revolve) got impeller zones from the
    geometric guess before."""
    if not crep:
        return None, "no conversion report"
    imp = crep.get("impeller") or {}
    method = ((crep.get("stages") or {}).get("build") or {}).get("method")
    if method == "impeller" or (imp and (imp.get("blades") or imp.get("n_blades"))):
        return "impeller", f"conversion report: impeller recognised ({imp.get('blades', '?')} blades, build {method})"
    gear = crep.get("gear") or {}
    if gear.get("detected"):
        return "gear", f"conversion report: gear recognised ({gear.get('teeth', '?')} teeth, build {method})"
    if method:
        return "none", f"conversion report: built as {method}, neither impeller nor gear recognised"
    return None, "conversion report has no recognised part type"


def classify_zones(scan, solid_tm, kind="auto"):
    """Per-vertex zone names for an impeller-like or gear-like part, or None if not recognised.

    Impeller: blades, hub, frontface, backface, bore (same geometric rules as scripts/stg1_metrology.py).
    Gear: flank, tip (outer-radius band), plus end_face. Returns (zones, kind, note)."""
    P = np.asarray(scan.vertices, float)
    # axis of rotation (was the largest-extent PCA axis, which is radial on a disc-like impeller)
    c, u = rotation_axis(solid_tm)
    z = (P - c) @ u
    rad_vec = (P - c) - u * z[:, None]
    r = np.linalg.norm(rad_vec, axis=1)
    er = rad_vec / np.maximum(r, 1e-9)[:, None]
    et = np.cross(u, er)
    n = np.asarray(scan.vertex_normals, float)
    nt = np.abs(np.einsum("ij,ij->i", n, et))
    nu = np.abs(n @ u)
    r_p95, r_p50 = float(np.percentile(r, 95)), float(np.percentile(r, 50))
    # rotational? similar radii in a mid-length slice
    mid = (z > np.percentile(z, 30)) & (z < np.percentile(z, 70))
    rotational = mid.any() and (float(np.std(r[mid])) / max(r_p50, 1e-6) < 0.55)
    if kind == "auto":
        kind = "none"
        if rotational and r_p95 > 1.5 * max(r_p50, 1e-6) and float(np.mean(nt > 0.5)) > 0.08:
            kind = "impeller" if float(np.mean(nt > 0.5)) > 0.15 else "gear"
        elif rotational and float(np.mean(nu > 0.85)) > 0.1:
            kind = "impeller"
    if kind == "impeller":
        inward = -np.einsum("ij,ij->i", n, er)
        bore_pts = r[(inward > 0.7) & (r < r_p50)]
        r_bore = float(np.percentile(bore_pts, 10)) if len(bore_pts) > 50 else float(np.percentile(r, 5))
        # the solid's bore (inward-facing faces near the axis) wins: a scan without a bore has none
        Fc = np.asarray(solid_tm.triangles_center, float) - c
        Fz = Fc @ u
        Fr_vec = Fc - u * Fz[:, None]
        Fr = np.linalg.norm(Fr_vec, axis=1)
        Fin = -np.einsum("ij,ij->i", np.asarray(solid_tm.face_normals, float), Fr_vec / np.maximum(Fr, 1e-9)[:, None])
        sb = (Fin > 0.95) & (Fr < r_p50)
        if sb.sum() > 20:
            r_bore = float(np.median(Fr[sb]))
        zlo, zhi = float(np.percentile(z, 2)), float(np.percentile(z, 98))
        zone = np.full(len(P), "hub", dtype=object)
        zone[nt > 0.5] = "blades"
        zone[(nu > 0.85) & (z < zlo + 0.15 * (zhi - zlo))] = "backface"
        zone[(nu > 0.85) & (z > zhi - 0.15 * (zhi - zlo))] = "frontface"
        zone[r < r_bore + max(0.5, 0.02 * r_p95)] = "bore"
        return zone, "impeller", "geometric impeller zones (blades, hub, frontface, backface, bore)"
    if kind == "gear":
        zone = np.full(len(P), "flank", dtype=object)
        zone[r >= 0.92 * r_p95] = "tip"
        zone[nu > 0.8] = "end_face"
        return zone, "gear", "geometric gear zones (flank, tip, end_face)"
    return None, "none", "part was not recognised as an impeller or gear; per-zone stats were skipped"


def worst_probes(scan, dv, n=10, excluded=None, extra_points=None, cap=None):
    """Auto-label the N worst |deviation| vertices (true values: points beyond the outlier cap are
    included and marked), plus any typed-in coordinates (closest vertex)."""
    keep = np.ones(len(dv), bool) if excluded is None else ~excluded
    order = np.argsort(-np.abs(np.where(keep, dv, 0.0)))
    out = []
    for i in order[: max(0, int(n))]:
        if not keep[i]:
            continue
        p = scan.vertices[i]
        row = {"kind": "worst", "vertex": int(i), "xyz_mm": [round(float(x), 4) for x in p],
               "deviation_mm": round(float(dv[i]), 5)}
        if cap is not None and abs(float(dv[i])) > float(cap):
            row["kind"] = "worst, beyond cap"
            row["beyond_cap"] = True
        out.append(row)
    if extra_points:
        tree_ok = True
        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(np.asarray(scan.vertices, float))
        except Exception:
            tree_ok = False
            tree = None
        for pt in extra_points:
            q = np.asarray(pt, float).reshape(3)
            if tree_ok:
                _, j = tree.query(q)
            else:
                j = int(np.argmin(np.linalg.norm(scan.vertices - q, axis=1)))
            p = scan.vertices[j]
            out.append({"kind": "typed", "vertex": int(j),
                        "query_mm": [round(float(x), 4) for x in q],
                        "xyz_mm": [round(float(x), 4) for x in p],
                        "deviation_mm": round(float(dv[j]), 5)})
    return out


def section_points(scan, dv, origin, normal, thick=0.25):
    """Cross-section of the scan: edge-plane intersections (plus any vertices in a thin slab).
    Deviation at an intersection is linearly interpolated from the edge's vertices."""
    n = np.asarray(normal, float)
    n = n / (np.linalg.norm(n) + 1e-15)
    o = np.asarray(origin, float)
    V = np.asarray(scan.vertices, float)
    h = (V - o) @ n
    pts, devs = [], []
    k = np.abs(h) <= float(thick)
    if k.any():
        pts.append(V[k]); devs.append(dv[k])
    # every unique edge that crosses the plane
    edges = np.asarray(scan.edges_unique, int)
    h0, h1 = h[edges[:, 0]], h[edges[:, 1]]
    cross = h0 * h1 <= 0
    cross &= np.abs(h0) + np.abs(h1) > 1e-15
    if cross.any():
        e = edges[cross]
        a, b = h[e[:, 0]], h[e[:, 1]]
        t = a / np.where(np.abs(b - a) < 1e-15, 1.0, (a - b))
        t = np.clip(t, 0.0, 1.0)
        P = V[e[:, 0]] + (V[e[:, 1]] - V[e[:, 0]]) * t[:, None]
        d = dv[e[:, 0]] + (dv[e[:, 1]] - dv[e[:, 0]]) * t
        pts.append(P); devs.append(d)
    if pts:
        P = np.vstack(pts); d = np.concatenate(devs)
    else:
        P = np.zeros((0, 3)); d = np.zeros(0)
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(n, a); e1 /= np.linalg.norm(e1) + 1e-15
    e2 = np.cross(n, e1)
    return {"n": int(len(P)), "u_mm": (P - o) @ e1 if len(P) else P, "v_mm": (P - o) @ e2 if len(P) else P,
            "deviation_mm": d, "xyz_mm": P, "origin": o.tolist(), "normal": n.tolist(),
            "thick_mm": float(thick)}


# ---- renders (impeller / pinion inspection look) -----------------------------------------------
def _world_axis_name(u):
    k = int(np.argmax(np.abs(u)))
    if abs(u[k]) > 0.95:
        return ("+" if u[k] > 0 else "-") + "XYZ"[k]
    return "({:+.2f}, {:+.2f}, {:+.2f})".format(*u)


def view_frames(mesh, axis=None):
    """Camera frames for the three map views, tied to the part's rotation axis (not to world Z: Fusion
    designs are often Y-up). Returns [(name, R, label)], R rows = screen right, screen up, toward viewer.
      end   - looking along the rotation axis (from the blade / nose side)
      side  - looking across the axis, axis vertical on the page
      iso   - 35 deg above the side view, turned 55 deg about the axis."""
    if axis is None:
        _c, u = rotation_axis(mesh)
    else:
        u = np.asarray(axis, float)
    u = u / (np.linalg.norm(u) + 1e-15)
    # face the long tail of the part (nose / hub side of an impeller) toward the viewer
    try:
        tc = np.asarray(mesh.triangles_center, float); wa = np.asarray(mesh.area_faces, float)
        t = (tc - np.average(tc, axis=0, weights=wa)) @ u
        if float(np.average(t ** 3, weights=wa)) < 0:
            u = -u
    except Exception:  # noqa: BLE001
        pass
    ref = np.eye(3)[int(np.argmin(np.abs(u)))]
    e1 = ref - u * (ref @ u); e1 /= np.linalg.norm(e1)
    e2 = np.cross(u, e1)

    def frame(w, up):
        w = w / np.linalg.norm(w)
        y = up - w * (up @ w); y /= np.linalg.norm(y)
        x = np.cross(y, w)
        return np.vstack([x, y, w])
    a = np.radians(-55.0)
    side_dir = np.cos(a) * e1 + np.sin(a) * e2
    iso_w = np.cos(np.radians(35.0)) * side_dir + np.sin(np.radians(35.0)) * u
    ax = _world_axis_name(u)
    return [("iso", frame(iso_w, u), "isometric"),
            ("end", np.vstack([e1, np.cross(u, e1), u]), f"end view (along the rotation axis, {ax})"),
            ("side", frame(side_dir, u), f"side view (across the axis, {ax} up)")]


def render_shaded(mesh, rgb, size=720, elev=25.0, azim=-55.0, bg=(18, 20, 24, 255), R=None,
                  outlines=None, outline_rgb=(255, 210, 40)):
    """Painter's-algorithm view: per-face colour x Lambert shade. rgb is (n_vertices, 3|4) 0-255.
    R (3x3, rows = screen right, up, toward viewer) overrides elev/azim.
    outlines: optional list of (N,3) world-frame polylines drawn on top (region boundaries)."""
    from PIL import Image, ImageDraw
    V = mesh.vertices - mesh.bounds.mean(0)
    if R is not None:
        P = V @ np.asarray(R, float).T
    else:
        a, e = np.radians(azim), np.radians(elev)
        Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        Rx = np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0, np.sin(e), np.cos(e)]])
        P = V @ Rz.T @ Rx.T
    F = mesh.faces
    face_rgb = np.asarray(rgb, float)[F].mean(1)[:, :3] / 255.0
    nrm = np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]])
    nrm /= np.linalg.norm(nrm, axis=1)[:, None] + 1e-12
    L = np.array([-0.35, 0.45, 0.82]); L /= np.linalg.norm(L)
    col = (np.clip(face_rgb * (0.35 + 0.65 * np.abs(nrm @ L))[:, None], 0, 1) * 255).astype(np.uint8)
    ext = np.abs(P[:, :2]).max() * 1.04 + 1e-9
    xy = (P[:, :2] / ext * 0.5 + 0.5) * (size - 1)
    xy[:, 1] = size - 1 - xy[:, 1]
    img = Image.new("RGBA", (size, size), bg)
    d = ImageDraw.Draw(img)
    order = np.argsort(P[F].mean(1)[:, 2])
    # draw in chunks of same-ish colour is slower to set up; just iterate (ok for typical scan sizes)
    for i in order:
        f = F[i]
        d.polygon([tuple(xy[f[0]]), tuple(xy[f[1]]), tuple(xy[f[2]])], fill=tuple(int(c) for c in col[i]) + (255,))
    if outlines:
        mean = mesh.bounds.mean(0)
        colour = tuple(int(c) for c in outline_rgb[:3]) + (255,)
        for pts in outlines:
            pts = np.asarray(pts, float)
            if len(pts) < 2:
                continue
            Q = (pts - mean) @ (np.asarray(R, float).T if R is not None else np.eye(3))
            # if no R was passed, match the elev/azim transform used above
            if R is None:
                a, e = np.radians(azim), np.radians(elev)
                Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
                Rx = np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0, np.sin(e), np.cos(e)]])
                Q = (pts - mean) @ Rz.T @ Rx.T
            px = (Q[:, :2] / ext * 0.5 + 0.5) * (size - 1)
            px[:, 1] = size - 1 - px[:, 1]
            ring = [tuple(p) for p in px] + [tuple(px[0])]
            d.line(ring, fill=colour, width=3)
    return img


def render_legend(bands, pass_tol, vmin, vmax, path, width=380, height=760, title="Deviation (mm)"):
    """Vertical banded legend, same palette as the map, with pass-band marks. Readable sizes:
    labels 12 pt, saved at 200 dpi."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    ax.set_xlim(0, 1); ax.set_ylim(vmin, vmax)
    ax.set_xticks([]); ax.set_yticks([])
    for b in bands:
        rgb = [c / 255.0 for c in b["rgb"]]
        ax.add_patch(Rectangle((0.04, b["lo"]), 0.30, max(b["hi"] - b["lo"], 1e-9),
                               facecolor=rgb, edgecolor="0.15", linewidth=0.6, transform=ax.transData))
        ax.text(0.38, 0.5 * (b["lo"] + b["hi"]), f"{b['lo']:+.3f} … {b['hi']:+.3f}",
                fontsize=12, va="center", color="0.1")
    for y, ls in ((0.0, "-"), (pass_tol, "--"), (-pass_tol, "--")):
        if vmin <= y <= vmax:
            ax.axhline(y, color="0.1", ls=ls, lw=1.0, xmin=0.02, xmax=0.36)
    ax.set_title(f"{title}\ngreen = within ±{pass_tol:g}", fontsize=13, pad=8)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    fig.tight_layout(pad=0.4)
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    return str(path)


def draw_legend(ax, bands, pass_tol, fontsize=12, title="Deviation (mm)", excluded=True, n_excluded=None,
                cap=None, n_cap=None, cap_rgb=CAP_RGB):
    """Colour-band legend drawn with matplotlib text (stays readable at any figure size). Extra rows:
    not inspected (grey) and beyond the outlier cap (cap colour), each with its count."""
    ax.set_xlim(0, 1)
    show_cap = cap is not None and float(cap) > 0
    n = len(bands) + (1 if excluded else 0) + (1 if show_cap else 0)
    ax.set_ylim(0, n + 1.6)
    ax.axis("off")
    ax.text(0.0, n + 1.0, title, fontsize=fontsize + 2, fontweight="bold", va="center")
    ax.text(0.0, n + 0.35, f"green = within ±{pass_tol:g} mm", fontsize=fontsize - 1, color="0.35", va="center")
    rows = [(b["rgb"], f"{b['lo']:+.3f} … {b['hi']:+.3f}" + ("   (in tol)" if b.get("in_tol") else ""))
            for b in reversed(list(bands))]
    if show_cap:
        rows.append((cap_rgb, f"beyond cap |d| > {float(cap):g}" + (f"  ({int(n_cap):,})" if n_cap is not None else "")))
    if excluded:
        rows.append((GRAY, "not inspected" + (f"  ({int(n_excluded):,})" if n_excluded is not None else "")))
    for i, (rgb, lab) in enumerate(rows):
        y = n - 1 - i
        ax.add_patch(__import__("matplotlib.patches", fromlist=["Rectangle"]).Rectangle(
            (0.0, y + 0.08), 0.22, 0.84, color=np.asarray(rgb[:3], float) / 255.0, ec="0.3", lw=0.6))
        ax.text(0.27, y + 0.5, lab, fontsize=fontsize, va="center")


def plot_section(fig, ax, section, vmin, vmax, fontsize=12, point_size=4, cap=None, cap_rgb=CAP_RGB):
    """Section scatter coloured by deviation, true scale. Tall profiles are turned on their side so they
    fill a wide panel (the axis labels say which in-plane direction is which)."""
    u = np.asarray(section["u_mm"], float); v = np.asarray(section["v_mm"], float)
    c = section["deviation_mm"]
    xl, yl = "u (mm)", "v (mm)"
    if np.ptp(v) > 1.3 * max(np.ptp(u), 1e-9):
        u, v, xl, yl = v, u, "v (mm)", "u (mm)"
    c = np.asarray(c, float)
    bad = ~np.isfinite(c)
    over = np.zeros(len(c), bool)
    if cap is not None and float(cap) > 0:
        over = ~bad & (np.abs(np.where(bad, 0.0, c)) > float(cap))
    if bad.any():
        ax.scatter(u[bad], v[bad], c=[np.asarray(GRAY) / 255.0], s=point_size, rasterized=True,
                   label="not inspected")
    if over.any():
        ax.scatter(u[over], v[over], c=[np.asarray(cap_rgb[:3]) / 255.0], s=point_size * 1.6, rasterized=True,
                   label=f"beyond cap |d| > {float(cap):g}")
    if bad.any() or over.any():
        ax.legend(fontsize=fontsize - 2, loc="best", markerscale=3)
    ok = ~bad & ~over
    sc = ax.scatter(u[ok], v[ok], c=c[ok], s=point_size, cmap="coolwarm", vmin=vmin, vmax=vmax,
                    rasterized=True)
    ax.set_aspect("equal", adjustable="datalim")
    pad = 0.04 * max(np.ptp(u), np.ptp(v), 1e-6)
    ax.set_xlim(u.min() - pad, u.max() + pad)
    ax.set_xlabel(xl, fontsize=fontsize); ax.set_ylabel(yl, fontsize=fontsize)
    ax.tick_params(labelsize=fontsize - 1)
    cb = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
    cb.ax.tick_params(labelsize=fontsize - 1)
    cb.set_label("deviation (mm)", fontsize=fontsize - 1)
    return sc


def render_report(out_dir, mesh, rgb, dv, bands, stats, pass_tol, vmin, vmax, hist=True,
                  zones=None, probes=None, section=None, title="MeshSolid metrology", notes=None, frames=None,
                  cap=None, cap_rgb=CAP_RGB, n_cap=None, n_excluded=None, outlines=None):
    """One inspection sheet, readable on screen: two panels per row (views, legend, histogram, stats,
    probes, section), title 20 pt, headings 14 pt, tables 12 pt, 200 dpi."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image as PILImage

    out_dir = Path(out_dir)
    views = []
    frames = frames or view_frames(mesh)
    labels = {name: lab for name, _R, lab in frames}
    for name, R, _lab in frames:
        im = render_shaded(mesh, rgb, size=900, R=R, outlines=outlines)
        p = out_dir / f"heatmap_{name}.png"
        im.save(p)
        views.append((name, p, im))
    legend = out_dir / "legend.png"
    render_legend(bands, pass_tol, vmin, vmax, legend)

    has_sec = bool(section and section["n"])
    nrows = 5 if has_sec else 4
    fig = plt.figure(figsize=(12, 4.3 * nrows), dpi=100)
    gs = fig.add_gridspec(nrows, 2, height_ratios=[1.25, 1.25, 0.9, 1.0] + ([1.1] if has_sec else []),
                          hspace=0.32, wspace=0.12, left=0.06, right=0.97, top=1 - 0.5 / nrows / 4.3 * 2.2,
                          bottom=0.035)
    fig.suptitle(title, fontsize=20, fontweight="bold")
    for (name, p, im), cell in zip(views, (gs[0, 0], gs[0, 1], gs[1, 0])):
        ax = fig.add_subplot(cell)
        ax.imshow(im)
        ax.set_title(labels[name], fontsize=14)
        ax.axis("off")
    axl = fig.add_subplot(gs[1, 1])
    draw_legend(axl, bands, pass_tol, fontsize=13, n_excluded=n_excluded, cap=cap, n_cap=n_cap, cap_rgb=cap_rgb)

    axh = fig.add_subplot(gs[2, :])
    if hist:
        keep = dv[np.isfinite(dv)]
        bins = np.linspace(vmin, vmax, 41)
        axh.hist(np.clip(keep, vmin, vmax), bins=bins, color=(0.29, 0.39, 0.66), edgecolor="white", linewidth=0.3)
        axh.axvspan(-pass_tol, pass_tol, color=(0.18, 0.62, 0.31), alpha=0.2, label=f"pass band ±{pass_tol:g} mm")
        axh.axvline(0, color="0.2", lw=1.0)
        axh.set_xlim(vmin, vmax)
        axh.set_xlabel("signed deviation (mm)", fontsize=12)
        axh.set_ylabel("vertices", fontsize=12)
        axh.tick_params(labelsize=11)
        axh.set_title("histogram", fontsize=14)
        if stats:
            # the histogram is clipped to the colour scale: print the TRUE extremes beside it
            tp, tm = stats.get("true_max_plus", stats.get("max_plus")), stats.get("true_max_minus", stats.get("max_minus"))
            txt = f"true extremes  {tp:+.3f} / {tm:+.3f} mm"
            if stats.get("cap_mm"):
                txt += f"\n{stats.get('n_beyond_cap', 0):,} beyond the {stats['cap_mm']:g} mm cap (not in the histogram)"
            if (tp is not None and tp > vmax) or (tm is not None and tm < vmin):
                txt += f"\nbars at the ends hold everything beyond {vmin:+.3f} / {vmax:+.3f} mm"
            axh.text(0.01, 0.97, txt, transform=axh.transAxes, fontsize=11, va="top", ha="left",
                     bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.9))
        axh.legend(fontsize=11, loc="upper right")
    else:
        axh.axis("off")

    axt = fig.add_subplot(gs[3, 0])
    axt.axis("off")
    axt.set_title("statistics", fontsize=14, loc="left")
    rows = []
    if stats:
        rows = [
            ["verdict (target)", stats.get("result_label") or stats.get("result", "—")],
            ["  rule", "rms ≤ 0.03, ≥ 90 % in band (cap = fail), ≤ 5 % excl."],
            ["engine-limit (1.0.1)", (stats.get("result_engine_limit_label") or "—")
             + f"  (≥ {stats.get('accept_pct', 95):g} % in band)"],
            ["rms uncapped / in-band v1", f"{(stats.get('spec_metrics') or {}).get('rms_uncapped', float('nan')):.4f} mm / "
                                          f"{(stats.get('spec_metrics') or {}).get('pass_pct_v1', float('nan')):.2f} %"],
            ["rms", f"{stats['rms']:.4f} mm"],
            ["mean (signed)", f"{stats['mean']:+.4f} mm"],
            ["p95 |d|", f"{stats['p95']:.4f} mm"],
            ["max + / − (inspected)", f"{stats['max_plus']:+.3f} / {stats['max_minus']:+.3f} mm"],
            ["within ±{0:g} mm".format(pass_tol), f"{stats.get('within_pass_pct', 0):.2f} %"],
        ]
        if stats.get("cap_mm"):
            rows += [["true max + / −", f"{stats['true_max_plus']:+.3f} / {stats['true_max_minus']:+.3f} mm"],
                     [f"beyond cap |d|>{stats['cap_mm']:g}", f"{stats['n_beyond_cap']:,} ({stats['pct_beyond_cap']:.2f} %)"]]
        rows += [
            ["vertices used / excl.", f"{stats.get('n', 0):,} / {stats.get('n_excluded', 0):,}"],
        ]
    if zones:
        for z, s_ in zones.items():
            if s_:
                rows.append([f"zone {z}", f"rms {s_['rms']:.4f} mm (n={s_['n']:,})"])
    if rows:
        t = axt.table(cellText=rows[:20], colWidths=[0.42, 0.58], loc="upper left", cellLoc="left")
        t.auto_set_font_size(False)
        t.set_fontsize(12)
        t.scale(1, 1.9)

    axp = fig.add_subplot(gs[3, 1])
    axp.axis("off")
    axp.set_title("worst points / probes (mm)" + ("   * = beyond cap" if cap else ""), fontsize=14, loc="left")
    if probes:
        prow = [[f"{pr['xyz_mm'][0]:.2f}", f"{pr['xyz_mm'][1]:.2f}", f"{pr['xyz_mm'][2]:.2f}",
                 f"{pr['deviation_mm']:+.3f}" + (" *" if pr.get("beyond_cap") else "")] for pr in probes[:10]]
        t = axp.table(cellText=prow, colLabels=["x", "y", "z", "dev"], colWidths=[0.24, 0.24, 0.24, 0.28],
                      loc="upper left", cellLoc="right")
        t.auto_set_font_size(False)
        t.set_fontsize(12)
        t.scale(1, 1.55)
    else:
        axp.text(0.0, 0.9, "off", fontsize=12, color="0.4")

    if has_sec:
        axs = fig.add_subplot(gs[4, :])
        plot_section(fig, axs, section, vmin, vmax, fontsize=12, cap=cap, cap_rgb=cap_rgb)
        axs.set_title("section", fontsize=14)

    report_png = out_dir / "report.png"
    fig.savefig(report_png, dpi=200, facecolor="white")
    plt.close(fig)
    # also a standalone heatmap strip (iso + end + side)
    W = views[0][2].size[0]
    strip = PILImage.new("RGBA", (W * 3, W), (18, 20, 24, 255))
    for i, (_, p, im) in enumerate(views):
        strip.paste(im, (i * W, 0))
    strip_path = out_dir / "heatmap.png"
    strip.save(strip_path)
    return {"report_png": str(report_png), "heatmap_png": str(strip_path), "legend_png": str(legend)}


def write_band_stls(mesh, band_idx, bands, out_dir, cap=None, cap_rgb=CAP_RGB):
    """One binary STL per colour band (Fusion mesh bodies have no vertex colours; appearances colour these)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    face_band = band_idx[mesh.faces]
    # a face belongs to a band if any vertex does (and not all excluded)
    for i, b in enumerate(bands):
        use = (face_band == i).any(1) & (face_band >= 0).all(1)
        if not use.any():
            continue
        sub = mesh.submesh([use], append=True, repair=False)
        if sub is None or len(getattr(sub, "faces", [])) == 0:
            continue
        path = out_dir / f"band_{i:02d}_{b['color'][1:]}.stl"
        sub.export(path)
        written.append({"index": i, "path": str(path), "color": b["color"], "rgb": b["rgb"],
                        "lo": b["lo"], "hi": b["hi"], "faces": int(len(sub.faces))})
    over = (face_band == -2).any(1) & ~(face_band == -1).any(1)
    if over.any():
        sub = mesh.submesh([over], append=True, repair=False)
        if sub is not None and len(getattr(sub, "faces", [])):
            path = out_dir / "band_beyond_cap.stl"
            sub.export(path)
            written.append({"index": -2, "path": str(path), "color": "#%02x%02x%02x" % tuple(cap_rgb[:3]),
                            "rgb": list(cap_rgb[:3]), "lo": None, "hi": None, "beyond_cap": True,
                            "cap_mm": None if cap is None else float(cap), "faces": int(len(sub.faces))})
    ex = (face_band == -1).any(1)
    if ex.any():
        sub = mesh.submesh([ex], append=True, repair=False)
        if sub is not None and len(getattr(sub, "faces", [])):
            path = out_dir / "band_excluded.stl"
            sub.export(path)
            written.append({"index": -1, "path": str(path), "color": "#%02x%02x%02x" % GRAY, "rgb": list(GRAY),
                            "lo": None, "hi": None, "excluded": True, "faces": int(len(sub.faces))})
    return written


def write_csv(path, mesh, dv, band_idx, zones=None, excluded=None, beyond_cap=None, face=None, sector=None):
    """One row per scan vertex. deviation_mm is the exact signed deviation (positive = scan outside the solid),
    solid_face the BRep face index (TopExp order) of the nearest solid point (-1 if unknown), sector the
    blade / tooth index about the axis (-1 when the part is not N-fold)."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        hdr = ["vertex", "x_mm", "y_mm", "z_mm", "deviation_mm", "band", "excluded", "beyond_cap",
               "solid_face", "sector"]
        if zones is not None:
            hdr.append("zone")
        w.writerow(hdr)
        n = len(dv)
        ex = np.zeros(n, bool) if excluded is None else excluded
        bc = np.zeros(n, bool) if beyond_cap is None else beyond_cap
        fa = np.full(n, -1) if face is None else np.asarray(face, int)
        se = np.full(n, -1) if sector is None else np.asarray(sector, int)
        for i, p in enumerate(mesh.vertices):
            row = [i, f"{p[0]:.5f}", f"{p[1]:.5f}", f"{p[2]:.5f}", f"{dv[i]:.6f}",
                   int(band_idx[i]), int(ex[i]), int(bc[i]), int(fa[i]), int(se[i])]
            if zones is not None:
                row.append("" if ex[i] else str(zones[i]))
            w.writerow(row)


def run(scan_path, solid_path, out_dir, range_mm=None, symmetric=True, pass_tol=0.05, n_bands=7,
        band_json=None, exclude=(), max_dist=None, align=False, device="auto", lin=0.01,
        n_probes=10, probes=(), section=None, do_map=True, do_zones=True, do_histogram=True,
        do_section=True, zones_text="", exclude_faces=(), convert_report=None, auto_exclude=True,
        accept_pct=95.0, meta=None, pdf=True, cap_rgb=CAP_RGB, exclude_mask=None,
        exclude_mask_reason="external vertex mask", keep_arrays=False, ang=0.1, align_method="v2",
        n_fold=None, align_T0=None, align_min_points=None, auto_exclude_2mm=True):
    """meta: report header strings (title, part_name, operator). zones_text: typed exclusion zones.
    exclude_faces: STL files (mm) of solid faces to disregard. convert_report: the MeshSolid conversion
    report that built the solid (found beside the STEP when omitted); hand-entered features in it are
    excluded unless auto_exclude is False. auto_exclude_2mm: connected scan regions with |d| > 2 mm
    from the solid become named whole-region exclusions (default on). accept_pct: PASS when this % of inspected vertices is within
    the pass band. exclude_mask: optional per-scan-vertex bool array (e.g. the engine's own gate exclusions,
    for a controlled comparison) listed under exclude_mask_reason. keep_arrays: attach the per-vertex
    arrays (deviation, exclusion and cap masks) to the returned dict under "_arrays" (never written to JSON).
    align_method: "v1" (1.0.1 trimmed point-to-point ICP) or "v2" (spec v2: point-to-plane ICP on >= 500k
    points bounded to half a pitch, then a rigid refinement on the signed deviation of the inspected vertices
    and a direct re-measurement check; see meshsolid/align_v2.py). n_fold: blade / tooth count (else taken
    from the conversion report, else estimated); it bounds the v2 rotation and gives the per-blade sectors.
    align_T0: v2 start transform (default identity: MeshSolid builds the solid in the scan's frame)."""
    from . import metrology_exclude as X
    t0 = time.time()
    meta = dict(meta or {})
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    from . import device as D
    resolved = D.set_device(device)
    scan = load_scan(scan_path)
    solid_tm, solid_kind = load_solid(solid_path, lin=lin, ang=ang)
    notes = []
    T = None
    med = None
    work = scan
    from . import align_v2 as AV
    axis = rotation_axis(solid_tm)
    conv0 = convert_report or X.find_convert_report(solid_path)
    conv_matched = None
    if not conv0:
        conv0, _job = X.find_convert_report_for_scan(scan_path)
        if conv0:
            conv_matched = (f"Conversion report found by scan match: job {_job} converted this exact scan "
                            f"(byte-identical input); its hand-entered features and blade count are used.")
    nf_src = None
    if n_fold:
        nf_src = "given"
    else:
        try:
            crep = json.loads(Path(conv0).read_text()) if conv0 else {}
        except (OSError, ValueError):
            crep = {}
        nb = (crep.get("impeller") or {}).get("blades") or (crep.get("gear") or {}).get("teeth")
        if isinstance(nb, (int, float)) and nb >= 2:
            n_fold, nf_src = int(nb), "conversion report"
    if not n_fold:
        try:
            k, strength = AV.estimate_n_fold(solid_tm, axis)
            if k and strength > 0.02:
                n_fold, nf_src = int(k), f"estimated from the solid (strength {strength:.3f})"
        except Exception:  # noqa: BLE001
            pass
    align_info = None
    if align and align_method == "v2":
        kw = {} if align_min_points is None else {"min_points": int(align_min_points)}
        T, align_info = AV.align_v2(scan, solid_tm, n_fold=n_fold, axis=axis, T0=align_T0, **kw)
        work = scan.copy()
        work.apply_transform(T)
        notes.append(f"Spec-v2 alignment: point-to-plane ICP on {align_info['n_points']:,} points "
                     f"({align_info['iterations']} iterations, converged {align_info['converged']}), rotation about "
                     f"the axis bounded to half a pitch ({n_fold or '-'} -fold, {nf_src or 'no N'}); a final rigid "
                     "refinement on the signed deviation follows. The original scan and solid were not moved.")
    elif align:
        work, T, med = align_scan(scan, solid_tm)
        notes.append(f"Best-fit ICP was applied to a copy of the scan (median residual {med:.4f} mm). "
                     "The deviations are measured in the aligned frame; the coloured map, probes and CSV are written "
                     "back in the original scan frame, so the map sits on the unmoved scan. "
                     "The original scan and solid were not moved.")
    else:
        notes.append("No alignment: the solid is assumed to be in the scan's coordinates "
                     "(MeshSolid builds it that way). The original scan and solid were not moved.")
    notes.append("Interactive point picking on a mesh is not reliable in the Fusion API; "
                 "probes are the N worst vertices plus any coordinates typed in.")
    notes.append("Fusion mesh bodies do not keep per-vertex colours, so the map is one mesh body "
                 "per colour band with a solid appearance.")

    dv, cp, _tri = signed_distances(work, solid_tm)
    excluded = np.zeros(len(dv), bool)
    not_inspected = []

    def _exclude(mask, reason, detail, label=None):
        """n = vertices this exclusion matches; n_new = those not already excluded by an earlier one
        (so the n_new column adds up to the total not inspected)."""
        mask = np.asarray(mask, bool)
        new = int((mask & ~excluded).sum())
        excluded[:] |= mask
        rec = {"reason": reason, "label": label or reason, "detail": detail, "n": int(mask.sum()),
               "n_new": new, "pct_of_scan": round(100.0 * float(mask.sum()) / max(len(mask), 1), 3)}
        not_inspected.append(rec)
        return rec

    # hand-entered features (default on)
    conv = conv0
    if conv_matched:
        notes.append(conv_matched)
    features = X.hand_entered_features(conv)
    if features and auto_exclude:
        for z in X.hand_entered_zones(features, solid_path, solid_tm):
            _exclude(X.zone_mask(work.vertices, z), "hand-entered feature",
                     f"{z['label']}: scan inside {X.zone_text(z)}",
                     label=f"{z.get('kind', 'feature')} Ø{z['d_mm']:g} mm" if z.get("d_mm") else z["label"])
    elif features:
        notes.append("Hand-entered features were NOT excluded (--no-auto-exclude): " +
                     "; ".join(f["label"] for f in features))
    ff_zones = X.face_feature_zones(conv)
    if ff_zones and auto_exclude:
        for z in ff_zones:
            _exclude(X.zone_mask(work.vertices, z), z.get("reason", "face feature"),
                     z.get("detail") or X.zone_text(z),
                     label=z.get("label") or "end-face feature")
    elif ff_zones:
        notes.append("End-face features were NOT excluded (--no-auto-exclude): " +
                     "; ".join(z.get("label", "feature") for z in ff_zones))
    for k, fpath in enumerate(exclude_faces or []):
        fm = load_scan(fpath)
        flabel = X.face_label(fpath, k)
        _exclude(X.faces_mask(cp, fm, tol=max(0.03, 2.0 * float(lin))), "user face selection",
                 f"scan whose nearest solid point is on {flabel} ({len(fm.faces)} triangles, {Path(fpath).name})",
                 label=flabel)
    for k, z in enumerate(X.parse_zones(zones_text)):
        _exclude(X.zone_mask(work.vertices, z), "typed zone", X.zone_text(z), label=f"typed zone {k + 1} ({z['type']})")
    if exclude_mask is not None:
        em = np.asarray(exclude_mask, bool)
        if em.shape != excluded.shape:
            raise ValueError(f"exclude_mask has {em.size} entries, scan has {excluded.size} vertices")
        _exclude(em, exclude_mask_reason, f"{int(em.sum()):,} vertices from an external mask ({exclude_mask_reason})")
    tags = [t.strip() for t in (exclude or []) if str(t).strip()]
    zones = zkind = znote = None
    zone_rules = None
    if do_zones:
        try:
            _crep = json.loads(Path(conv0).read_text()) if conv0 else {}
        except (OSError, ValueError):
            _crep = {}
        rk, rsrc = part_type_from_report(_crep)
        zones, zkind, znote = classify_zones(work, solid_tm, kind=rk or "auto")
        auto_kind = None if rk is None else classify_zones(work, solid_tm, kind="auto")[1]
        zone_rules = {"kind": zkind, "source": rsrc if rk else f"geometric auto-detect ({rsrc})",
                      "geometric_guess": auto_kind}
        notes.append(f"Zone rules: {zkind} - {zone_rules['source']}"
                     + (f" (the geometric guess was '{auto_kind}', overridden)" if auto_kind and auto_kind != zkind else "")
                     + ". " + znote)
    # zone exclusion tags: each evaluated with the rules of ITS part type (before, ticking a gear tag made
    # every impeller tag silently match nothing), and every tag is listed, even when it matched nothing
    tag_map = {"impeller_back": "backface", "gear_tips": "tip"}
    kind_of = {"blades": "impeller", "hub": "impeller", "frontface": "impeller", "backface": "impeller",
               "bore": "impeller", "flank": "gear", "tip": "gear", "end_face": "gear"}
    by_kind = {}
    for t in tags:
        zname = tag_map.get(t, t)
        kind = kind_of.get(zname)
        if kind is None:
            notes.append(f"Unknown zone exclusion '{t}' was ignored.")
            continue
        if kind not in by_kind:
            if zones is not None and zkind == kind:
                by_kind[kind] = zones
            else:
                by_kind[kind] = classify_zones(work, solid_tm, kind=kind)[0]
        zk = by_kind[kind]
        m = (zk == zname) if zk is not None else np.zeros(len(dv), bool)
        hint = ""
        if zkind not in (None, "none") and zkind != kind:
            hint = f" Note: the part was recognised as {'an' if zkind[0] in 'aeiou' else 'a'} {zkind}, not {'an' if kind[0] in 'aeiou' else 'a'} {kind}."
        if zname == "tip":
            hint += (" On an impeller the 'tip' band is the outer rim / blade tips (r >= 92 % of the outer radius), "
                     "not the nose cone - use 'Disregard faces' for the cone.")
        label = f"'{zname}' zone ({kind})"
        if zname == "tip" and zkind == "impeller":
            # the gear 'tip' rule on an impeller removes the outer rim and the blade tips: say so in the label
            label = "outer rim + blade tips, r >= 92 % of outer radius (gear 'tip' rule)"
        _exclude(m, "zone exclusion", f"'{zname}' zone by the geometric {kind} rules: {int(m.sum()):,} vertices.{hint}",
                 label=label)
    if align_info is not None:
        dT, rinfo = AV.refine_on_deviation(work.vertices, solid_tm, mask=~excluded)
        T = dT @ T
        work = scan.copy()
        work.apply_transform(T)
        dv, cp, _tri = signed_distances(work, solid_tm)
        align_info["refinement"] = rinfo
        align_info["direct_check"] = AV.direct_check(work.vertices, solid_tm, mask=~excluded, axis=axis[1])
        if rinfo.get("warning"):
            notes.append("Alignment warning: " + rinfo["warning"])
        if not align_info["direct_check"]["ok"]:
            notes.append("Alignment warning: the direct re-measurement found a lower rms at a perturbed pose.")
    # 2 mm auto-exclusion: connected scan regions more than 2 mm from the solid, any part.
    # Named, listed, greyed with their boundary on the map; kept out of a cheap second alignment refine.
    ex2 = {"threshold_mm": 2.0, "regions": [], "n_regions": 0, "n_points": 0, "pct_of_scan": 0.0,
           "over_5pct": False, "over_5pct_flag": None, "skipped": "auto_exclude_2mm=off"}
    ex2_outlines = []
    if auto_exclude_2mm:
        from .exclude_2mm import detect as _detect_2mm, outlines as _outlines_2mm, public_report as _pub_2mm
        raw2, m2 = _detect_2mm(work.vertices, dv, mesh=work, already_excluded=excluded)
        for r in raw2["regions"]:
            rm = np.zeros(len(dv), bool)
            rm[np.asarray(r["_ix"], int)] = True
            rec = _exclude(rm, "auto-exclusion |d| > 2 mm",
                           f"{r['reason']}; area {r['area_mm2']:.1f} mm², "
                           f"{r['pct_of_scan']:.2f} % of the scan, max |d| {r['depth_max_mm']:.2f} mm "
                           f"({r['sign']})",
                           label=r["label"])
            rec["area_mm2"] = r["area_mm2"]
            rec["depth_max_mm"] = r["depth_max_mm"]
        if m2.any() and align_info is not None:
            dT, rinfo = AV.refine_on_deviation(work.vertices, solid_tm, mask=~excluded)
            T = dT @ T
            work = scan.copy()
            work.apply_transform(T)
            dv, cp, _tri = signed_distances(work, solid_tm)
            align_info["refinement_after_2mm"] = rinfo
        if raw2.get("over_5pct"):
            notes.append("Exclusion flag: " + raw2["over_5pct_flag"])
        if raw2["n_regions"]:
            notes.append(f"Auto-excluded {raw2['n_regions']} connected region(s) with |d| > 2 mm "
                         f"({raw2['n_points']:,} vertices, {raw2['pct_of_scan']:.2f} % of the scan).")
        ex2 = _pub_2mm(raw2)
        ex2_outlines = _outlines_2mm(ex2)
    # outlier cap: |d| beyond it is left out of the statistics but reported with its TRUE values
    measured = ~excluded.copy()                 # inspected before the cap (face / zone exclusions removed)
    cap = float(max_dist) if (max_dist is not None and float(max_dist) > 0) else None
    beyond_cap = np.zeros(len(dv), bool)
    cap_info = None
    cap_rgb = tuple(int(x) for x in cap_rgb[:3])
    if cap is not None:
        beyond_cap = measured & (np.abs(dv) > cap)
        nb = int(beyond_cap.sum())
        dm = dv[measured]
        cap_info = {"cap_mm": cap, "n_beyond_cap": nb,
                    "pct_beyond_cap": round(100.0 * nb / max(int(measured.sum()), 1), 3),
                    "n_beyond_plus": int((beyond_cap & (dv > 0)).sum()),
                    "n_beyond_minus": int((beyond_cap & (dv < 0)).sum()),
                    "true_max_plus": round(float(dm.max()), 5) if dm.size else None,
                    "true_max_minus": round(float(dm.min()), 5) if dm.size else None}
        _exclude(beyond_cap, "beyond outlier cap", label=f"outlier cap |d| > {cap:g} mm", detail=(
                 f"|d| > {cap:g} mm: {nb:,} points ({cap_info['pct_beyond_cap']:.2f} % of inspected; "
                 f"{cap_info['n_beyond_plus']:,} above +{cap:g}, {cap_info['n_beyond_minus']:,} below -{cap:g}). "
                 f"True extremes {cap_info['true_max_plus']:+.3f} / {cap_info['true_max_minus']:+.3f} mm. "
                 f"Left out of the statistics; shown in the cap colour"))
        notes.append(f"Outlier cap {cap:g} mm: {nb:,} points with |d| above it were left out of the statistics "
                     f"(listed under 'beyond outlier cap', shown in the cap colour). The worst-points table and "
                     f"'true max' show their real deviations.")
    face_zone_excluded = excluded & ~beyond_cap

    used = dv[~excluded]
    if range_mm is None:
        span = float(np.percentile(np.abs(used), 99)) if used.size else 0.2
        span = max(span, float(pass_tol) * 2, 1e-3)
        vmin, vmax = (-span, span) if symmetric else (float(used.min()), float(used.max())) if used.size else (-span, span)
    else:
        vmin, vmax = float(range_mm[0]), float(range_mm[1])
        if symmetric:
            m = max(abs(vmin), abs(vmax), 1e-9)
            vmin, vmax = -m, m
    bands = load_bands(band_json, vmin, vmax, pass_tol, n_bands)
    clashes = colour_clashes(bands, cap_rgb if cap is not None else None,
                             GRAY if bool(face_zone_excluded.any()) else None)
    for c in clashes:
        notes.append("Colour warning: " + c["text"] + ". Colours are drawn exactly as chosen; pick another colour "
                     "if they are hard to tell apart.")
    rgb, band_idx = vertex_colors(dv, bands, face_zone_excluded, beyond_cap, cap_rgb)
    st = _stats(used)
    if st:
        st["within_pass_pct"] = round(float(100.0 * np.mean(np.abs(used) <= abs(pass_tol))), 3) if used.size else 0.0
        st["n_excluded"] = int(excluded.sum())
        st["pass_tol_mm"] = float(pass_tol)
        st["accept_pct"] = float(accept_pct)
        from .metrology_spec import SPECS, metrics as _metrics, verdict as _verdict
        _spec = SPECS["v2" if align_method == "v2" else "v1"]
        _cap_v = beyond_cap if cap is not None else (~face_zone_excluded & (np.abs(dv) > _spec["outlier_cap"]["cap_mm"]))
        sm = _metrics(dv, face_zone_excluded, _cap_v, tol=pass_tol)
        vd = _verdict(sm, st["within_pass_pct"], accept_pct, spec=_spec)
        st["spec_metrics"] = sm
        try:
            from .support_gate import solid_to_scan, merge_into_verdict
            st["solid_to_scan_gate"] = solid_to_scan(work, solid_tm, excluded_vertices=face_zone_excluded)
            merge_into_verdict(vd, st["solid_to_scan_gate"])
        except Exception as _e:  # noqa: BLE001
            st["solid_to_scan_gate"] = {"error": f"{type(_e).__name__}: {_e}"}
        st["verdict"] = vd
        st["pct_excluded"] = sm["pct_excluded"]
        st["excluded_over_5pct"] = bool(float(sm["pct_excluded"]) > 5.0)
        st["auto_exclusions_2mm"] = {k: ex2[k] for k in ("n_regions", "n_points", "pct_of_scan",
                                                         "over_5pct", "over_5pct_flag", "threshold_mm") if k in ex2}
        st["result"] = vd["target"]["result"]                       # the honest verdict
        st["result_label"] = vd["target"]["label"]
        st["result_engine_limit"] = vd["engine_limit"]["result"]    # 1.0.1 rule, PASS refused above 5 % excluded
        st["result_engine_limit_label"] = vd["engine_limit"]["label"]
        st["n_not_inspected"] = int(face_zone_excluded.sum())
        st["excluded_breakdown"] = [{k: e[k] for k in ("reason", "label", "n", "n_new", "pct_of_scan")}
                                    for e in not_inspected]
        by_reason = {}
        for e in not_inspected:
            by_reason[e["reason"]] = by_reason.get(e["reason"], 0) + e["n_new"]
        st["excluded_by_reason"] = by_reason
        if cap_info:
            st.update(cap_info)
        else:
            st["true_max_plus"], st["true_max_minus"] = st["max_plus"], st["max_minus"]
    if used.size:
        bidx = assign_bands(used, bands)
        for i, b in enumerate(bands):
            b["pct"] = round(float(100.0 * np.mean(bidx == i)), 3)
    zone_stats = None
    if do_zones and zones is not None:
        zone_stats = {}
        for z in sorted(set(zones)):
            zone_stats[z] = _stats(dv[(zones == z) & ~excluded])
        if st:
            st["zones"] = zone_stats

    # outputs (map, bands, probes, section, csv) are written in the ORIGINAL scan frame: vertex i of the aligned copy
    # is vertex i of the scan, so the coloured map sits on the unmoved scan body and typed points are user-frame
    outm = scan
    probe_rows = worst_probes(outm, dv, n=n_probes, excluded=face_zone_excluded, extra_points=probes, cap=cap)
    sec = None
    if do_section and section:
        origin, normal = section
        sec = section_points(outm, np.where(face_zone_excluded, np.nan, dv), origin, normal)   # NaN = grey
        # drop arrays from the JSON later; keep a thin summary
    files = {}
    ply = out_dir / "deviation.ply"
    mcol = trimesh.Trimesh(outm.vertices, outm.faces, vertex_colors=rgb, process=False)
    mcol.export(ply)
    files["deviation_ply"] = str(ply)
    frames = view_frames(work, axis=rotation_axis(solid_tm)[1])
    report_views = {name: lab for name, _R, lab in frames}
    if do_map:
        files["bands"] = write_band_stls(outm, band_idx, bands, out_dir / "bands", cap=cap, cap_rgb=cap_rgb)
        _ttl = (meta.get("title") or "MeshSolid metrology") + (f": {meta['part_name']}" if meta.get("part_name") else "")
        files.update(render_report(out_dir, work, rgb, dv[~excluded], bands, st, pass_tol, vmin, vmax,
                                   hist=do_histogram, zones=zone_stats, probes=probe_rows,
                                   section=sec if (sec and sec["n"]) else None,
                                   title=_ttl, notes=notes, frames=frames, cap=cap, cap_rgb=cap_rgb,
                                   n_cap=int(beyond_cap.sum()), n_excluded=int(face_zone_excluded.sum()),
                                   outlines=ex2_outlines or None))
    elif do_histogram or True:
        # still write a legend + histogram-only sheet when the map is off
        files["legend_png"] = render_legend(bands, pass_tol, vmin, vmax, out_dir / "legend.png")

    face_of = None
    if solid_tm.metadata.get("brep_face") is not None:
        face_of = np.asarray(solid_tm.metadata["brep_face"], int)[np.asarray(_tri, int)]
    sector = None
    sector_info = None
    if n_fold:
        try:
            field = AV.SectorField(solid_tm, axis, n_fold, nz=8 if n_fold <= 20 else 64)
            sector = field(work.vertices)
            sector_info = {"n_fold": int(n_fold), "source": nf_src,
                           "method": "angular sector about the axis, centred on blade/tooth k, following the "
                                     "blade wrap (phase of the N-th harmonic of tangential-normal area per r,z bin)"}
        except Exception as e:  # noqa: BLE001
            sector_info = {"n_fold": int(n_fold), "source": nf_src, "error": f"{type(e).__name__}: {e}"}
    insp = ~excluded
    per_face = None
    if face_of is not None:
        types = solid_tm.metadata.get("brep_face_types") or []
        per_face = []
        for fid in np.unique(face_of[insp]):
            m = insp & (face_of == fid)
            s_ = _stats(dv[m])
            if s_:
                s_["face"] = int(fid)
                s_["type"] = types[fid] if 0 <= fid < len(types) else "?"
                s_["n_beyond_cap"] = int((m & (np.abs(dv) > cap)).sum()) if cap else 0
                per_face.append(s_)
        per_face.sort(key=lambda r: -r["rms"])
    per_sector = None
    if sector is not None:
        nrm = np.asarray(work.vertex_normals, float)
        c0, u0 = axis
        V0 = np.asarray(work.vertices, float) - c0
        rad = V0 - np.outer(V0 @ u0, u0)
        rad /= np.maximum(np.linalg.norm(rad, axis=1), 1e-12)[:, None]
        tang = np.abs(np.einsum("ij,ij->i", nrm, np.cross(u0, rad))) > 0.5
        per_sector = []
        for k in range(int(n_fold)):
            m = insp & (sector == k)
            row = {"sector": k, "all": _stats(dv[m]), "blade_surfaces": _stats(dv[m & tang])}
            per_sector.append(row)
    csv_path = out_dir / "deviation.csv"
    write_csv(csv_path, outm, dv, band_idx, zones, face_zone_excluded, beyond_cap, face=face_of, sector=sector)
    files["csv"] = str(csv_path)

    sec_json = None
    if sec:
        sec_json = {"n": sec["n"], "origin": sec["origin"], "normal": sec["normal"], "thick_mm": sec["thick_mm"]}
        np.savez(out_dir / "section.npz", u=sec["u_mm"], v=sec["v_mm"], d=sec["deviation_mm"], xyz=sec["xyz_mm"])
        files["section_npz"] = str(out_dir / "section.npz")

    report = {
        "status": "ok",
        "scan": str(scan_path),
        "solid": str(solid_path),
        "solid_kind": solid_kind,
        "device": D.report(),
        "device_resolved": resolved,
        "align": None if T is None else {"transform": np.round(T, 6).tolist(), **pose_of(T),
                                       "icp_median_residual_mm_pre_alignment_sampled": med},
        "scale": {"vmin": vmin, "vmax": vmax, "symmetric": bool(symmetric), "pass_tol_mm": float(pass_tol)},
        "bands": bands,
        "stats": st,
        "zones_kind": zkind,
        "zone_rules": zone_rules,
        "zones": zone_stats,
        "probes": probe_rows,
        "section": sec_json,
        "excluded_tags": tags,
        "n_excluded": int(excluded.sum()),
        "not_inspected": not_inspected,
        "auto_exclusions_2mm": ex2,
        "hand_entered": features,
        "convert_report": conv,
        "zones_typed": zones_text or "",
        "meta": meta,
        "settings": {"range_mm": [vmin, vmax], "symmetric": bool(symmetric), "pass_tol_mm": float(pass_tol),
                     "n_bands": int(len(bands)), "accept_pct": float(accept_pct), "align": bool(align),
                     "device": device, "lin_mm": float(lin), "ang_rad": float(ang), "n_probes": int(n_probes),
                     "typed_probes": [list(map(float, q)) for q in (probes or [])],
                     "section": None if not section else [list(map(float, section[0])), list(map(float, section[1]))],
                     "exclude_tags": tags, "max_dist_mm": None if max_dist is None else float(max_dist),
                     "outlier_cap_mm": cap, "cap_color": "#%02x%02x%02x" % cap_rgb,
                     "zones": zones_text or "", "exclude_faces": [str(f) for f in exclude_faces or []],
                     "auto_exclude_hand_entered": bool(auto_exclude),
                     "auto_exclude_2mm": bool(auto_exclude_2mm), "map": bool(do_map),
                     "zones_stats": bool(do_zones), "histogram": bool(do_histogram)},
        "files": files,
        "notes": notes,
        "seconds": round(time.time() - t0, 3),
        "watertight_solid": bool(getattr(solid_tm, "is_watertight", False)),
        "views": report_views,
        "alignment": align_info,
        "pose": pose_of(T),
        "sectors": sector_info,
        "per_face": per_face,
        "per_sector": per_sector,
    }
    if not report["watertight_solid"]:
        report["notes"].append("The solid tessellation is not watertight; the sign of the distance "
                               "uses the tessellated face normal and may flip on open edges.")
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=_js))
    files["report_json"] = str(out_dir / "report.json")
    if pdf:
        try:
            from .metrology_pdf import write_pdf
            if not (out_dir / "heatmap_iso.png").is_file():
                for name, R, _lab in frames:
                    render_shaded(work, rgb, size=900, R=R).save(out_dir / f"heatmap_{name}.png")
            if not (out_dir / "legend.png").is_file():
                render_legend(bands, pass_tol, vmin, vmax, out_dir / "legend.png")
            files["report_pdf"] = write_pdf(report, out_dir, dv=dv[~excluded], section=sec)
        except Exception as e:  # noqa: BLE001  (the PDF never loses the measurement)
            import traceback
            report["notes"].append(f"PDF report could not be written: {e}")
            (out_dir / "pdf_error.txt").write_text(traceback.format_exc())
    report["files"] = files
    report["seconds"] = round(time.time() - t0, 3)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=_js))
    if keep_arrays:
        report["_arrays"] = {"dv": dv, "excluded": excluded.copy(), "beyond_cap": beyond_cap.copy(),
                             "not_inspected_mask": face_zone_excluded.copy(), "T": T, "face": face_of,
                             "sector": sector, "points": np.asarray(work.vertices, float),
                             "normals_solid": np.asarray(solid_tm.face_normals, float)[np.asarray(_tri, int)],
                             "axis": axis, "n_fold": n_fold}
    return report


def _hex_rgb(s):
    s = str(s).strip().lstrip("#")
    if len(s) != 6:
        raise SystemExit(f"--cap-color: expected #rrggbb, got {s!r}")
    return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))


def _parse_xyz(s):
    p = [float(x) for x in s.replace(";", ",").split(",") if x.strip()]
    if len(p) != 3:
        raise argparse.ArgumentTypeError(f"expected x,y,z (got {s!r})")
    return p


def _parse_section(s):
    # ORIGIN:NORMAL  e.g. 0,0,0:0,0,1
    if ":" not in s:
        raise argparse.ArgumentTypeError("section must be origin:normal, e.g. 0,0,0:0,0,1")
    a, b = s.split(":", 1)
    return _parse_xyz(a), _parse_xyz(b)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m meshsolid.metrology",
                                 description="Signed deviation metrology of a scan against a solid (STEP or mesh).")
    ap.add_argument("scan", help="scan mesh (STL/OBJ/PLY) or a MeshSolid job folder")
    ap.add_argument("solid", nargs="?", help="solid STEP (or mesh). Omit when SCAN is a job folder.")
    ap.add_argument("--out", required=True, help="output folder (report.json, report.png, bands/, CSV)")
    ap.add_argument("--range", nargs=2, type=float, metavar=("MIN", "MAX"),
                    help="colour scale in mm (default: symmetric 99th-percentile of |d|)")
    ap.add_argument("--symmetric", action="store_true", default=True)
    ap.add_argument("--no-symmetric", action="store_false", dest="symmetric")
    ap.add_argument("--pass", dest="pass_tol", type=float, default=0.05, help="pass band ±tol mm (default 0.05)")
    ap.add_argument("--bands", type=int, default=7, help="number of colour bands (default 7)")
    ap.add_argument("--band-json", help="JSON file of [{lo,hi,color}] overriding the default palette")
    ap.add_argument("--exclude", default="", help="comma tags: backface,bore,tip,impeller_back,gear_tips")
    ap.add_argument("--max-dist", "--outlier-cap", dest="max_dist", type=float, default=None,
                    help="outlier cap (mm): |d| above it is left out of the statistics, listed as 'beyond outlier "
                         "cap' with its count, drawn in --cap-color; the true max is still reported")
    ap.add_argument("--cap-color", default="#ff0000", help="colour of points beyond the outlier cap (default red)")
    ap.add_argument("--align", action="store_true", help="best-fit ICP of a scan copy onto the solid (off by default)")
    ap.add_argument("--align-method", choices=["v1", "v2"], default="v2",
                    help="v1 = 1.0.1 ICP; v2 = spec-v2 point-to-plane, bounded, refined, verified")
    ap.add_argument("--n-fold", type=int, default=None, help="blade / tooth count (default: report or estimate)")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--lin", type=float, default=0.01, help="STEP tessellation linear deflection (mm)")
    ap.add_argument("--probes", type=int, default=10, help="auto-label this many worst |d| vertices (default 10)")
    ap.add_argument("--probe", action="append", default=[], type=_parse_xyz, help="typed probe x,y,z (repeatable, mm)")
    ap.add_argument("--section", type=_parse_section, default=None, help="cross-section plane origin:normal (mm)")
    ap.add_argument("--no-map", action="store_true")
    ap.add_argument("--no-zones", action="store_true")
    ap.add_argument("--no-histogram", action="store_true")
    ap.add_argument("--no-section-plot", action="store_true")
    ap.add_argument("--zones", default="", help="typed exclusion zones: 'box(x0,y0,z0,x1,y1,z1); cyl(px,py,pz,dx,dy,dz,r[,t0,t1])'")
    ap.add_argument("--exclude-faces", action="append", default=[], help="STL (mm) of solid faces to disregard (repeatable)")
    ap.add_argument("--convert-report", default=None, help="MeshSolid conversion report of the solid (default: beside the STEP)")
    ap.add_argument("--no-auto-exclude", action="store_true", help="inspect hand-entered features too")
    ap.add_argument("--no-auto-exclude-2mm", action="store_true",
                    help="do not auto-exclude connected regions with |d| > 2 mm from the solid")
    ap.add_argument("--accept-pct", type=float, default=95.0, help="PASS when this %% of vertices is within the pass band")
    ap.add_argument("--title", default="Metrology inspection report")
    ap.add_argument("--part-name", default="")
    ap.add_argument("--operator", default="")
    ap.add_argument("--no-pdf", action="store_true")
    a = ap.parse_args(argv)
    scan, solid = a.scan, a.solid
    if solid is None:
        job = find_job_files(scan)
        if not job["scan"] or not job["solid"]:
            print("job folder does not contain a scan mesh and a STEP", file=sys.stderr)
            return 2
        scan, solid = job["scan"], job["solid"]
    tags = [t.strip() for t in a.exclude.split(",") if t.strip()]
    from .metrology_exclude import parse_zones
    try:
        parse_zones(a.zones)
    except ValueError as e:
        print(f"--zones: {e}", file=sys.stderr)
        return 2
    try:
        from . import device as D
        if a.device != "cpu":
            D.set_device(a.device)
    except RuntimeError as e:
        print(e, file=sys.stderr)
        return 2
    rep = run(scan, solid, a.out, range_mm=a.range, symmetric=a.symmetric, pass_tol=a.pass_tol,
              n_bands=a.bands, band_json=a.band_json, exclude=tags, max_dist=a.max_dist,
              cap_rgb=_hex_rgb(a.cap_color),
              align=a.align, align_method=a.align_method, n_fold=a.n_fold, device=a.device, lin=a.lin, n_probes=a.probes, probes=a.probe,
              section=None if a.no_section_plot else a.section,
              do_map=not a.no_map, do_zones=not a.no_zones, do_histogram=not a.no_histogram,
              do_section=not a.no_section_plot, zones_text=a.zones, exclude_faces=a.exclude_faces,
              convert_report=a.convert_report, auto_exclude=not a.no_auto_exclude,
              auto_exclude_2mm=not a.no_auto_exclude_2mm, accept_pct=a.accept_pct,
              meta={"title": a.title, "part_name": a.part_name or Path(solid).stem, "operator": a.operator},
              pdf=not a.no_pdf)
    st = rep.get("stats") or {}
    print(f"status {rep['status']}  rms {st.get('rms')} mm  p95 {st.get('p95')} mm  "
          f"within ±{a.pass_tol:g}: {st.get('within_pass_pct')}%  device {rep.get('device_resolved')}  "
          f"{rep.get('seconds')} s")
    print("report", rep["files"].get("report_json"))
    if rep["files"].get("report_pdf"):
        print("pdf", rep["files"]["report_pdf"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
