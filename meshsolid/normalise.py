"""Pre-recognition normalise stage: part axis to +Z, nose up, units/scale, shaft / fixture detection.

estimate(mesh) returns the exact 4x4 transform N (scan frame -> normalised frame) and what it found. With
Params.normalise == "apply" the pipeline works on N @ mesh and, at the end, maps the STEP and the deviation
map back with N^-1, so every output is in the user's original scan frame (restore_outputs). With "report"
(default until the accuracy set has been run both ways) N is computed and stored only.

Normalised frame: origin on the rotation axis at the lowest point of the part, +Z along the axis towards the
nose (the end whose cross-section is smaller, e.g. an impeller's eye), +X along the first non-axial PCA axis.
Units: a part whose largest extent is below 1.0 is taken as metres (x1000 to mm); anything else is mm. An
inch export cannot be told from a small mm part by size alone, so it is flagged (possible_inches) but never
rescaled."""
from __future__ import annotations

import numpy as np


def _rot_axis(C, A):
    """Area-weighted PCA of face centroids: (centre, axis, strength). strength = how alike the two radial
    eigenvalues are (1 = perfectly rotational)."""
    w = A / max(A.sum(), 1e-30)
    c = (C * w[:, None]).sum(0)
    X = C - c
    cov = (X * w[:, None]).T @ X
    ev, V = np.linalg.eigh(cov)
    ev = np.maximum(ev, 1e-15)
    # the odd one out is the axis; the other two should match for a rotational part
    odd = [abs(np.log(ev[i] / (0.5 * (ev[(i + 1) % 3] + ev[(i + 2) % 3])))) for i in range(3)]
    i = int(np.argmax(odd))
    j, k = [x for x in range(3) if x != i]
    strength = float(min(ev[j], ev[k]) / max(ev[j], ev[k]))
    return c, V[:, i] / np.linalg.norm(V[:, i]), V[:, j], strength


def _components(mesh):
    try:
        import trimesh
        lab = trimesh.graph.connected_component_labels(mesh.face_adjacency, node_count=len(mesh.faces))
        return lab
    except Exception:  # noqa: BLE001
        return np.zeros(len(mesh.faces), int)


def estimate(mesh, rot_strength_min=0.9):
    V = np.asarray(mesh.vertices, float)
    C = np.asarray(mesh.triangles_center, float)
    A = np.asarray(mesh.area_faces, float)
    ext = float(np.ptp(V, axis=0).max()) if len(V) else 0.0
    info = {"largest_extent": round(ext, 6)}
    scale = 1.0
    if 0 < ext < 1.0:
        scale, info["units"] = 1000.0, "m (largest extent < 1, scaled x1000 to mm)"
    else:
        info["units"] = "mm"
        if 2.0 <= ext <= 20.0:
            info["possible_inches"] = True        # flagged only: a small mm part looks the same
    c, u, e1, strength = _rot_axis(C, A)
    info["rotational_strength"] = round(strength, 4)
    rotational = strength >= rot_strength_min
    info["rotational"] = bool(rotational)
    z = (V - c) @ u
    r = np.linalg.norm((V - c) - np.outer(z, u), axis=1)
    lo, hi = np.percentile(z, [1, 99])
    band = 0.15 * (hi - lo)
    r_top = float(np.median(r[z > hi - band])) if np.any(z > hi - band) else 0.0
    r_bot = float(np.median(r[z < lo + band])) if np.any(z < lo + band) else 0.0
    if r_bot < r_top:                         # nose (smaller end) must be +Z
        u, z = -u, -z
        r_top, r_bot = r_bot, r_top
    info["nose"] = {"r_nose_end": round(r_top, 4), "r_other_end": round(r_bot, 4)}
    e1 = e1 - (e1 @ u) * u
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(u, e1)
    zmin = float(z.min())
    o = c + zmin * u
    R = np.vstack([e1, e2, u])                # rows: new axes in scan coordinates
    N = np.eye(4)
    N[:3, :3] = scale * R
    N[:3, 3] = -scale * (R @ o)
    info["axis_scan_frame"] = {"point": np.round(o, 6).tolist(), "dir": np.round(u, 9).tolist()}
    # shaft / stub detection along the axis: an end section much thinner than the part
    Rm = float(np.percentile(r, 99))
    zz = (z - zmin)
    H = float(zz.max()) if len(zz) else 0.0
    shafts = []
    if rotational and H > 0:
        nb = 40
        edges = np.linspace(0, H, nb + 1)
        rmax = np.array([np.percentile(r[(zz >= a) & (zz < b)], 99) if np.sum((zz >= a) & (zz < b)) > 20 else np.nan
                         for a, b in zip(edges[:-1], edges[1:])])
        for side, rng in (("nose", range(nb - 1, -1, -1)), ("base", range(nb))):
            run = 0
            for i in rng:
                if np.isfinite(rmax[i]) and rmax[i] < 0.25 * Rm:
                    run += 1
                else:
                    break
            if run * H / nb > 0.08 * H:
                shafts.append({"end": side, "length": round(run * H / nb, 3),
                               "radius": round(float(np.nanmax(rmax[list(rng)[:run]])), 3)})
    info["shaft_candidates"] = shafts
    lab = _components(mesh)
    ncomp = int(lab.max()) + 1 if len(lab) else 0
    small = []
    if ncomp > 1:
        tot = A.sum()
        for k in range(ncomp):
            a = float(A[lab == k].sum())
            if a < 0.01 * tot:
                small.append({"component": k, "area_frac": round(a / tot, 5), "faces": int((lab == k).sum())})
    info["components"] = ncomp
    info["fixture_candidates"] = small
    return N, info, lab


def apply(mesh, mode="report"):
    """-> (mesh in the normalised frame (or unchanged for 'report'), report dict with the 4x4 transform)."""
    N, info, lab = estimate(mesh)
    rep = {"mode": mode, "transform_scan_to_normalised": np.round(N, 12).tolist(),
           "transform_normalised_to_scan": np.round(np.linalg.inv(N), 12).tolist(), **info, "applied": False}
    if mode != "apply":
        return mesh, rep
    m = mesh.copy()
    stripped = 0
    if info["fixture_candidates"]:
        keep = np.ones(len(m.faces), bool)
        for fc in info["fixture_candidates"]:
            keep &= lab != fc["component"]
        stripped = int((~keep).sum())
        if stripped:
            m.update_faces(keep)
            m.remove_unreferenced_vertices()
    m.apply_transform(N)
    rep["applied"] = True
    rep["fixture_faces_stripped"] = stripped
    return m, rep


def decimate_for_recognition(mesh, target=300_000):
    """1.0.3: reduce a mesh to about `target` triangles for recognition using quadric edge collapse
    (pymeshlab).  Returns a new Trimesh with at most `target` faces, preserving topology and normals
    as far as the decimator allows.  Raises RuntimeError if pymeshlab is unavailable or decimation
    fails; the caller falls back to the undecimated mesh."""
    import pymeshlab
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(mesh.vertices, mesh.faces), mesh="in")
    ms.apply_filter("simplification_quadric_edge_collapse_decimation",
                    targetfacenum=target, preservetopology=True)
    out = ms.current_mesh()
    return trimesh.Trimesh(vertices=out.vertex_matrix(), faces=out.face_matrix(),
                           process=False)


def check_units(mesh):
    """1.0.3: flag scale/units ambiguity.  Returns a dict with the largest extent, the assumed unit,
    and a flag when the part is large enough that mm vs inches is ambiguous (e.g. ~1,650 mm could be
    a 1,650 mm part or a ~65 inch part exported as mm).  Never rescalts automatically."""
    ext = float(np.linalg.norm(np.ptp(np.asarray(mesh.vertices, axis=0), axis=0)))
    info = {"largest_extent_mm": round(ext, 3)}
    if ext >= 1000.0:
        info["unit"] = "mm"
        info["ambiguous_inches"] = True
        info["note"] = (f"part is {ext:.0f} mm across; if the source was inches this is "
                        f"{ext / 25.4:.0f} in — confirm units before build")
    elif 2.0 <= ext <= 20.0:
        info["unit"] = "mm"
        info["possible_inches"] = True
        info["note"] = "small part; a mm export and an inch export look the same by size alone"
    elif 0 < ext < 1.0:
        info["unit"] = "m (scaled x1000 to mm)"
        info["ambiguous_inches"] = False
        info["note"] = "largest extent < 1: interpreted as metres and scaled to mm"
    else:
        info["unit"] = "mm"
        info["ambiguous_inches"] = False
        info["note"] = "mm"
    return info


def _trsf(M):
    """gp_Trsf for x -> M[:3,:3] x + M[:3,3] with M[:3,:3] = s * rotation (uniform scale s)."""
    from OCP.gp import gp_Trsf
    s = float(np.cbrt(abs(np.linalg.det(M[:3, :3]))))
    Rm = M[:3, :3] / s
    rigid = gp_Trsf()
    rigid.SetValues(*[float(x) for x in np.c_[Rm, M[:3, 3]].ravel()])
    if abs(s - 1.0) <= 1e-12:
        return rigid
    sc = gp_Trsf()
    sc.SetScaleFactor(s)
    return rigid.Multiplied(sc)          # scale about the origin first, then rotate + translate


def restore_outputs(rep_norm, step_path=None, ply_path=None):
    """Map the STEP and the deviation PLY written in the normalised frame back to the scan frame (N^-1)."""
    Ni = np.asarray(rep_norm["transform_normalised_to_scan"], float)
    done = []
    if step_path:
        from pathlib import Path
        if Path(step_path).is_file():
            from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
            from .occ_util import read_step, write_step
            sh = read_step(step_path)
            out = BRepBuilderAPI_Transform(sh, _trsf(Ni), True).Shape()
            write_step(out, step_path)
            done.append("step")
    if ply_path:
        from pathlib import Path
        if Path(ply_path).is_file():
            import trimesh
            m = trimesh.load(str(ply_path), process=False)
            m.apply_transform(Ni)
            m.export(str(ply_path))
            done.append("deviation_ply")
    return done
