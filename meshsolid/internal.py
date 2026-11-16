"""Internal scan faces of a turned part (capture caps, overlapping captures).

A scan assembled from several captures is often closed with a flat cap where each capture was cut;
such caps lie INSIDE the part (no real surface does).  Scan-only test, about the part axis: a face whose
normal is axial (|n.u| > 0.7) is internal when, at its angle, the scan has skin further out than the face
on BOTH axial sides, well beyond it (0.8 min(R(z - d), R(z + d)) > r + d at its angle, +-1 deg): material surrounds it, so it cannot be an
outer surface.  A real shoulder or end face has the envelope on one side only; gear roots and flanks
have radial normals and sit near the envelope.  Known limit: the bottom of a blind axial hole (a centre
drill) also passes; it is listed with the rest, so it shows.  Never derived from the model, so it cannot hide a model error."""
from __future__ import annotations

import numpy as np


def capture_caps(mesh, a0, u, d=1.0, dz=0.25, nth=720, margin_frac=0.1):
    a0 = np.asarray(a0, float); u = np.asarray(u, float) / np.linalg.norm(u)
    e1 = np.cross(u, [1.0, 0, 0]); e1 = e1 if np.linalg.norm(e1) > 0.1 else np.cross(u, [0, 1.0, 0])
    e1 /= np.linalg.norm(e1); e2 = np.cross(u, e1)
    import trimesh
    # envelope from area-uniform samples (a decimated scan has too few vertices per 0.5 deg bin)
    Ps, fi = trimesh.sample.sample_surface(mesh, max(200000, 4 * len(mesh.faces)), seed=0)
    V = np.asarray(Ps) - a0
    z = V @ u; x = V @ e1; y = V @ e2
    r = np.hypot(x, y); th = np.arctan2(y, x)
    z0 = z.min()
    iz = ((z - z0) / dz).astype(int); it = ((th + np.pi) / (2 * np.pi) * nth).astype(int) % nth
    nz = iz.max() + 1
    R = np.zeros((nz, nth)); np.maximum.at(R, (iz, it), r)
    C = np.asarray(mesh.triangles_center) - a0
    cz = C @ u; cx = C @ e1; cy = C @ e2; cr = np.hypot(cx, cy); cth = np.arctan2(cy, cx)
    nax = np.abs(np.asarray(mesh.face_normals) @ u)
    k = int(round(d / dz))
    jz = ((cz - z0) / dz).astype(int); jt = ((cth + np.pi) / (2 * np.pi) * nth).astype(int) % nth
    lo = np.clip(jz - k, 0, nz - 1); hi = np.clip(jz + k, 0, nz - 1)
    # envelope over the neighbouring angle bins too (a cap is wide; noise at one bin should not decide)
    # envelope = outermost skin within +-1 deg (the scan is sparse at 0.5 deg bins); both axial sides
    Rlo = np.maximum.reduce([R[lo, (jt + s) % nth] for s in (-2, -1, 0, 1, 2)])
    Rhi = np.maximum.reduce([R[hi, (jt + s) % nth] for s in (-2, -1, 0, 1, 2)])
    inside = (nax > 0.7) & (np.minimum(Rlo, Rhi) - cr > np.maximum(d, margin_frac * np.minimum(Rlo, Rhi))) & (jz - k >= 0) & (jz + k <= nz - 1)
    info = {"faces": int(inside.sum()), "area_mm2": round(float(mesh.area_faces[inside].sum()), 2)}
    if inside.any():
        info["z_range_axis_frame"] = [round(float(cz[inside].min()), 3), round(float(cz[inside].max()), 3)]
        info["r_max"] = round(float(cr[inside].max()), 3)
        zz = np.sort(cz[inside]); cuts = np.where(np.diff(zz) > 2.0)[0]
        lo_ = 0; cl = []
        for hi_ in list(cuts) + [len(zz) - 1]:
            sel = inside & (cz >= zz[lo_] - 1e-9) & (cz <= zz[hi_] + 1e-9)
            cl.append({"z_axis_frame": [round(float(zz[lo_]), 2), round(float(zz[hi_]), 2)], "faces": int(sel.sum()),
                       "area_mm2": round(float(mesh.area_faces[sel].sum()), 2), "r_max": round(float(cr[sel].max()), 2)})
            lo_ = hi_ + 1
        info["clusters"] = cl
    return inside, info
