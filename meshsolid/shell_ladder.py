"""1.0.6 Shell-closing ladder: always try to produce a valid solid.

Order: looser sewing tolerance -> shape healing -> prismatic builder ->
freeform fallback.
"""

from __future__ import annotations

import numpy as np
import trimesh

from . import accuracy as _AC
from . import meshio as _MI
from . import freeform as _FF
from .segment import SegParams, segment
from .build_local import build_local
from .deviation import deviation


def _shell_closing_ladder(mesh, regions, labels, tol, pr):
    """1.0.6: shell-closing ladder - always try to produce a valid solid."""
    approx_faces: set = set()

    # --- Step 1: Looser sewing tolerance (2x original) ---
    try:
        loose_tol = tol * 2.0
        sp = SegParams(sharp_deg=pr.sharp_deg, tol=loose_tol, strict_torus=False)
        regions2, labels2, sinfo2 = segment(mesh, sp, rng_seed=pr.seed)
        b2 = build_local(mesh, labels2, regions2, loose_tol, volume_offset=0.0)
        if b2 is not None and b2.ok:
            g = _AC.resolve_gates(pr.accuracy, loose_tol)
            dv2, per_region_dev2, dv2_face, face_dev2, solid_mesh2 = deviation(
                mesh, b2.shape, labels2, exclude_faces=None, lin=0.01, ang=0.1
            )
            vd2 = (
                dv2["mesh_vertices_to_solid"]
                if "mesh_vertices_to_solid" in dv2
                else dv2["mesh_vertices_to_solid_excl"]
            )
            s2m2 = (
                dv2["solid_to_mesh_excl"]
                if "solid_to_mesh_excl" in dv2
                else dv2["solid_to_mesh"]
            )
            sag95 = float(np.percentile(np.abs(face_dev2), 95)) if face_dev2.size else 0.0
            lim_s2m = g["p95_limit"] + sag95
            accepted2 = (
                vd2["rms"] <= g["rms_limit"]
                and vd2["p95"] <= g["p95_limit"]
                and s2m2["p95"] <= lim_s2m
            )
            if accepted2:
                return True, b2.shape, set(), f"loose sewing tol={loose_tol:.4f} mm succeeded"
    except Exception as ex:  # noqa: BLE001
        pass

    # --- Step 2: Shape healing ---
    try:
        v = np.asarray(mesh.vertices, float)
        f = np.asarray(mesh.faces, int)
        # Merge close vertices
        key = np.round(v / 1e-6).astype(int)
        _, idx, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
        inv = inv.ravel()
        v = v[idx]
        f = f[inv]
        # Remove degenerate faces (zero area)
        area = 0.5 * np.linalg.norm(
            np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), axis=1
        )
        scale = float(np.linalg.norm(v.max(0) - v.min(0))) or 1.0
        degen = (
            (f[:, 0] == f[:, 1])
            | (f[:, 1] == f[:, 2])
            | (f[:, 0] == f[:, 2])
            | (area < (1e-12 * scale * scale))
        )
        f = f[~degen]
        # Remove duplicate faces (same winding)
        sf = np.sort(f, axis=1)
        _, ui, inv2, cnt = np.unique(sf, axis=0, return_index=True, return_inverse=True, return_counts=True)
        inv2 = inv2.ravel()
        r0 = np.argmin(f, axis=1)
        rot = np.stack([f[np.arange(len(f)), (r0 + k) % 3] for k in range(3)], 1)
        par = (rot[:, 1] < rot[:, 2]).astype(int)
        pos = np.bincount(inv2, par, minlength=len(ui))
        wall = (cnt >= 2) & (pos > 0) & (pos < cnt)
        keep = np.zeros(len(f), bool)
        keep[ui[~wall]] = True
        f = f[np.sort(np.where(keep)[0])]
        # Fill small holes (<=32 edges)
        loops, nm = _MI._boundary_loops(f)
        new_faces = list(f)
        new_verts = list(v)
        for loop in loops:
            if len(loop) > 32:
                continue
            c = v[loop].mean(0)
            new_verts.append(c)
            ci = len(new_verts) - 1
            for i in range(len(loop)):
                new_faces.append([loop[i], loop[(i + 1) % len(loop)], ci])
        rep_heal = {"holes_found": nm, "holes_filled": sum(1 for l in loops if len(l) <= 32)}
        if len(new_faces) > len(f):
            m = trimesh.Trimesh(np.array(new_verts), np.array(new_faces), process=False)
            m.remove_unreferenced_vertices()
            if m.is_watertight and m.volume > 0:
                mesh = m
                # Try prismatic builder on healed mesh
                sp = SegParams(sharp_deg=pr.sharp_deg, tol=tol, strict_torus=False)
                regions_h, labels_h, sinfo_h = segment(mesh, sp, rng_seed=pr.seed)
                bh = build_local(mesh, labels_h, regions_h, tol, volume_offset=0.0)
                if bh is not None and bh.ok:
                    return (
                        True,
                        bh.shape,
                        set(),
                        f"shape healing + prismatic builder (tol={tol:.4f} mm) succeeded",
                    )
    except Exception as ex:  # noqa: BLE001
        pass

    # --- Step 3: Prismatic builder at original tolerance ---
    try:
        sp = SegParams(sharp_deg=pr.sharp_deg, tol=tol, strict_torus=False)
        regions_p, labels_p, sinfo_p = segment(mesh, sp, rng_seed=pr.seed)
        bp = build_local(mesh, labels_p, regions_p, tol, volume_offset=0.0)
        if bp is not None and bp.ok:
            return True, bp.shape, set(), f"prismatic builder (tol={tol:.4f} mm) succeeded"
    except Exception as ex:  # noqa: BLE001
        pass

    # --- Step 4: Freeform fallback ---
    try:
        n_reg = max(1, len(regions) // 4)
        labels_f = -np.ones(len(mesh.faces), int)
        for i in range(n_reg):
            start = i * (mesh.faces.shape[0] // n_reg)
            end = min((i + 1) * (mesh.faces.shape[0] // n_reg), mesh.faces.shape[0])
            labels_f[start:end] = i
        flog = _FF.apply_freeform(mesh, labels_f, tol, pr.do_round, pr.round_tol, pr.max_tools)
        for entry in flog:
            if entry.get("accepted") and "faces" in entry:
                for fi in entry["faces"]:
                    approx_faces.add(fi)
        accepted_count = sum(1 for e in flog if e.get("accepted"))
        return (
            True,
            None,
            approx_faces,
            f"freeform fallback: {accepted_count}/{len(flog)} regions accepted (approximate)",
        )
    except Exception as ex:  # noqa: BLE001
        pass

    # All steps failed
    return (
        False,
        None,
        set(),
        "all ladder steps failed: loose sewing, shape healing, prismatic builder, and freeform fallback",
    )