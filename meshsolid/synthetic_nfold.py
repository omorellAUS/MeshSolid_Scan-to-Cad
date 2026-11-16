"""Synthetic N-fold part (hub disc + N swept blades) and a 'scan' of it with noise, per-blade scatter and a
known rigid motion, for the spec-v2 alignment self-test and the sector / per-blade tests."""
from __future__ import annotations

import numpy as np
import trimesh

from .align_v2 import rigid, rot_about


def blade_mesh(r0=12.0, r1=40.0, h=14.0, thick=1.6, sweep_deg=25.0, subdiv=3):
    b = trimesh.creation.box(extents=(r1 - r0, thick, h))
    b.apply_translation([(r0 + r1) / 2, 0, h / 2])
    for _ in range(subdiv):
        b = b.subdivide()
    V = b.vertices.copy()
    r = np.hypot(V[:, 0], V[:, 1])
    th = np.arctan2(V[:, 1], V[:, 0]) + np.radians(sweep_deg) * (r - r0) / (r1 - r0)
    b.vertices = np.stack([r * np.cos(th), r * np.sin(th), V[:, 2]], 1)
    return b


def part(n_fold=15, sweep_deg=25.0, subdiv=3, hub_sections=180):
    """(mesh, blade_of_face): hub disc (r 42, 6 mm thick, z -6..0) + N blades on top. Blade faces carry their index."""
    hub = trimesh.creation.cylinder(radius=42.0, height=6.0, sections=hub_sections)
    hub.apply_translation([0, 0, -3.0])
    hub = hub.subdivide().subdivide()
    meshes, owner = [hub], [np.full(len(hub.faces), -1)]
    b0 = blade_mesh(sweep_deg=sweep_deg, subdiv=subdiv)
    for k in range(n_fold):
        b = b0.copy()
        T = np.eye(4)
        T[:3, :3] = rot_about([0, 0, 1], 2 * np.pi * k / n_fold)
        b.apply_transform(T)
        meshes.append(b)
        owner.append(np.full(len(b.faces), k))
    m = trimesh.util.concatenate(meshes)
    return m, np.concatenate(owner)


def scan_of(solid, owner, noise=0.013, blade_rot_deg=0.0, blade_thick_mm=0.0, T=None, seed=1, densify=2):
    """Densified copy of the solid as a scan (each subdivision x4 triangles): per-blade rotation about z (deg,
    list) and thickness offset along the face normal (mm, list), Gaussian noise along the normal, then the rigid
    motion T. Returns (scan mesh, vertex blade index, -1 = hub)."""
    rng = np.random.default_rng(seed)
    V, F = np.asarray(solid.vertices, float), np.asarray(solid.faces)
    own = np.asarray(owner)
    for _ in range(int(densify)):
        V, F, idx = trimesh.remesh.subdivide(V, F, return_index=True)
        new = np.empty(len(F), own.dtype)
        if isinstance(idx, dict):
            for k, v in idx.items():
                new[np.asarray(v)] = own[k]
        else:
            new[:] = own[np.asarray(idx)]
        own = new
    m0 = trimesh.Trimesh(V, F, process=False)
    V = V.copy()
    N = np.asarray(m0.vertex_normals, float)
    vb = np.full(len(V), -1)
    owner = own
    for k in range(int(owner.max()) + 1):
        vb[np.unique(F[owner == k])] = k
    nb = int(owner.max()) + 1
    br = np.broadcast_to(np.asarray(blade_rot_deg, float), (nb,))
    bt = np.broadcast_to(np.asarray(blade_thick_mm, float), (nb,))
    for k in range(nb):
        m = vb == k
        V[m] = V[m] + bt[k] * N[m]
        R = rot_about([0, 0, 1], np.radians(br[k]))
        V[m] = V[m] @ R.T
    V = V + rng.normal(0.0, noise, len(V))[:, None] * N
    if T is not None:
        V = V @ T[:3, :3].T + T[:3, 3]
    return trimesh.Trimesh(V, F.copy(), process=False), vb


def known_motion(rot_deg=0.1, axis=(0.05, -0.03, 1.0), t_mm=(0.03, -0.02, 0.015), centre=(0, 0, 0)):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    return rigid(np.radians(rot_deg) * a, np.asarray(t_mm, float), np.asarray(centre, float))


def balanced_scatter(n_fold, std, seed=0):
    """Per-blade values with the 0th and 1st angular harmonics removed. A mean (0th) per-blade rotation IS a
    rigid rotation, and a once-per-revolution (1st harmonic) pattern IS a rigid translation, so only the rest is
    scatter that a rigid pose cannot (and must not) absorb."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, std, n_fold)
    th = 2 * np.pi * np.arange(n_fold) / n_fold
    B = np.stack([np.ones(n_fold), np.cos(th), np.sin(th)], 1)
    x = x - B @ np.linalg.lstsq(B, x, rcond=None)[0]
    return x * (std / max(float(np.std(x)), 1e-12))
