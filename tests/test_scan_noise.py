"""Patch noise estimate recovers a known noise level on a flat plate and a cylinder."""
from __future__ import annotations

import numpy as np
import pytest
import trimesh

from meshsolid import scan_noise as SN


def _noisy(mesh, sigma, seed=0):
    rng = np.random.default_rng(seed)
    m = mesh.copy()
    m.vertices = m.vertices + rng.normal(0, sigma, len(m.vertices))[:, None] * m.vertex_normals
    return m


def test_flat_plate_noise_recovered():
    plate = trimesh.creation.box(extents=(80.0, 80.0, 6.0))
    for _ in range(6):
        plate = plate.subdivide()
    m = _noisy(plate, 0.013)
    top = np.asarray(m.vertices)[:, 2] > 2.9
    est = SN.estimate(m, region_mask=top)
    assert est["planar"]["median_rms_quadric"] == pytest.approx(0.013, rel=0.15)
    V = np.asarray(m.vertices)
    fl = SN.flatness(m, top & (np.abs(V[:, 0]) < 35) & (np.abs(V[:, 1]) < 35))   # edge vertices move sideways
    assert fl["rms"] == pytest.approx(0.013, rel=0.15)


def _grid_cylinder(r=11.0, h=60.0, nt=600, nz=240):
    th = np.linspace(0, 2 * np.pi, nt, endpoint=False)
    z = np.linspace(-h / 2, h / 2, nz)
    T, Z = np.meshgrid(th, z)
    V = np.stack([r * np.cos(T).ravel(), r * np.sin(T).ravel(), Z.ravel()], 1)
    i = np.arange(nz - 1)[:, None] * nt + np.arange(nt)[None, :]
    j = np.arange(nz - 1)[:, None] * nt + (np.arange(nt)[None, :] + 1) % nt
    a, b, c, d = i.ravel(), j.ravel(), i.ravel() + nt, j.ravel() + nt
    F = np.concatenate([np.stack([a, b, d], 1), np.stack([a, d, c], 1)])
    return trimesh.Trimesh(V, F, process=False)


def test_cylinder_curvature_is_not_counted_as_noise():
    m = _noisy(_grid_cylinder(), 0.010)
    V = np.asarray(m.vertices)
    side = np.abs(V[:, 2]) < 24.0
    est = SN.estimate(m, region_mask=side)
    assert est["all"]["median_rms_quadric"] == pytest.approx(0.010, rel=0.2)
    assert est["all"]["median_rms_plane"] > 0.1                 # a plane fit would call curvature noise


def test_step_inside_a_patch_is_rejected():
    plate = trimesh.creation.box(extents=(80.0, 80.0, 6.0))
    for _ in range(6):
        plate = plate.subdivide()
    m = _noisy(plate, 0.005)
    V = np.asarray(m.vertices).copy()
    top = V[:, 2] > 2.9
    V[top & (V[:, 0] > 0.3), 2] += 0.2                         # a 0.2 mm step along x = 0
    m.vertices = V
    est = SN.estimate(m, region_mask=top, spacing=5.0)
    near = [r for r in est["rows"] if abs(r["centre"][0]) < 2.0 and abs(r["centre"][1]) < 30]
    assert near and not any(r["ok"] for r in near)
