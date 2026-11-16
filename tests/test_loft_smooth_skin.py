"""Section-loft on faceted CAD swept skins: corner-free loops must tolerate polyline tessellation turns."""
from __future__ import annotations

import math

import numpy as np
import trimesh
from scipy.spatial import Delaunay

from meshsolid import loft


def _airfoil_xy(n=40, chord=100.0, t=0.12):
    """Closed sharp-TE airfoil polyline in 2D (upper then lower), CCW."""
    x = np.linspace(0, 1, n // 2)
    yt = 5 * t * chord * (0.2969 * np.sqrt(np.maximum(x, 0)) - 0.1260 * x - 0.3516 * x ** 2
                          + 0.2843 * x ** 3 - 0.1015 * x ** 4)
    xu, yu = chord * x, yt
    xl, yl = chord * x[::-1], -yt[::-1]
    Q = np.c_[np.r_[xu, xl[1:-1]], np.r_[yu, yl[1:-1]]]
    if loft._area2(Q) < 0:
        Q = Q[::-1]
    return Q


def _extruded_airfoil_mesh(height=40.0, n=36, chord=80.0, t=0.15):
    Q = _airfoil_xy(n, chord=chord, t=t)
    n = len(Q)
    V0 = np.c_[Q, np.zeros(n)]
    V1 = np.c_[Q, np.full(n, height)]
    F = []
    for i in range(n):
        j = (i + 1) % n
        F += [[i, j, n + j], [i, n + j, n + i]]
    dln = Delaunay(Q)
    tris = np.array([t for t in dln.simplices if loft._inside(Q[t].mean(0), Q)], int)
    # bottom (z=0): outward -Z => CW from +Z; top: outward +Z => CCW
    F += tris[:, ::-1].tolist()
    F += (tris + n).tolist()
    mesh = trimesh.Trimesh(np.vstack([V0, V1]), F, process=False)
    mesh.fix_normals()
    return mesh


def test_corners_like_corner_free_allows_tessellation_turns():
    Q = _airfoil_xy(48)
    t = loft._turn(Q)
    th = math.radians(35)
    assert np.any(np.abs(t) > th), "fixture must have polyline turns above the corner threshold"
    out = loft._corners_like(Q, prev_pts=np.zeros((0, 2)), prev_sign=np.zeros(0), th=th)
    assert len(out) == 0


def test_loft_builds_faceted_extruded_airfoil():
    """Straight extrusion of a faceted airfoil via smooth section-loft (corner_deg<=0)."""
    height = 40.0
    mesh = _extruded_airfoil_mesh(height=height)
    Q = _airfoil_xy(36, chord=80.0, t=0.15)
    expect = abs(loft._area2(Q)) * height
    assert mesh.is_watertight and abs(mesh.volume - expect) / expect < 0.02
    tol = 0.05
    res = loft.build_loft(mesh, tol=tol, corner_deg=0.0, time_budget=90.0, max_sections=8)
    assert res.ok, res.reason
    assert res.n_solids == 1 and res.brepcheck_valid
    assert abs(res.volume - mesh.volume) / abs(mesh.volume) < 0.02


def _tapered_airfoil_mesh(height=50.0, n=36, chord0=80.0, chord1=40.0, t=0.15, layers=8):
    """Linearly tapered extruded airfoil (non-prismatic smooth skin)."""
    Q0 = _airfoil_xy(n, chord=chord0, t=t)
    # same parametric shape scaled to chord1 about centroid
    c0 = Q0.mean(0)
    scale = chord1 / chord0
    Q1 = c0 + (Q0 - c0) * scale
    zs = np.linspace(0, height, layers + 1)
    verts = []
    for z, s in zip(zs, np.linspace(1.0, scale, layers + 1)):
        Q = c0 + (Q0 - c0) * s
        verts.append(np.c_[Q, np.full(len(Q), z)])
    V = np.vstack(verts)
    nq = len(Q0)
    F = []
    for li in range(layers):
        for i in range(nq):
            j = (i + 1) % nq
            a, b = li * nq + i, li * nq + j
            c, d = (li + 1) * nq + j, (li + 1) * nq + i
            F += [[a, b, c], [a, c, d]]
    dln = Delaunay(Q0)
    tris = np.array([t for t in dln.simplices if loft._inside(Q0[t].mean(0), Q0)], int)
    F += tris[:, ::-1].tolist()  # bottom
    F += (tris + layers * nq).tolist()  # tip (scaled)
    # tip face uses Q1 topology at last layer — Delaunay on Q0 indices still valid under uniform scale
    mesh = trimesh.Trimesh(V, F, process=False)
    mesh.fix_normals()
    return mesh


def test_loft_builds_faceted_tapered_airfoil():
    """Tapered faceted airfoil must become a true solid (equal section sampling), not an open shell."""
    height = 50.0
    # Dense span tessellation so mesh vertices lie near the continuous loft (not stair-steps).
    mesh = _tapered_airfoil_mesh(height=height, chord0=80.0, chord1=40.0, n=48, layers=40)
    assert mesh.is_watertight and mesh.volume > 0
    tol = 0.08
    res = loft.build_loft(mesh, tol=tol, corner_deg=0.0, time_budget=90.0, max_sections=12)
    assert res.ok, res.reason
    assert res.n_solids == 1 and res.brepcheck_valid
    assert abs(res.volume - mesh.volume) / abs(mesh.volume) < 0.02


def _pointed_spindle_mesh(n_theta=36, n_span=24, r_mid=20.0, length=80.0):
    """Double-cone spindle (point tips, no planar caps) along Z."""
    zs = np.linspace(-length / 2, length / 2, n_span)
    # radius tapers linearly to 0 at both ends
    rs = r_mid * (1.0 - np.abs(zs) / (length / 2))
    rs = np.maximum(rs, 1e-3)  # tiny tip radius so mesh stays manifold
    th = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    verts = []
    for z, r in zip(zs, rs):
        verts.append(np.c_[r * np.cos(th), r * np.sin(th), np.full(n_theta, z)])
    # true tip points
    verts = [np.array([[0.0, 0.0, -length / 2]])] + verts + [np.array([[0.0, 0.0, length / 2]])]
    V = np.vstack(verts)
    F = []
    # tip fans
    for i in range(n_theta):
        j = (i + 1) % n_theta
        F.append([0, 1 + j, 1 + i])  # bottom tip
        top0 = len(V) - 1
        base = top0 - n_theta
        F.append([top0, base + i, base + j])
    # side quads between rings (rings start at index 1)
    for li in range(n_span - 1):
        a0 = 1 + li * n_theta
        b0 = 1 + (li + 1) * n_theta
        for i in range(n_theta):
            j = (i + 1) % n_theta
            F += [[a0 + i, a0 + j, b0 + j], [a0 + i, b0 + j, b0 + i]]
    mesh = trimesh.Trimesh(V, F, process=True)
    mesh.fix_normals()
    return mesh


def test_loft_builds_pointed_spindle():
    """Capless pointed spindle must close with tip apexes — planar end caps leave tip verts far."""
    mesh = _pointed_spindle_mesh(n_theta=48, n_span=40)
    assert mesh.is_watertight and mesh.volume > 0
    tol = 0.08
    res = loft.build_loft(mesh, tol=tol, corner_deg=0.0, time_budget=90.0, max_sections=24)
    assert res.ok, res.reason
    assert res.n_solids == 1 and res.brepcheck_valid
    assert abs(res.volume - mesh.volume) / abs(mesh.volume) < 0.02


def test_te_le_corners_splits_sharp_trailing_edge():
    """Sharp TE airfoil gets TE+LE corner indices that stay separated."""
    Q = _airfoil_xy(64, chord=80.0, t=0.12)
    # Airfoil TE is the sharp cusp (max |turn|); LE is farthest from it.
    c, te_pt = loft._te_le_corners(Q)
    assert len(c) == 2
    n = len(Q)
    sep = min(int((c[1] - c[0]) % n), int((c[0] - c[1]) % n))
    assert sep >= 3
    # TE is a sharp turn
    assert abs(loft._turn(Q)[int(c[0])]) > abs(loft._turn(Q)).mean()

def test_faceted_self_check_s2m_uses_chord_sag_vertices_stay_tight():
    """First principles for faceted CAD: verts on true surface; s2m widened by sag_p90 only.

    Vertex gates remain 1.5/3*tol (must still reject a 0.89-rms tip-cap miss and a 43% underfill
    via the volume gate). solid->mesh limit becomes 3*tol+sag_p90, matching pipeline.accept —
    not a per-part loosen to pass any drone wingtip.
    """
    from meshsolid.facets import facet_stats
    mesh = _extruded_airfoil_mesh()
    st = facet_stats(mesh)
    assert st["coplanar_len_frac"] > 0.02 and st["turning_smooth_len_frac"] > 0.15
    sag = float(st["sag_p90"])
    tol = 0.05
    # Contract encoded by build_loft self_check
    lim_s2m = 3.0 * tol + sag
    assert lim_s2m < 4.0 * tol + sag + 1e-12
    # Vertex gates unchanged: a 0.89 rms loft (pre-apex tip miss) still fails 1.5*tol
    assert 0.89 > 1.5 * tol
    # 43% underfill fails the volume gate (2%), not the deviation gate
    assert abs(0.43 - 1.0) > 0.02
    res = loft.build_loft(mesh, tol=tol, corner_deg=0.0, time_budget=90.0, max_sections=8)
    assert res.ok, res.reason
    assert res.self_check.get("faceted") is True
    assert abs(res.self_check["facet_sag_p90"] - sag) < 1e-4
    assert abs(res.self_check["solid_to_mesh_p95_limit"] - lim_s2m) < 1e-4
    # Vertices still gated tightly relative to tol
    assert res.self_check["rms"] <= 1.5 * tol
    assert res.self_check["mesh_vertices_to_solid_p95"] <= 3 * tol
