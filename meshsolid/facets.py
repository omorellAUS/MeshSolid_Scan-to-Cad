"""Faceted-CAD STL detection (drone/VTOL prints exported from CAD at coarse tessellation).

A CAD export approximates curved surfaces by flat facets: vertices lie exactly on the true surface, so
the scan-noise estimate is ~0 and the automatic tolerance collapses to a few microns, while every facet
deviates from the surface it stands for by its chord sag.  Region growing then sees each facet strip of
a cylinder or an airfoil skin as its own plane (hundreds of planes, slivers, timeouts).

For every interior edge between two faces whose normals differ by a small angle theta (smooth surface,
below the sharp-edge angle) the sag of a circular arc through the two facets is ~ w theta / 8, w the
facet width across the edge (2 area / edge length).  A mesh is "faceted CAD" when it is exact (no
noise) and a sizable share of its smooth-edge length turns by a visible angle; its fit tolerance is then
set from the sag distribution (p90), so a faceted cylinder is one cylinder and a skin one B-spline.
"""
from __future__ import annotations

import numpy as np


def facet_stats(mesh, sharp_deg=30.0, min_turn_deg=0.5):
    adj = mesh.face_adjacency
    if len(adj) == 0:
        return {"faceted": False, "reason": "no face adjacency"}
    ang = np.degrees(mesh.face_adjacency_angles)
    e = mesh.face_adjacency_edges
    V = mesh.vertices
    L = np.linalg.norm(V[e[:, 0]] - V[e[:, 1]], axis=1)
    A = mesh.area_faces
    w = (2 * A[adj[:, 0]] / np.maximum(L, 1e-12) + 2 * A[adj[:, 1]] / np.maximum(L, 1e-12)) / 2
    smooth = ang < sharp_deg
    turn = smooth & (ang >= min_turn_deg)
    sag = w * np.radians(ang) / 8
    Ls = L[smooth].sum()
    flat = ang < 0.01                          # exactly coplanar neighbours: CAD triangulated a plane
    out = {"edges": int(len(adj)), "coplanar_len_frac": round(float(L[flat].sum() / max(L.sum(), 1e-12)), 4),
           "smooth_edge_len_frac": round(float(Ls / max(L.sum(), 1e-12)), 4),
           "turning_smooth_len_frac": round(float(L[turn].sum() / max(Ls, 1e-12)), 4),
           "dihedral_turning_median_deg": round(float(np.median(ang[turn])), 3) if turn.any() else 0.0,
           "facet_width_median": round(float(np.median(w)), 4)}
    if turn.any():
        wt = L[turn]
        o = np.argsort(sag[turn]); cs = np.cumsum(wt[o]) / wt.sum()
        s = sag[turn][o]
        out["sag_p50"] = round(float(s[np.searchsorted(cs, 0.5)]), 5)
        out["sag_p90"] = round(float(s[min(np.searchsorted(cs, 0.9), len(s) - 1)]), 5)
    else:
        out["sag_p50"] = out["sag_p90"] = 0.0
    return out


def detect(mesh, noise_sigma, exact, sharp_deg=30.0):
    """Returns (is_faceted, stats).  CAD tessellation: exact by the noise estimate, or a sizable share of
    exactly coplanar neighbour faces (a scan never has those); faceted = CAD with turning smooth edges."""
    st = facet_stats(mesh, sharp_deg)
    st["cad_tessellation"] = bool(exact) or st.get("coplanar_len_frac", 0) > 0.02
    st["faceted"] = bool(st["cad_tessellation"] and st.get("turning_smooth_len_frac", 0) > 0.15)
    return st["faceted"], st


def tolerance(st, diag):
    """Fit tolerance for a CAD tessellation (the scan-noise estimate means nothing there: thin walls put
    both skins in one neighbourhood): covers the chord sag of 90 % of the turning facets (x 1.5), floor as
    for exact meshes, capped at 0.5 % of the part size."""
    return float(min(max(1.5 * st.get("sag_p90", 0.0), 5e-5 * diag, 0.002), 0.005 * diag))
