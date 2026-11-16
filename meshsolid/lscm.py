"""Least-squares conformal map (Levy et al. 2002) of a disk-like triangle patch to the plane.

Used to parametrise free-form skins that are not height fields (a wing skin wrapping round its
leading edge, a fuselage shell): the B-spline is then fitted over the conformal (u, v) instead of a
projection plane, so the whole smooth region between sharp feature edges is one surface."""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr


def topology(F):
    """(n_vertices, n_edges, n_faces, n_boundary_loops, genus) of a triangle patch F (local indices)."""
    E = np.sort(np.r_[F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], axis=1)
    Eu, cnt = np.unique(E, axis=0, return_counts=True)
    nv = int(F.max()) + 1
    be = Eu[cnt == 1]
    # boundary loops = connected components of the boundary edge graph
    loops = 0
    if len(be):
        from scipy.sparse.csgraph import connected_components
        G = csr_matrix((np.ones(len(be)), (be[:, 0], be[:, 1])), shape=(nv, nv))
        used = np.unique(be.ravel())
        nc, lab = connected_components(G, directed=False)
        loops = len(np.unique(lab[used]))
    chi = nv - len(Eu) + len(F)
    genus = (2 - chi - loops) // 2
    return nv, len(Eu), len(F), loops, int(genus)


def lscm(V, F):
    """uv (n, 2) for vertices V (n, 3) of triangles F (local indices), scaled to the 3D metric.
    Returns (uv, info) with info["scale_spread"] = p95 / p5 of the per-face linear scale."""
    V = np.asarray(V, float)
    F = np.asarray(F, int)
    n = len(V)
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    e1 = b - a
    nrm = np.cross(e1, c - a)
    A2 = np.linalg.norm(nrm, axis=1)
    ok = A2 > 1e-14 * max(1.0, float(A2.max()))
    F, a, b, c, e1, nrm, A2 = F[ok], a[ok], b[ok], c[ok], e1[ok], nrm[ok], A2[ok]
    x = e1 / np.linalg.norm(e1, axis=1, keepdims=True)
    y = np.cross(nrm / A2[:, None], x)
    # local 2D coordinates of the three corners
    q = np.zeros((len(F), 3, 2))
    q[:, 1, 0] = np.einsum("ij,ij->i", b - a, x)
    q[:, 2, 0] = np.einsum("ij,ij->i", c - a, x)
    q[:, 2, 1] = np.einsum("ij,ij->i", c - a, y)
    s = 1.0 / np.sqrt(A2)
    # W_j = q_{j+2} - q_{j+1} (complex), row: sum_j W_j U_j = 0 (real and imaginary parts)
    rows, cols, vals = [], [], []
    m = len(F)
    for j in range(3):
        W = q[:, (j + 2) % 3] - q[:, (j + 1) % 3]
        wr, wi = W[:, 0] * s, W[:, 1] * s
        vj = F[:, j]
        r_re, r_im = np.arange(m), m + np.arange(m)
        # real row: wr*u - wi*v ; imaginary row: wi*u + wr*v
        rows += [r_re, r_re, r_im, r_im]
        cols += [vj, n + vj, vj, n + vj]
        vals += [wr, -wi, wi, wr]
    M = csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(2 * m, 2 * n))
    # pin the two vertices farthest apart (first principal direction extremes)
    C = V - V.mean(0)
    w = np.linalg.eigh(C.T @ C)[1][:, -1]
    p0, p1 = int(np.argmin(C @ w)), int(np.argmax(C @ w))
    L = float(np.linalg.norm(V[p1] - V[p0]))
    fixed = {p0: (0.0, 0.0), p1: (L, 0.0)}
    fcols = [p0, p1, n + p0, n + p1]
    fval = np.array([fixed[p0][0], fixed[p1][0], fixed[p0][1], fixed[p1][1]])
    free = np.setdiff1d(np.arange(2 * n), fcols)
    Mf = M[:, free]
    rhs = -(M[:, fcols] @ fval)
    sol = lsqr(Mf, rhs, atol=1e-12, btol=1e-12, iter_lim=20 * n)[0]
    X = np.zeros(2 * n)
    X[free] = sol
    X[fcols] = fval
    uv = np.c_[X[:n], X[n:]]
    # per-face linear scale (3D area / uv area); global scale to the 3D metric
    ua, ub, uc = uv[F[:, 0]], uv[F[:, 1]], uv[F[:, 2]]
    A2uv = (ub[:, 0] - ua[:, 0]) * (uc[:, 1] - ua[:, 1]) - (ub[:, 1] - ua[:, 1]) * (uc[:, 0] - ua[:, 0])
    flipped = float(np.mean(A2uv < 0)) if np.mean(A2uv > 0) >= 0.5 else float(np.mean(A2uv > 0))
    sc = np.sqrt(A2 / np.maximum(np.abs(A2uv), 1e-300))
    g = float(np.sqrt(A2.sum() / max(np.abs(A2uv).sum(), 1e-300)))
    uv *= g
    sc = sc / g
    info = {"faces": int(m), "flipped_frac": round(flipped, 5),
            "scale_spread": float(np.percentile(sc, 95) / max(np.percentile(sc, 5), 1e-300))}
    return uv, info
