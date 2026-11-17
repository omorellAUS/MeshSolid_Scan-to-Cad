"""M4 freeform surfaces: least-squares tensor-product cubic B-spline fits for regions no primitive fits.

Two parametrisations:
  * "patch": height field over the best-fit plane (open surface, clamped knots in u and v);
  * "tube":  sweep-like surfaces (bent pipes): u = arc length along a fitted centreline, v = angle round
             it in a rotation-minimising frame; periodic in v (closed surface, no seam to trim against).
The control net is solved directly from the scan points with a small second-difference smoothing term,
which also extends the surface smoothly past the data (a margin so neighbours can trim it). The result
is converted to an OCC Geom_BSplineSurface (cubic, clamped u, clamped or periodic v)."""
from __future__ import annotations

import numpy as np
from scipy.interpolate import BSpline
from scipy.sparse import csr_matrix, diags, kron, identity, vstack as svstack
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree

from . import fit as F

DEG = 3


# ---------------------------------------------------------------- basis
def _clamped_knots(a, b, n):
    inner = np.linspace(a, b, n - DEG + 1)
    return np.r_[[a] * DEG, inner, [b] * DEG]


def _basis_clamped(x, a, b, n):
    t = _clamped_knots(a, b, n)
    x = np.clip(x, a, b)
    return BSpline.design_matrix(x, t, DEG).tocsr()


def _basis_periodic(x, n):
    """Uniform periodic cubic basis on [0, 2pi) with n functions; function j is centred at knot j+2
    (matches OCC's periodic convention, checked numerically in _to_occ)."""
    h = 2 * np.pi / n
    t = h * np.arange(-DEG, n + DEG + 1)
    x = np.mod(x, 2 * np.pi)
    B = BSpline.design_matrix(x, t, DEG).tocsr()       # n + DEG columns
    B = B.tocoo()
    cols = np.mod(B.col, n)
    return csr_matrix((B.data, (B.row, cols)), shape=(len(x), n))


def _second_diff(n, periodic=False):
    if n < 3:
        return csr_matrix((0, n))
    if periodic:
        D = diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n, n + 2)).tolil()
        D = D.tocsr()
        D = D[:, :n] + csr_matrix((D[:, n:].toarray() @ np.eye(2, n)))
        return csr_matrix(D)
    return diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n - 2, n)).tocsr()


def _tensor(Bu, Bv):
    """Row-wise Kronecker product: rows = points, cols = nu*nv (u major)."""
    Bu, Bv = Bu.tocsr(), Bv.tocsr()
    rows, cols, vals = [], [], []
    nv = Bv.shape[1]
    for i in range(Bu.shape[0]):
        a0, a1 = Bu.indptr[i], Bu.indptr[i + 1]
        b0, b1 = Bv.indptr[i], Bv.indptr[i + 1]
        cu, vu = Bu.indices[a0:a1], Bu.data[a0:a1]
        cv, vv = Bv.indices[b0:b1], Bv.data[b0:b1]
        c = (cu[:, None] * nv + cv[None, :]).ravel()
        rows.append(np.full(len(c), i)); cols.append(c); vals.append((vu[:, None] * vv[None, :]).ravel())
    return csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(Bu.shape[0], Bu.shape[1] * nv))


def _solve(A, P, nu, nv, periodic_v, lam):
    Du = kron(_second_diff(nu), identity(nv))
    Dv = kron(identity(nu), _second_diff(nv, periodic_v))
    R = svstack([Du, Dv]).tocsr()
    M = (A.T @ A + lam * (R.T @ R) + 1e-10 * identity(nu * nv)).tocsc()
    Q = np.column_stack([spsolve(M, A.T @ P[:, k]) for k in range(3)])
    return Q.reshape(nu, nv, 3)


# ---------------------------------------------------------------- evaluation
class Surf:
    def __init__(self, Q, ua, ub, periodic_v, va=0.0, vb=2 * np.pi):
        self.Q, self.ua, self.ub, self.pv, self.va, self.vb = Q, ua, ub, periodic_v, va, vb
        self.nu, self.nv = Q.shape[:2]

    def basis(self, u, v):
        Bu = _basis_clamped(u, self.ua, self.ub, self.nu)
        Bv = _basis_periodic(v, self.nv) if self.pv else _basis_clamped(v, self.va, self.vb, self.nv)
        return _tensor(Bu, Bv)

    def eval(self, u, v):
        return self.basis(u, v) @ self.Q.reshape(-1, 3)

    def grid(self, nu=240, nv=120):
        u = np.linspace(self.ua, self.ub, nu)
        v = np.linspace(self.va, self.vb, nv, endpoint=not self.pv)
        U, Vv = np.meshgrid(u, v, indexing="ij")
        S = self.eval(U.ravel(), Vv.ravel()).reshape(nu, nv, 3)
        du = np.gradient(S, axis=0)
        dv = (np.roll(S, -1, 1) - np.roll(S, 1, 1)) if self.pv else np.gradient(S, axis=1)
        n = np.cross(du, dv)
        n /= np.maximum(np.linalg.norm(n, axis=2, keepdims=True), 1e-12)
        return S.reshape(-1, 3), n.reshape(-1, 3)


class Sampler:
    """Signed distance to a fitted surface from a dense sample (nearest sample + tangent plane)."""

    def __init__(self, surf, nu=300, nv=160):
        self.S, self.N = surf.grid(nu, nv)
        self.tree = cKDTree(self.S)

    def distance(self, P):
        _, i = self.tree.query(np.atleast_2d(P))
        return np.einsum("ij,ij->i", np.atleast_2d(P) - self.S[i], self.N[i])

    def normal(self, P):
        _, i = self.tree.query(np.atleast_2d(P))
        return self.N[i]


# ---------------------------------------------------------------- parametrisations
def _rmf(C, T):
    """Rotation-minimising frame (double reflection) along polyline C with tangents T."""
    t0 = T[0]
    a = np.array([1.0, 0, 0]) if abs(t0[0]) < 0.9 else np.array([0, 1.0, 0])
    r = a - (a @ t0) * t0
    r /= np.linalg.norm(r)
    R = [r]
    for i in range(len(C) - 1):
        v1 = C[i + 1] - C[i]
        c1 = v1 @ v1
        if c1 < 1e-18:
            R.append(R[-1]); continue
        rl = R[-1] - (2 / c1) * (v1 @ R[-1]) * v1
        tl = T[i] - (2 / c1) * (v1 @ T[i]) * v1
        v2 = T[i + 1] - tl
        c2 = v2 @ v2
        rn = rl - (2 / c2) * (v2 @ rl) * v2 if c2 > 1e-18 else rl
        rn -= (rn @ T[i + 1]) * T[i + 1]
        R.append(rn / np.linalg.norm(rn))
    R = np.array(R)
    return R, np.cross(T, R)


def tube_params(P, N, n_sub=3000, seed=0):
    """Centreline + (s, phi) for a tube-like point set. Returns dict or None."""
    from scipy.sparse.csgraph import shortest_path
    from scipy.interpolate import make_smoothing_spline
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(P), min(n_sub, len(P)), replace=False)
    Ps, Ns = P[idx], N[idx]
    diag = float(np.linalg.norm(np.ptp(P, axis=0)))
    best = None
    for sign in (1.0, -1.0):
        for rho in np.geomspace(0.01 * diag, 0.6 * diag, 50):
            C = Ps - sign * rho * Ns
            tr = cKDTree(C)
            _, nn = tr.query(C, 16)
            G = C[nn] - C[nn].mean(1, keepdims=True)
            ev = np.linalg.eigvalsh(np.einsum("nki,nkj->nij", G, G) / 16)
            score = float(np.mean(np.sqrt(np.maximum(ev[:, 0] + ev[:, 1], 0)))) / rho
            if best is None or score < best[0]:
                best = (score, sign, rho)
    score, sign, rho = best
    if score > 0.15:            # centres do not collapse onto a curve: not a tube
        return None
    C = Ps - sign * rho * Ns
    tr = cKDTree(C)
    d, nn = tr.query(C, 10)
    rows = np.repeat(np.arange(len(C)), 10)
    G = csr_matrix((d.ravel() + 1e-9, (rows, nn.ravel())), shape=(len(C), len(C)))
    # centreline ordering on the largest connected piece of the centre cloud (scan gaps can split
    # the kNN graph; the other points get their s by projection below)
    from scipy.sparse.csgraph import connected_components
    ncomp, comp = connected_components(G, directed=False)
    keep = comp == np.bincount(comp).argmax()
    if keep.sum() < 0.6 * len(C):
        return None
    C, G = C[keep], G[keep][:, keep]
    D0 = shortest_path(G, directed=False, indices=[0])[0]
    e = int(np.argmax(D0))
    s = shortest_path(G, directed=False, indices=[e])[0]
    # bin along s, smoothing spline centreline
    nb = int(np.clip(s.max() / (0.5 * rho), 8, 80))
    edges = np.linspace(0, s.max(), nb + 1)
    k = np.clip(np.searchsorted(edges, s) - 1, 0, nb - 1)
    sb = np.array([s[k == i].mean() for i in range(nb) if np.any(k == i)])
    cb = np.array([C[k == i].mean(0) for i in range(nb) if np.any(k == i)])
    spl = [make_smoothing_spline(sb, cb[:, j], lam=None) for j in range(3)]
    sd = np.linspace(sb[0], sb[-1], 600)
    Cd = np.column_stack([f(sd) for f in spl])
    # re-parametrise by true arc length
    L = np.r_[0, np.cumsum(np.linalg.norm(np.diff(Cd, axis=0), axis=1))]
    T = np.gradient(Cd, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    E1, E2 = _rmf(Cd, T)
    # project all points: nearest dense sample, then along the local tangent
    _, j = cKDTree(Cd).query(P)
    dP = P - Cd[j]
    sP = L[j] + np.einsum("ij,ij->i", dP, T[j])
    dP = dP - np.einsum("ij,ij->i", dP, T[j])[:, None] * T[j]
    phi = np.mod(np.arctan2(np.einsum("ij,ij->i", dP, E2[j]), np.einsum("ij,ij->i", dP, E1[j])), 2 * np.pi)
    return {"s": sP, "phi": phi, "rho": rho, "sign": sign, "length": float(L[-1]), "score": score}


def _uv_patch(P, N):
    n = N.mean(0)
    if np.linalg.norm(n) < 0.6:
        return None
    n /= np.linalg.norm(n)
    if np.min(N @ n) < 0.15:     # folds over: not a height field over this plane
        return None
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = a - (a @ n) * n; e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    c = P.mean(0)
    return (P - c) @ e1, (P - c) @ e2


def fit_freeform(P, N, tol, margin, kind=None, max_ctrl=(48, 32)):
    """Fit a B-spline surface to points P (normals N). Returns (Primitive or None, info dict)."""
    info = {}
    cands = []
    if kind in (None, "tube"):
        tp = tube_params(P, N)
        if tp is not None:
            cands.append(("tube", tp["s"], tp["phi"], True, tp))
    if kind in (None, "patch"):
        uv = _uv_patch(P, N)
        if uv is not None:
            cands.append(("patch", uv[0], uv[1], False, {}))
    best = None
    for name, u, v, pv, extra in cands:
        ua, ub = float(u.min() - margin), float(u.max() + margin)
        if pv:
            va, vb = 0.0, 2 * np.pi
            circ = 2 * np.pi * extra["rho"]
        else:
            va, vb = float(v.min() - margin), float(v.max() + margin)
            circ = vb - va
        lu = ub - ua
        # control spacing: start coarse, refine until the fit is inside the tolerance
        for sp in (8.0, 5.0, 3.5, 2.5, 1.8):
            h = sp * max(tol, 0.05) * 10
            nu = int(np.clip(lu / h, 4, max_ctrl[0]))
            nv = int(np.clip(circ / h, 8 if pv else 4, max_ctrl[1]))
            Bu = _basis_clamped(u, ua, ub, nu)
            Bv = _basis_periodic(v, nv) if pv else _basis_clamped(v, va, vb, nv)
            A = _tensor(Bu, Bv)
            Q = _solve(A, P, nu, nv, pv, lam=1e-3 * len(P) / (nu * nv))
            res = np.linalg.norm(A @ Q.reshape(-1, 3) - P, axis=1)
            rms, p95 = float(np.sqrt(np.mean(res ** 2))), float(np.percentile(res, 95))
            rec = {"param": name, "nu": nu, "nv": nv, "rms": rms, "p95": p95, "max": float(res.max())}
            if best is None or rms < best[1]["rms"]:
                best = (Surf(Q, ua, ub, pv, va, vb), rec)
            if rms <= 0.5 * tol and p95 <= 1.2 * tol:
                break
    if best is None:
        info["error"] = "no parametrisation (neither tube nor height field)"
        return None, info
    surf, rec = best
    info.update(rec)
    prim = F.Primitive("bspline", {"poles": surf.Q, "ua": surf.ua, "ub": surf.ub, "va": surf.va, "vb": surf.vb,
                                   "periodic_v": float(surf.pv)})
    prim._surf = surf
    prim._sampler = Sampler(surf)
    d = prim._sampler.distance(P)
    prim.rms, prim.max_err = float(np.sqrt(np.mean(d ** 2))), float(np.abs(d).max())
    sn = prim._sampler.normal(P)
    c = np.abs(np.sum(sn * N, axis=1)).clip(0, 1)
    prim.normal_err_deg = float(np.degrees(np.sqrt(np.mean(np.arccos(c) ** 2))))
    info["rms_true"], info["p95_true"] = prim.rms, float(np.percentile(np.abs(d), 95))
    return prim, info


def sampler_of(prim):
    if getattr(prim, "_sampler", None) is None:
        q = prim.params
        prim._surf = Surf(np.asarray(q["poles"]), q["ua"], q["ub"], bool(q["periodic_v"]), q["va"], q["vb"])
        prim._sampler = Sampler(prim._surf)
    return prim._sampler


# ---------------------------------------------------------------- OCC
def to_occ(prim):
    """Geom_BSplineSurface equal to the fitted surface (checked against our own evaluation)."""
    from OCP.Geom import Geom_BSplineSurface
    from OCP.collections import Array2_gp_Pnt as TColgp_Array2OfPnt, Array1_double as TColStd_Array1OfReal
    from OCP.collections import Array1_int as TColStd_Array1OfInteger
    from OCP.gp import gp_Pnt
    q = prim.params
    Q = np.asarray(q["poles"])
    nu, nv = Q.shape[:2]
    pv = bool(q["periodic_v"])
    uk = np.linspace(q["ua"], q["ub"], nu - DEG + 1)
    um = [DEG + 1] + [1] * (len(uk) - 2) + [DEG + 1]
    if pv:
        vk = np.linspace(0, 2 * np.pi, nv + 1)
        vm = [1] * (nv + 1)
    else:
        vk = np.linspace(q["va"], q["vb"], nv - DEG + 1)
        vm = [DEG + 1] + [1] * (len(vk) - 2) + [DEG + 1]

    def arr_r(x):
        a = TColStd_Array1OfReal(1, len(x))
        for i, v in enumerate(x):
            a.SetValue(i + 1, float(v))
        return a

    def arr_i(x):
        a = TColStd_Array1OfInteger(1, len(x))
        for i, v in enumerate(x):
            a.SetValue(i + 1, int(v))
        return a

    surf = getattr(prim, "_surf", None) or sampler_of(prim) and prim._surf
    rng = np.random.default_rng(0)
    tu = rng.uniform(q["ua"], q["ub"], 20)
    tv = rng.uniform(q["va"], q["vb"], 20)
    ref = surf.eval(tu, tv)
    best = None
    for shift in (range(nv) if pv else [0]):
        Qs = np.roll(Q, shift, axis=1)
        poles = TColgp_Array2OfPnt(1, nu, 1, nv)
        for i in range(nu):
            for j in range(nv):
                poles.SetValue(i + 1, j + 1, gp_Pnt(*map(float, Qs[i, j])))
        S = Geom_BSplineSurface(poles, arr_r(uk), arr_r(vk), arr_i(um), arr_i(vm), DEG, DEG, False, pv)
        got = np.array([[S.Value(float(a), float(b)).X(), S.Value(float(a), float(b)).Y(),
                         S.Value(float(a), float(b)).Z()] for a, b in zip(tu, tv)])
        err = float(np.abs(got - ref).max())
        if best is None or err < best[0]:
            best = (err, S)
        if err < 1e-7:
            break
    if best[0] > 1e-5:
        raise RuntimeError(f"OCC B-spline does not reproduce the fit (max {best[0]:.2e})")
    return best[1]


# ---------------------------------------------------------------- region merging
def _skin_like(prim, diag, center):
    """A quadric far larger than the part (cylinder / sphere radius > the part size, torus > 2 x,
    cone apex > 3 x away): on a faceted CAD skin that is a strip of a free-form surface, not a design
    primitive."""
    q = prim.params
    if prim.kind in ("cylinder", "sphere") and float(q.get("radius", 0)) > diag:
        return True
    if prim.kind == "torus" and float(q.get("major_radius", 0)) + float(q.get("minor_radius", 0)) > 2 * diag:
        return True
    if prim.kind == "cone":
        return float(np.linalg.norm(np.asarray(q["apex"], float) - center)) > 3 * diag
    return False


def smooth_clusters(mesh, labels, regions, max_angle_deg=12.0, min_regions=3, faceted=False):
    """Clusters of regions joined across G1 (smooth) boundaries where the two sides are the same
    primitive kind with different parameters: one smooth surface the segmenter cut into several
    primitives (e.g. a bent, ovalised pipe split into tori). Planes never join."""
    fa = np.asarray(mesh.face_adjacency)
    ang = np.degrees(np.asarray(mesh.face_adjacency_angles))
    la, lb = labels[fa[:, 0]], labels[fa[:, 1]]
    m = (la >= 0) & (lb >= 0) & (la != lb)
    by = {r.id: r for r in regions}
    pair = {}
    for a, b, g in zip(la[m], lb[m], ang[m]):
        k = (min(a, b), max(a, b))
        pair.setdefault(k, []).append(g)
    parent = {r.id: r.id for r in regions}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    diag = float(np.linalg.norm(np.ptp(mesh.vertices, axis=0)))
    # faceted CAD: big facets give few boundary edges; skin strips fitted as huge quadrics of any kind join
    ctr = mesh.bounds.mean(0)
    # on a faceted skin, strips come out as quadrics of any kind (often huge ones): all curved regions
    # meeting across a smooth boundary are one candidate surface; the B-spline fit decides
    kind = lambda p: "smooth" if faceted else p.kind   # noqa: E731
    for (a, b), gs in pair.items():
        pa, pb = by[a].primitive, by[b].primitive
        if pa is None or pb is None or pa.kind == "plane" or pb.kind == "plane" or kind(pa) != kind(pb):
            continue
        if "thread_zone" in by[a].flags or "thread_zone" in by[b].flags:
            continue
        if len(gs) < (1 if faceted else 5) or float(np.median(gs)) > max_angle_deg:
            continue
        parent[find(a)] = find(b)
    groups = {}
    for r in regions:
        groups.setdefault(find(r.id), []).append(r.id)
    need = 2 if faceted else min_regions
    return [sorted(g) for g in groups.values() if len(g) >= need]


def apply_freeform(mesh, g, regions, labels, tol, region_points, accept_rms=0.5, accept_p95=1.5, log=None,
                   faceted=False, deadline=None):
    """Replace each smooth multi-primitive cluster by one B-spline region when the single surface fits
    the cluster within the tolerance (rms <= accept_rms*tol, p95 <= accept_p95*tol); small smooth
    neighbours (and unassigned faces) lying on the fitted surface are absorbed. Returns regions,
    labels (renumbered when changed) and a log."""
    log = [] if log is None else log
    labels = labels.copy()
    by = {r.id: r for r in regions}
    area = np.asarray(mesh.area_faces)
    C = np.asarray(mesh.triangles_center)
    changed = False
    work = list(smooth_clusters(mesh, labels, regions, faceted=faceted))
    diag = float(np.linalg.norm(np.ptp(np.asarray(mesh.vertices), axis=0)))
    ctr = mesh.bounds.mean(0)
    import time as _time
    while work:
        if deadline is not None and _time.time() > deadline:
            # 1.0.2a stage cap: stop cleanly, the unfitted clusters keep their primitive regions
            log.append({"regions": [], "accepted": False, "stopped": f"time cap reached, {len(work)} cluster(s) not tried"})
            break
        cl = work.pop(0)
        faces = np.where(np.isin(labels, cl))[0]
        P, N = region_points(g, faces) if len(faces) else (np.zeros((0, 3)), np.zeros((0, 3)))
        if len(P) < 20:                       # cluster without enough samples: nothing to fit
            log.append({"regions": [int(x) for x in cl], "accepted": False, "reason": f"only {len(P)} samples"})
            continue
        ext = float(np.linalg.norm(np.ptp(P, axis=0)))
        prim, info = fit_freeform(P, N, tol, margin=max(1.0, 0.08 * ext))
        rec = {"regions": [int(x) for x in cl], "area": round(float(area[faces].sum()), 2), **info}
        if prim is None or info["rms_true"] > accept_rms * tol or info["p95_true"] > accept_p95 * tol:
            rec["accepted"] = False
            log.append(rec)
            if faceted:
                # faceted mode joins every curved region across smooth seams; when one surface does not
                # explain them all, retry the skin strips (huge quadrics) and each design primitive kind apart
                key = lambda i: "skin" if _skin_like(by[i].primitive, diag, ctr) else by[i].primitive.kind   # noqa: E731
                subs = {}
                for i in cl:
                    if by[i].primitive is not None:
                        subs.setdefault(key(i), []).append(i)
                for sub in subs.values():
                    if 2 <= len(sub) < len(cl):
                        work.append(sorted(sub))
            continue
        tid = cl[0]
        labels[faces] = tid
        # absorb: unassigned faces and small non-plane neighbours whose faces lie on the surface
        d = np.abs(prim.distance(C))
        near = d < 2 * tol
        for r in regions:
            if r.id in cl or r.primitive is None or "thread_zone" in r.flags:
                continue
            f = np.where(labels == r.id)[0]
            big_ok = faceted and _skin_like(r.primitive, diag, ctr)      # a skin strip of any size
            if len(f) and (r.area < 0.15 * rec["area"] or big_ok) and np.mean(near[f]) > 0.9:
                labels[f] = tid
                rec.setdefault("absorbed", []).append(int(r.id))
        un = (labels < 0) & near
        labels[un] = tid
        rec["absorbed_unassigned_faces"] = int(un.sum())
        f = np.where(labels == tid)[0]
        by[tid].primitive = prim
        by[tid].faces = f
        by[tid].area = float(area[f].sum())
        by[tid].flags = [x for x in by[tid].flags if x not in ("high_fit_residual", "small_region")] + ["freeform"]
        rec["accepted"] = True
        rec["region"] = int(tid)
        log.append(rec)
        changed = True
    if not changed:
        return regions, labels, log
    out, remap = [], {}
    for r in regions:
        f = np.where(labels == r.id)[0]
        if len(f) == 0:
            continue
        r.faces = f
        remap[r.id] = len(out)
        out.append(r)
    lab2 = -np.ones_like(labels)
    for o, n in remap.items():
        lab2[labels == o] = n
    for r in out:
        r.id = remap[r.id]
        fl = []
        for x in r.flags:
            if x.startswith("coaxial_cone_with_"):
                j = int(x.rsplit("_", 1)[1])
                if j not in remap:
                    continue
                x = f"coaxial_cone_with_{remap[j]}"
            fl.append(x)
        r.flags = fl
    for rec in log:
        if rec.get("accepted"):
            rec["region"] = remap[rec["region"]]
    return out, lab2, log
