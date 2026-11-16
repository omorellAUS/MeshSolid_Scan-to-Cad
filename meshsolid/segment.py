"""Sharp-edge detection and fit-driven region growing segmentation.

Pipeline:
 1. faces are split into components by sharp edges (dihedral angle > sharp_deg)
 2. each component is first tried as a single primitive (fast path)
 3. components that no single primitive explains are decomposed greedily:
    candidates are grown from seeds for every primitive kind, the candidate
    covering the most area wins, its faces are removed, repeat (Schnabel-style
    greedy coverage, but with connectivity-aware growing instead of RANSAC)
 4. boundary relaxation: faces adjacent to several regions go to the primitive
    with smallest distance, then all regions are refit.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from . import fit as F

PATCH_SIZE = {"plane": 3, "cylinder": 8, "cone": 10, "sphere": 8, "torus": 16}
KIND_ORDER = {"plane": 0, "cylinder": 1, "cone": 2, "sphere": 3, "torus": 4}


@dataclass
class Region:
    id: int
    faces: np.ndarray
    primitive: F.Primitive | None
    area: float
    confidence: float = 0.0
    flags: list = field(default_factory=list)

    def to_json(self):
        return {"id": self.id, "n_faces": int(len(self.faces)), "area": float(self.area),
                "primitive": None if self.primitive is None else self.primitive.to_json(),
                "confidence": float(self.confidence), "flags": self.flags}


@dataclass
class SegParams:
    sharp_deg: float = 30.0
    tol: float = 0.1            # max point-to-primitive distance to belong to a region (mm)
    normal_deg: float = 20.0    # max face-normal vs surface-normal deviation while growing
    min_area_frac: float = 2e-4 # regions smaller than this fraction of total area are not created
    seeds_per_iter: int = 8
    max_iters: int = 400
    kinds: tuple = ("plane", "cylinder", "cone", "sphere", "torus")
    strict_torus: bool = False  # CAD tessellation: reject tori a narrow band cannot determine (torus_determined)


class MeshGraph:
    def __init__(self, mesh: trimesh.Trimesh):
        self.m = mesh
        arr = lambda a: np.array(a, copy=True).view(np.ndarray)  # plain arrays: TrackedArray is slow
        self.V = arr(mesh.vertices)
        self.Fc = arr(mesh.faces)
        self.fn = arr(mesh.face_normals)
        self.area = arr(mesh.area_faces)
        self.cent = arr(mesh.triangles_center)
        self.adj = arr(mesh.face_adjacency)
        self.adj_ang = np.degrees(arr(mesh.face_adjacency_angles))
        n = len(self.Fc)
        nb = [[] for _ in range(n)]
        for (a, b), ang in zip(self.adj, self.adj_ang):
            nb[a].append((b, ang))
            nb[b].append((a, ang))
        self.nb = nb


def region_points(g: MeshGraph, faces):
    """Unique vertices of the faces with region-local area-weighted vertex normals."""
    tri = g.Fc[faces]
    vid, inv = np.unique(tri.ravel(), return_inverse=True)
    N = np.zeros((len(vid), 3))
    np.add.at(N, inv, np.repeat(g.fn[faces] * g.area[faces, None], 3, axis=0))
    N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-15)
    return g.V[vid], N


def fit_region(g, faces, kind, init=None):
    P, N = region_points(g, faces)
    try:
        if init is not None and init.kind == kind == "torus":
            return F.fit_torus(P, N, init=init)
        if init is not None and init.kind == kind == "cylinder":
            return F.fit_cylinder(P, N, axis0=init.params["axis"])
        if kind == "plane":
            return F.fit_plane(P, N)
        if kind == "cylinder":
            if len(P) < 6:
                return None
            return F.fit_cylinder(P, N)
        if kind == "sphere":
            if len(P) < 5:
                return None
            return F.fit_sphere(P, N)
        if kind == "cone":
            if len(P) < 8:
                return None
            return F.fit_cone(P, N)
        if kind == "torus":
            if len(P) < 12:
                return None
            return F.fit_torus(P, N)
    except Exception:
        return None


def _face_ok(g, prim, fids, tol, cosmin):
    """Vectorised membership test for faces against primitive."""
    tri = g.V[g.Fc[fids]]  # (k,3,3)
    d = np.abs(F.signed_distance(prim, tri.reshape(-1, 3))).reshape(-1, 3).max(1)
    sn = F.surface_normal(prim, g.cent[fids])
    c = np.abs(np.sum(sn * g.fn[fids], axis=1))
    return (d <= tol) & (c >= cosmin)


def _plausible(prim, extent):
    if prim is None or not np.isfinite(prim.rms):
        return False
    q = prim.params
    if prim.kind in ("cylinder", "sphere") and q["radius"] > 20 * max(extent, 1e-6):
        return False
    if prim.kind == "torus" and (q["major_radius"] > 20 * max(extent, 1e-6) or q["minor_radius"] > 2 * extent
                                 or q["major_radius"] < 0.2 * q["minor_radius"]):
        return False
    if prim.kind == "cone" and (q["half_angle"] < np.radians(3) or q["half_angle"] > np.radians(85)):
        return False
    return True


def grow(g, seed_faces, kind, allowed, sp: SegParams, extent, prim=None):
    """Grow a region of `kind` from seed faces within `allowed` (bool mask), not crossing sharp edges."""
    region = set(int(f) for f in seed_faces)
    if prim is None:
        prim = fit_region(g, np.fromiter(region, int), kind)
    if not _plausible(prim, extent):
        return None, None
    cosmin = np.cos(np.radians(sp.normal_deg))
    for _round in range(12):
        last_fit = len(region)
        frontier = deque(region)
        seen = set(region)
        while frontier:
            f = frontier.popleft()
            cand = [n for n, ang in g.nb[f] if n not in seen and allowed[n] and ang <= sp.sharp_deg]
            if not cand:
                continue
            seen.update(cand)
            cand = np.array(cand)
            ok = _face_ok(g, prim, cand, sp.tol, cosmin)
            for n in cand[ok]:
                region.add(int(n))
                frontier.append(int(n))
            if len(region) > 1.3 * last_fit + 10:
                p2 = fit_region(g, np.fromiter(region, int), kind, init=prim)
                if _plausible(p2, extent):
                    prim = p2
                last_fit = len(region)
        p2 = fit_region(g, np.fromiter(region, int), kind, init=prim)
        if not _plausible(p2, extent):
            return None, None
        prim = p2
        # drop faces that no longer fit after refit, keep largest connected piece implicitly via next round
        fl = np.fromiter(region, int)
        keep = _face_ok(g, prim, fl, sp.tol * 1.5, cosmin)
        region = set(fl[keep].tolist()) or region
        if len(region) == last_fit:
            break
    faces = np.fromiter(region, int)
    if len(faces) == 0:
        return None, None
    if sp.strict_torus and prim.kind == "torus" and not torus_determined(prim, g.V[g.Fc[faces]].reshape(-1, 3)):
        # a narrow band (e.g. the 1.6 mm wall of a slot in a plate) is explained by many tori within
        # tolerance; the torus is not determined by it and trims wildly outside the band
        return None, None
    return faces, prim


TORUS_MIN_TUBE_SPAN_DEG = 20.0
TORUS_MIN_RING_COVER_DEG = 20.0


def torus_ring_cover_deg(prim, P, bin_deg=2.0):
    """Angular coverage (deg, occupied bins) of points around the torus axis (along the ring)."""
    q = prim.params
    a = np.asarray(q["axis"], float)
    h = np.asarray(P, float) - np.asarray(q["center"], float)
    r = h - np.outer(h @ a, a)
    u = np.cross(a, [1.0, 0, 0] if abs(a[0]) < 0.9 else [0, 1.0, 0])
    u /= np.linalg.norm(u)
    v = np.cross(a, u)
    th = np.degrees(np.arctan2(r @ v, r @ u)) % 360.0
    return float(len(np.unique((th // bin_deg).astype(int))) * bin_deg)


def torus_determined(prim, P):
    """A torus is only determined by a region that spans enough of both its circles: a narrow band
    (the 1.6 mm wall of a slot in a plate, a strip of a faceted skin) is explained by many tori within
    tolerance, and the fitted one trims wildly outside the band."""
    return (torus_tube_span_deg(prim, P) >= TORUS_MIN_TUBE_SPAN_DEG
            and torus_ring_cover_deg(prim, P) >= TORUS_MIN_RING_COVER_DEG)


def torus_tube_span_deg(prim, P):
    """Angular span (deg) of points around the torus tube (the cross-section circle)."""
    q = prim.params
    a = np.asarray(q["axis"], float)
    h = np.asarray(P, float) - np.asarray(q["center"], float)
    ha = h @ a
    rho = np.linalg.norm(h - np.outer(ha, a), axis=1)
    phi = np.sort(np.arctan2(ha, rho - q["major_radius"]))
    if len(phi) < 2:
        return 0.0
    gaps = np.diff(np.r_[phi, phi[0] + 2 * np.pi])
    return float(np.degrees(2 * np.pi - gaps.max()))


def _seed_patch(g, f, allowed, sp, n=12):
    out, q, seen = [f], deque([f]), {f}
    while q and len(out) < n:
        x = q.popleft()
        for y, ang in g.nb[x]:
            if y not in seen and allowed[y] and ang <= sp.sharp_deg:
                seen.add(y)
                out.append(y)
                q.append(y)
    return np.array(out)


def sharp_components(g: MeshGraph, sp: SegParams):
    smooth = g.adj_ang <= sp.sharp_deg
    a = g.adj[smooth]
    n = len(g.Fc)
    M = coo_matrix((np.ones(len(a)), (a[:, 0], a[:, 1])), shape=(n, n))
    k, lab = connected_components(M, directed=False)
    return lab, int((~smooth).sum())


def segment(mesh: trimesh.Trimesh, sp: SegParams | None = None, rng_seed: int = 0):
    sp = sp or SegParams()
    g = MeshGraph(mesh)
    rng = np.random.default_rng(rng_seed)
    total = g.area.sum()
    min_area = sp.min_area_frac * total
    lab, n_sharp = sharp_components(g, sp)
    label = -np.ones(len(g.Fc), int)
    prims = []
    # run time budget: region growing may take at most 40% of what the run has left (the build needs
    # the rest); faces not reached stay unassigned and the stop is reported
    import time as _time
    from . import budget as _B
    _left = _B.time_left()
    t_stop = _time.time() + 0.4 * _left if np.isfinite(_left) else None
    budget_hit = False
    for c in np.unique(lab):
        if t_stop is not None and _time.time() > t_stop:
            budget_hit = True
            break
        comp = np.where(lab == c)[0]
        ext = np.linalg.norm(np.ptp(g.cent[comp], axis=0)) if len(comp) > 1 else 1.0
        if g.area[comp].sum() < min_area:
            continue  # tiny component (remesh/scan artefact): left unassigned, reported
        # fast path: single primitive explains whole component
        best = None
        for k in sp.kinds:
            p = fit_region(g, comp, k)
            if _plausible(p, ext) and p.max_err <= sp.tol * 1.5 and p.normal_err_deg < sp.normal_deg / 2:
                if sp.strict_torus and p.kind == "torus" and not torus_determined(p, g.V[g.Fc[comp]].reshape(-1, 3)):
                    continue
                best = p
                break
        if best is not None:
            label[comp] = len(prims)
            prims.append(best)
            continue
        allowed = np.zeros(len(g.Fc), bool)
        allowed[comp] = True
        empty_rounds = 0
        for _ in range(sp.max_iters):
            if t_stop is not None and _time.time() > t_stop:
                budget_hit = True
                break
            free = np.where(allowed)[0]
            if g.area[free].sum() < min_area:
                break
            # seeds: prefer faces whose neighbours are smooth (interior of surfaces)
            ns = min(sp.seeds_per_iter, len(free))
            seeds = rng.choice(free, ns, replace=False, p=g.area[free] / g.area[free].sum())
            cands = []
            for s in seeds:
                for k in sp.kinds:
                    # smallest seed patch with enough points for the kind (big patches straddle blends)
                    patch = _seed_patch(g, int(s), allowed, sp, n=PATCH_SIZE[k])
                    p0 = fit_region(g, patch, k)
                    # only grow kinds that already explain the seed patch
                    if not _plausible(p0, ext) or p0.max_err > sp.tol or p0.normal_err_deg > sp.normal_deg:
                        continue
                    faces, p = grow(g, patch, k, allowed, sp, ext, prim=p0)
                    if faces is None:
                        continue
                    a = g.area[faces].sum()
                    cands.append((a * (1 - 0.03 * KIND_ORDER[k]), faces, p))
            if not cands:
                empty_rounds += 1
                if empty_rounds >= 6:
                    break
                continue
            empty_rounds = 0
            sc, faces, p = max(cands, key=lambda t: t[0])
            if g.area[faces].sum() < min_area:
                break
            label[faces] = len(prims)
            prims.append(p)
            allowed[faces] = False
    _relax(g, label, prims, sp)
    n_simpl, n_abs = _consolidate(g, label, prims, sp)
    n_merged = _merge_small(g, label, prims, sp, total)
    regions, label = _make_regions(g, label, prims, sp)
    info = {"sharp_edges": n_sharp, "sharp_components": int(lab.max() + 1),
            "small_regions_merged": n_merged,
            "regions_simplified": n_simpl, "regions_absorbed": n_abs,
            "unassigned_faces": int((label < 0).sum()),
            "unassigned_area_frac": float(g.area[label < 0].sum() / total),
            "budget_hit": budget_hit}
    return regions, label, info


def _explains(g, prim, faces, tol, cosmin):
    if prim is None or len(faces) == 0:
        return False
    return bool(_face_ok(g, prim, faces, tol, cosmin).all())


def _consolidate(g, label, prims, sp, passes=6):
    """Model selection after growing: (1) replace a region's primitive by a simpler kind that
    explains it equally (e.g. a sphere/torus fitted to a narrow band of a cylinder);
    (2) absorb a region into an adjacent region whose primitive already explains all its faces
    (the same surface split into pieces), simpler kinds and larger regions absorbing first."""
    cosmin = np.cos(np.radians(sp.normal_deg))
    kinds = sorted(sp.kinds, key=lambda k: KIND_ORDER[k])
    n_simpl = n_abs = 0
    a, b = g.adj[:, 0], g.adj[:, 1]
    a, b = g.adj[:, 0], g.adj[:, 1]
    for i, p in enumerate(prims):
        faces = np.where(label == i)[0]
        if p is None or len(faces) == 0:
            continue
        ext = np.linalg.norm(np.ptp(g.cent[faces], axis=0)) if len(faces) > 1 else 1.0
        for k in kinds:
            if KIND_ORDER[k] >= KIND_ORDER[p.kind]:
                break
            q = fit_region(g, faces, k)
            if _plausible(q, ext) and q.max_err <= sp.tol and _explains(g, q, faces, sp.tol * 1.5, cosmin):
                prims[i] = q
                n_simpl += 1
                break
            # robust variant: the simpler surface explains >= 90% of the area exactly and the few
            # remaining faces (blend triangles swallowed at a tangent boundary) belong to neighbours
            if q is None or len(faces) < 6:
                continue
            ok = _face_ok(g, q, faces, sp.tol, cosmin)
            for _ in range(2):
                if ok.sum() < 3:
                    break
                q2 = fit_region(g, faces[ok], k, init=q)
                if q2 is None:
                    break
                q = q2
                ok = _face_ok(g, q, faces, sp.tol, cosmin)
            if not _plausible(q, ext) or g.area[faces[ok]].sum() < 0.9 * g.area[faces].sum() or ok.all():
                continue
            m = (label[a] == i) & (label[b] != i)
            nbs = set(label[b[m]].tolist()) | set(label[a[(label[b] == i) & (label[a] != i)]].tolist())
            nbs.discard(-1)
            handoff = {}
            for f in faces[~ok]:
                tgt = [j for j in nbs if prims[j] is not None and _face_ok(g, prims[j], np.array([f]), sp.tol, cosmin)[0]]
                if not tgt:
                    handoff = None
                    break
                handoff[f] = tgt[0]
            if handoff is None:
                continue
            for f, j in handoff.items():
                label[f] = j
            prims[i] = fit_region(g, faces[ok], k, init=q) or q
            n_simpl += 1
            break
    # (1b) a region explained by another region's primitive (e.g. one lobe face of a cylinder
    # shared by several lobes, fitted on a few slivers) takes over that better-supported surface
    area = np.bincount(label[label >= 0], weights=g.area[label >= 0], minlength=len(prims))
    nfac = np.bincount(label[label >= 0], minlength=len(prims))
    for i in sorted(range(len(prims)), key=lambda i: area[i]):
        faces = np.where(label == i)[0]
        if prims[i] is None or len(faces) == 0:
            continue
        best = None
        for j in range(len(prims)):
            if j == i or prims[j] is None or area[j] < 2 * area[i] or nfac[j] < 2 * nfac[i]:
                continue
            if nfac[i] >= 30 and KIND_ORDER[prims[j].kind] > KIND_ORDER[prims[i].kind]:
                continue
            if _explains(g, prims[j], faces, sp.tol, cosmin):
                key = (KIND_ORDER[prims[j].kind], -area[j])
                if best is None or key < best[0]:
                    best = (key, j)
        if best is not None:
            prims[i] = prims[best[1]]
            n_simpl += 1
    for _ in range(passes):
        changed = False
        area = np.bincount(label[label >= 0], weights=g.area[label >= 0], minlength=len(prims))
        order = sorted(range(len(prims)), key=lambda i: (-KIND_ORDER[prims[i].kind] if prims[i] is not None else 0, area[i]))
        for i in order:
            faces = np.where(label == i)[0]
            if len(faces) == 0:
                continue
            m = (label[a] == i) & (label[b] != i)
            nbs = set(label[b[m]].tolist()) | set(label[a[(label[b] == i) & (label[a] != i)]].tolist())
            nbs.discard(-1)
            cands = [j for j in nbs if prims[j] is not None and _explains(g, prims[j], faces, sp.tol, cosmin)]
            if not cands:
                continue
            j = min(cands, key=lambda j: (KIND_ORDER[prims[j].kind], -area[j]))
            merged = np.r_[np.where(label == j)[0], faces]
            q = fit_region(g, merged, prims[j].kind, init=prims[j])
            if q is not None and q.max_err <= sp.tol * 1.5 and _explains(g, q, merged, sp.tol * 1.5, cosmin):
                prims[j] = q
            label[faces] = j
            area[j] += area[i]
            area[i] = 0
            n_abs += 1
            changed = True
        if not changed:
            break
    return n_simpl, n_abs


def _merge_small(g, label, prims, sp, total, frac=5e-3):
    """Absorb small regions into an adjacent region whose primitive already explains them
    (spurious slivers at blend boundaries); genuinely different small surfaces are kept."""
    merged = 0
    a, b = g.adj[:, 0], g.adj[:, 1]
    for i in sorted(range(len(prims)), key=lambda i: g.area[label == i].sum()):
        faces = np.where(label == i)[0]
        if len(faces) == 0 or g.area[faces].sum() > frac * total:
            continue
        m = (label[a] == i) & (label[b] != i)
        nbs = set(label[b[m]].tolist()) | set(label[a[(label[b] == i) & (label[a] != i)]].tolist())
        nbs.discard(-1)
        P, _ = region_points(g, faces)
        best, bd = None, np.inf
        for j in nbs:
            d = float(np.abs(F.signed_distance(prims[j], P)).max())
            if d < bd:
                best, bd = j, d
        if best is not None and bd <= 1.5 * sp.tol:
            label[faces] = best
            merged += 1
        elif len(faces) < 20 and g.area[faces].sum() < 1e-3 * total:
            label[faces] = -1   # outlier sliver: left unassigned (reported), neighbours' surfaces cover it
            merged += 1
    return merged


def _relax(g, label, prims, sp, iters=2):
    cosmin = np.cos(np.radians(sp.normal_deg))
    for _ in range(iters):
        changed = 0
        a, b = g.adj[:, 0], g.adj[:, 1]
        border = np.unique(np.r_[a[label[a] != label[b]], b[label[a] != label[b]]])
        for f in border:
            cands = {label[n] for n, ang in g.nb[f] if ang <= sp.sharp_deg and label[n] >= 0}
            cands.add(label[f]) if label[f] >= 0 else None
            if not cands:
                continue
            tri = g.V[g.Fc[f]]
            best, bd = label[f], np.inf
            for c in cands:
                if c < 0:
                    continue
                d = np.abs(F.signed_distance(prims[c], tri)).max()
                cn = abs(F.surface_normal(prims[c], g.cent[f][None])[0] @ g.fn[f])
                if cn >= cosmin and d < bd:
                    best, bd = c, d
            if best != label[f] and bd <= sp.tol:
                label[f] = best
                changed += 1
        for i in range(len(prims)):
            faces = np.where(label == i)[0]
            if len(faces):
                p = fit_region(g, faces, prims[i].kind)
                if p is not None:
                    prims[i] = p
        if not changed:
            break


def _make_regions(g, label, prims, sp):
    regs = []
    new = -np.ones_like(label)
    for i, p in enumerate(prims):
        faces = np.where(label == i)[0]
        if len(faces) == 0:
            continue
        a = float(g.area[faces].sum())
        new[faces] = len(regs)
        regs.append(Region(len(regs), faces, p, a))
    return regs, new
