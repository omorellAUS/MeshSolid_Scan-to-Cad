"""Section-loft builder for smooth swept skins (wings, ailerons, fuselage shells, hatches).

Faceted CAD exports of lofted parts carry a skin that is tangent-continuous all the way round the
leading edge; segmenting it into a B-spline skin plus near-coaxial LE cylinders/cones and trimming
those against each other is ill-conditioned (the surfaces meet tangentially). Here the smooth skin is
never split at curvature changes: the part is cut by planes across a sweep direction d, every
cross-section loop is split only at its sharp corners (turning above `corner_deg`, i.e. real feature
edges), each corner-to-corner run becomes one B-spline (identical parameters in every section, so
the knots match), and BRepOffsetAPI_ThruSections lofts the sections into one continuous face per run
with planar end caps. Planar steps normal to d (cap levels, e.g. an aileron bay) split the part into
slabs that are lofted separately and fused; inner loops (spar tubes, thin-wall shell interiors) are
lofted the same way and cut out. Sections are refined along d until the mid-section between two
neighbours is predicted within tol/4. The result goes through the pipeline's deviation gate like
every other builder; nothing here is tuned per part."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from .extrude import _area2, _inside, _loops, simplify_loop


@dataclass
class LoftResult:
    ok: bool = False
    method: str = "section_loft"
    shape: object = None
    reason: str = ""
    sweep_dir: list = None
    sweep_area_frac: float = 0.0
    levels: list = field(default_factory=list)
    slabs: list = field(default_factory=list)
    brepcheck_valid: bool = False
    n_solids: int = 0
    n_faces: int = 0
    volume: float = 0.0
    mesh_volume: float = 0.0
    self_check: dict = None
    seconds: float = 0.0

    def to_json(self):
        return {k: v for k, v in self.__dict__.items() if k != "shape"}


class LoftError(Exception):
    pass


def _hk(h):
    """Stable dict key for section heights (float linspace noise)."""
    return round(float(h), 9)


def sweep_axis(mesh, side_deg=30.0, cap_deg=1.0):
    """(d, frac, cap_frac): direction maximising the area of faces that are caps (normal ~ +-d) or
    swept skin (normal within side_deg of the plane normal to d). Candidates: principal axes of the
    vertices and the normals of the largest faces."""
    N = np.asarray(mesh.face_normals)
    A = np.asarray(mesh.area_faces)
    tot = float(A.sum()) or 1.0
    V = np.asarray(mesh.vertices)
    _, ev = np.linalg.eigh(np.cov((V - V.mean(0)).T))
    cand = [ev[:, i] for i in range(3)]
    for i in np.argsort(-A)[:300]:
        if all(abs(float(N[i] @ c)) < 0.999 for c in cand):
            cand.append(N[i])
        if len(cand) >= 12:
            break
    cc, ss = math.cos(math.radians(cap_deg)), math.sin(math.radians(side_deg))
    best = (None, 0.0, 0.0)
    for d in cand:
        dot = np.abs(N @ d)
        cap = dot > cc
        f = float(A[cap | (dot < ss)].sum() / tot)
        if f > best[1]:
            best = (d / np.linalg.norm(d), f, float(A[cap].sum() / tot))
    return best


def _cap_levels(mesh, d, merge, min_area):
    """Heights of planar faces normal to d (steps), clustered; the two extremes always included."""
    N = np.asarray(mesh.face_normals)
    A = np.asarray(mesh.area_faces)
    h = np.asarray(mesh.triangles_center) @ d
    cap = np.abs(N @ d) > math.cos(math.radians(1.0))
    hv = np.asarray(mesh.vertices) @ d
    lo, hi = float(hv.min()), float(hv.max())
    order = np.argsort(h[cap])
    hs, As = h[cap][order], A[cap][order]
    groups = []
    for x, a in zip(hs, As):
        if groups and x - groups[-1][0][-1] <= merge:
            groups[-1][0].append(x)
            groups[-1][1] += a
        else:
            groups.append([[x], a])
    lv = [lo, hi]
    for g, a in groups:
        m = float(np.mean(g))
        if a >= min_area and lo + merge < m < hi - merge:
            lv.append(m)
    lv = sorted(lv)
    out = [lv[0]]
    for x in lv[1:]:
        if x - out[-1] > merge:
            out.append(x)
    out[-1] = hi
    return out



def _corner_count_breaks(mesh, d, u, v, tol, ext, lo, hi, th_deg=35.0, samples=41,
                         min_slab=None, persist=5):
    """Insert loft-slab breaks where the number of sharp section corners changes.

    Samples cross-sections along d; counts turns above th_deg on the outer loop. Counts are
    median-filtered (window 5) so tessellation flicker does not invent slabs. A break is
    placed between two regimes when both last at least `persist` stations and the filtered
    counts differ by >= 4. Each slab then lofts with a stable corner set.
    """
    lo, hi = float(lo), float(hi)
    if hi - lo <= 0:
        return [lo, hi]
    min_slab = float(min_slab) if min_slab is not None else max(12.0 * tol, 0.12 * (hi - lo), 12.0)
    th = math.radians(th_deg)
    eps = max(tol, 1e-4 * (hi - lo))
    hs = np.linspace(lo + eps, hi - eps, int(samples))
    raw = []
    for h in hs:
        try:
            sec = _section(mesh, d, u, v, float(h), tol, ext)
            if not sec.loops:
                raw.append(-1)
                continue
            Q = max(sec.loops, key=lambda q: abs(_area2(q)))
            raw.append(int(np.sum(np.abs(_turn(Q)) > th)))
        except LoftError:
            raw.append(-1)
    counts = list(raw)
    for i in range(len(raw)):
        if raw[i] < 0:
            continue
        win = [raw[j] for j in range(max(0, i - 2), min(len(raw), i + 3)) if raw[j] >= 0]
        if win:
            counts[i] = int(np.median(win))
    breaks = [lo]
    i = 0
    n = len(counts)
    while i < n:
        if counts[i] < 0:
            i += 1
            continue
        j = i + 1
        while j < n and counts[j] >= 0 and abs(counts[j] - counts[i]) <= 2:
            j += 1
        if j >= n:
            break
        if counts[j] < 0:
            i = j + 1
            continue
        k = j + 1
        while k < n and counts[k] >= 0 and abs(counts[k] - counts[j]) <= 2:
            k += 1
        if (j - i) >= persist and (k - j) >= persist and abs(counts[j] - counts[i]) >= 4:
            mid = 0.5 * (float(hs[j - 1]) + float(hs[j]))
            if mid - breaks[-1] >= min_slab and hi - mid >= min_slab:
                breaks.append(mid)
        i = j
    if hi - breaks[-1] < min_slab * 0.5 and len(breaks) > 1:
        breaks[-1] = hi
    else:
        breaks.append(hi)
    return breaks


def _area_span(mesh, d, u, v, tol, ext, samples=201, min_frac=0.05):
    """Heights along d where the cross-section area is at least min_frac of its max.
    Used when a swept skin has no planar caps (cap_frac~0) so vertex AABB ends are empty tips.
    min_frac<=0 keeps every non-empty section (pointed tips still need apex closure).
    """
    hv = np.asarray(mesh.vertices) @ d
    lo, hi = float(hv.min()), float(hv.max())
    if hi - lo <= 0:
        return lo, hi
    hs = np.linspace(lo, hi, samples)
    areas = []
    for h in hs:
        try:
            sec = _section(mesh, d, u, v, float(h), tol, ext)
            areas.append(sum(abs(_area2(q)) for q in sec.loops) if sec.loops else 0.0)
        except LoftError:
            areas.append(0.0)
    areas = np.asarray(areas, float)
    amax = float(areas.max()) if len(areas) else 0.0
    if amax <= 0:
        return lo, hi
    thr = 0.0 if min_frac <= 0 else min_frac * amax
    ok = np.where(areas > thr)[0] if min_frac <= 0 else np.where(areas >= thr)[0]
    if len(ok) == 0:
        return lo, hi
    return float(hs[ok[0]]), float(hs[ok[-1]])



def _outer_section_span(mesh, d, u, v, tol, ext, lo, hi, tip_band=None):
    """Extend [lo, hi] to the outermost heights that still yield a valid cross-section.

    Pointed tips keep a few mm of micro-sections between `_area_span` and the vertex AABB;
    lofting those (then AddVertex apexes) cuts tip vertex error vs stopping at area_span alone.
    """
    hv = np.asarray(mesh.vertices) @ d
    lo_v, hi_v = float(hv.min()), float(hv.max())
    band = float(tip_band) if tip_band is not None else max(3.0 * tol, 0.02 * (hi_v - lo_v))
    a, b = float(lo), float(hi)
    for h in np.linspace(lo_v + 0.1 * tol, max(lo_v + 0.1 * tol, a), 24):
        try:
            _section(mesh, d, u, v, float(h), tol, ext)
            a = float(h)
            break
        except LoftError:
            continue
    for h in np.linspace(hi_v - 0.1 * tol, min(hi_v - 0.1 * tol, b), 24):
        try:
            _section(mesh, d, u, v, float(h), tol, ext)
            b = float(h)
            break
        except LoftError:
            continue
    # Keep a tiny tip_gap so AddVertex still fires when tips are pointed
    if a > lo_v + 0.05 * tol and b < hi_v - 0.05 * tol:
        return a, b
    return float(lo), float(hi)


def _tip_apexes(mesh, d, band_mm=None):
    """(lo_apex, hi_apex) 3D points closing pointed tips of a capless sweep along d.

    Prefer the extreme vertex along d when the tip is pointed; otherwise the centroid of
    vertices within band_mm of each AABB end. ThruSections AddVertex at these apexes
    replaces planar end caps that leave tip verts far from the solid.
    """
    V = np.asarray(mesh.vertices, float)
    hv = V @ d
    lo_i, hi_i = int(np.argmin(hv)), int(np.argmax(hv))
    lo, hi = float(hv[lo_i]), float(hv[hi_i])
    span = max(hi - lo, 1e-12)
    band = float(band_mm) if band_mm is not None else max(1.0, 0.01 * span)
    lo_pts = V[hv <= lo + band]
    hi_pts = V[hv >= hi - band]
    lo_c = lo_pts.mean(0) if len(lo_pts) else V[lo_i]
    hi_c = hi_pts.mean(0) if len(hi_pts) else V[hi_i]
    # If the band centroid sits well inside the span, the tip is pointed → use extreme vertex.
    if abs(float(lo_c @ d) - lo) > 0.25 * band:
        lo_c = V[lo_i]
    if abs(float(hi_c @ d) - hi) > 0.25 * band:
        hi_c = V[hi_i]
    return lo_c, hi_c


def _te_le_corners(Q, te_hint=None):
    """Indices (TE, LE) on closed section polyline Q for C0 loft splits.

    TE = sharpest turn near te_hint (or global max |turn|); LE = farthest vertex from TE.
    Raises LoftError if TE/LE collapse (too close on a tiny tip section).
    """
    t = np.abs(_turn(Q))
    n = len(Q)
    if te_hint is None:
        te_i = int(np.argmax(t))
    else:
        _, P = _project_on_loop(Q, te_hint)
        te_i = int(np.argmin(np.linalg.norm(Q - P, axis=1)))
        window = [(te_i + k) % n for k in range(-4, 5)]
        te_i = max(window, key=lambda j: float(t[j]))
    le_i = int(np.argmax(np.linalg.norm(Q - Q[te_i], axis=1)))
    if le_i == te_i:
        le_i = (te_i + n // 2) % n
    sep = min((le_i - te_i) % n, (te_i - le_i) % n)
    if sep < max(3, n // 12):
        raise LoftError(f"TE/LE corners collapsed (sep={sep}, n={n})")
    return np.array([te_i, le_i], int), Q[te_i]


def _turn(Q):
    e = np.roll(Q, -1, 0) - Q
    ang = np.arctan2(e[:, 1], e[:, 0])
    return (ang - np.roll(ang, 1) + np.pi) % (2 * np.pi) - np.pi


def _dedupe(Q, eps):
    keep = np.linalg.norm(Q - np.roll(Q, 1, 0), axis=1) > eps
    if keep.sum() < 3:
        return Q
    return Q[keep]


def _project_on_loop(Q, p):
    """(segment index i, point) of the closest point to p on closed polyline Q."""
    A, B = Q, np.roll(Q, -1, 0)
    u = B - A
    t = np.clip(np.einsum("ij,ij->i", p - A, u) / np.maximum(np.einsum("ij,ij->i", u, u), 1e-300), 0, 1)
    P = A + t[:, None] * u
    i = int(np.argmin(np.linalg.norm(P - p, axis=1)))
    return i, P[i]


def _resample(run, M, end_cluster=False, dense_at=None, dense_power=1.5, le_band=None,
              by_curvature=False, curv_gain=4.0, le_frac=0.33):
    """M points along open polyline `run` by arc length.

    dense_at: global power-law cluster (regressed WINGTIP L 2 — do not use on closed skins).
    le_band: (lo, hi) fraction of arc length to locally densify (e.g. (0.35, 0.65) around LE
    on a TE-seamed closed airfoil). Puts half the samples in the band, half outside — avoids
    starving TE/mid the way dense_at=0.5 did.
    by_curvature: weight arc length by local turning so concave belly / LE get more samples
    without a fixed band (section-fit diagnosis on WING 2 L: uniform resample rms~0.22 on
    lower skin vs ~0.005 upper).
    end_cluster kept for API compat but unused.
    """
    seg = np.linalg.norm(np.diff(run, axis=0), axis=1)
    if by_curvature and M > 6 and len(run) >= 4:
        # turning at vertices (open run: endpoints 0); map to segments as avg of ends
        # build a closed-ish turn estimate from the open polyline interior
        Q = run
        if np.linalg.norm(Q[0] - Q[-1]) < 1e-9:
            turn = np.abs(_turn(Q[:-1]))
            turn_s = 0.5 * (turn + np.roll(turn, -1))
            # last segment wraps — drop duplicate end
            if len(turn_s) == len(seg):
                w = seg * (1.0 + float(curv_gain) * turn_s / (turn_s.max() + 1e-12))
            else:
                w = seg * (1.0 + float(curv_gain) * turn_s[:len(seg)] / (turn_s.max() + 1e-12))
        else:
            # open: turn from triples
            turn_v = np.zeros(len(Q))
            for i in range(1, len(Q) - 1):
                a, b, c = Q[i] - Q[i - 1], Q[i + 1] - Q[i]
                na, nb = np.linalg.norm(a), np.linalg.norm(b)
                if na > 1e-12 and nb > 1e-12:
                    turn_v[i] = abs(math.atan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1]))
            turn_s = 0.5 * (turn_v[:-1] + turn_v[1:])
            w = seg * (1.0 + float(curv_gain) * turn_s / (turn_s.max() + 1e-12))
        s = np.r_[0.0, np.cumsum(w)]
    else:
        s = np.r_[0.0, np.cumsum(seg)]
    if s[-1] <= 0:
        return np.repeat(run[:1], M, 0)
    if le_band is not None and M > 6:
        lo, hi = float(le_band[0]), float(le_band[1])
        lo, hi = max(0.05, min(lo, hi)), min(0.95, max(lo, hi))
        if hi - lo < 0.05:
            t = np.linspace(0.0, s[-1], M)
        else:
            # Mild LE densify (default le_frac=0.33). Half-band (0.5) starved mid/TE
            # and regressed WINGTIP L 2 overall rms to ~0.10–0.15.
            n_le = max(3, int(round(M * float(np.clip(le_frac, 0.15, 0.45)))))
            n_out = M - n_le
            # Split outside samples proportional to arc outside the band
            left_w, right_w = lo, 1.0 - hi
            n_left = int(round(n_out * left_w / max(left_w + right_w, 1e-12)))
            n_right = n_out - n_left
            parts = []
            if n_left > 0:
                parts.append(np.linspace(0.0, lo, n_left, endpoint=False))
            parts.append(np.linspace(lo, hi, n_le, endpoint=False))
            if n_right > 0:
                parts.append(np.linspace(hi, 1.0, n_right))
            else:
                parts.append(np.array([1.0]))
            t01 = np.unique(np.concatenate(parts))
            # Ensure exactly M by re-interp if unique shortened
            if len(t01) != M:
                t01 = np.linspace(0.0, 1.0, M)
                n_le = max(3, int(round(M * float(np.clip(le_frac, 0.15, 0.45)))))
                t01 = np.r_[np.linspace(0.0, lo, M - n_le - (M - n_le)//2, endpoint=False),
                            np.linspace(lo, hi, n_le, endpoint=False),
                            np.linspace(hi, 1.0, (M - n_le)//2 + 1)]
                t01 = t01[:M]
                t01[-1] = 1.0
            t = t01 * s[-1]
    elif dense_at is not None and M > 4:
        xi = np.linspace(0.0, 1.0, M)
        a = float(np.clip(dense_at, 0.05, 0.95))
        left = xi <= a
        t01 = np.empty(M)
        t01[left] = a * (xi[left] / max(a, 1e-12)) ** dense_power
        t01[~left] = 1.0 - (1.0 - a) * ((1.0 - xi[~left]) / max(1.0 - a, 1e-12)) ** dense_power
        t = t01 * s[-1]
    else:
        t = np.linspace(0.0, s[-1], M)
    return np.c_[np.interp(t, s, run[:, 0]), np.interp(t, s, run[:, 1])]



def _resample_te_le_protect(run, M, te_frac=0.30, le_lo=0.44, le_hi=0.56, le_frac=0.14,
                            mid_floor_frac=0.50, te_w=0.14):
    """Closed TE-seamed tip resample: absolute mid/TE sample floors + mild LE densify.

    Raises M for LE without stealing TE/mid absolute counts below a uniform-96 floor:
      n_te >= te_frac * M  (split across both TE ends, width te_w each)
      n_mid >= mid_floor_frac * M
      n_le = remainder (capped by le_frac * M)
    Narrow LE band (default 0.44–0.56) so mid keeps chord coverage. Callers must still
    probe TE/mid zone rms vs baseline — prior half-band LE densify regressed overall.
    """
    seg = np.linalg.norm(np.diff(run, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(seg)]
    if s[-1] <= 0 or M < 8:
        return _resample(run, M)
    n_te = max(6, int(round(M * float(te_frac))))
    if n_te % 2:
        n_te += 1
    n_mid = max(8, int(round(M * float(mid_floor_frac))))
    n_le = M - n_te - n_mid
    n_le_cap = max(3, int(round(M * float(le_frac))))
    if n_le > n_le_cap:
        n_mid += n_le - n_le_cap
        n_le = n_le_cap
    if n_le < 3:
        n_le = 3
        n_mid = max(4, M - n_te - n_le)
    # Absolute floors vs uniform-96: keep at least those TE/mid counts when M>=96
    n_te = max(n_te, 20 if M >= 96 else n_te)
    n_mid = max(n_mid, 56 if M >= 96 else n_mid)
    if n_te + n_mid + 3 > M:
        # shrink LE first, then mid slightly
        n_le = max(3, M - n_te - n_mid)
        if n_le + n_te + n_mid > M:
            n_mid = max(4, M - n_te - n_le)
    else:
        n_le = M - n_te - n_mid
    n_te_each = n_te // 2
    parts = [
        np.linspace(0.0, te_w, n_te_each, endpoint=False),
        np.linspace(te_w, le_lo, max(2, n_mid // 2), endpoint=False),
        np.linspace(le_lo, le_hi, n_le, endpoint=False),
        np.linspace(le_hi, 1.0 - te_w, max(2, n_mid - n_mid // 2), endpoint=False),
        np.linspace(1.0 - te_w, 1.0, n_te_each),
    ]
    t01 = np.concatenate(parts)
    t01 = np.unique(np.clip(t01, 0.0, 1.0))
    if len(t01) != M:
        t01 = np.interp(np.linspace(0, 1, M), np.linspace(0, 1, len(t01)), t01)
        t01[0], t01[-1] = 0.0, 1.0
    tt = t01 * s[-1]
    return np.c_[np.interp(tt, s, run[:, 0]), np.interp(tt, s, run[:, 1])]


def _densify_closed(Q, M):
    """Keep every native loop vertex; insert midpoints on longest edges until len==M.

    Uniform arc-length resample moves samples off faceted CAD verts and cuts corners
    (WING 2 L section-fit: poly160 belly rms~0.22, poly_nat~0.09). Densify-only keeps
    natives on the loft wire so per-section 2D fit → 0 when M>=n_nat; long belly edges
    get the extra samples. If n_nat>M, fall back to arc resample (caller should raise M).
    """
    P = np.asarray(Q, dtype=float)
    if M < 3:
        M = 3
    if len(P) < 3:
        return _resample(np.r_[P, P[:1]], M + 1)[:-1]
    if len(P) == M:
        return P.copy()
    if len(P) > M:
        return _resample(np.r_[P, P[:1]], M + 1)[:-1]
    pts = P.tolist()
    while len(pts) < M:
        arr = np.asarray(pts, dtype=float)
        dlen = np.linalg.norm(np.roll(arr, -1, axis=0) - arr, axis=1)
        i = int(np.argmax(dlen))
        mid = 0.5 * (arr[i] + arr[(i + 1) % len(arr)])
        pts.insert(i + 1, mid.tolist())
    return np.asarray(pts, dtype=float)



def _align_closed(run, prev):
    """Cyclic shift (and optional reverse) of closed `run` minimizing MSD to `prev`.

    Keeps ThruSections from twisting when the projected start point drifts on a taper.
    """
    n = len(run)
    if n == 0 or len(prev) != n:
        return run
    best = (np.inf, run)
    for orient in (run, run[::-1]):
        # cumulative cost via roll: try every phase
        for s in range(n):
            r = np.roll(orient, s, axis=0)
            cost = float(np.mean(np.sum((r - prev) ** 2, axis=1)))
            if cost < best[0]:
                best = (cost, r)
    return best[1]


def _align_open(run, prev):
    """Open corner-to-corner run: leave as-is (endpoints already fixed by corner order).

    Must NOT cyclic-shift or reverse a single run — reversing one run breaks shared
    corner vertices with its neighbours and MakeWire fails ('section wire is not connected').
    Whole-loop orientation is fixed by `_follow` corner matching.
    """
    return run


class _Section:
    """One plane section: loops in the (u, v) frame, oriented counter-clockwise, with corners."""

    def __init__(self, h, loops, depth):
        self.h, self.loops, self.depth = h, loops, depth
        self.corners = []           # per loop: vertex indices of the matched corners (cyclic order)
        self.runs = []              # per loop: list of resampled runs (or one closed run)


def _section(mesh, d, u, v, h, tol, ext):
    from trimesh.intersections import mesh_plane
    segs = mesh_plane(mesh, d, d * h)
    if len(segs) == 0:
        raise LoftError(f"empty cross-section at h={h:.4f}")
    segs = np.asarray(segs)
    # Mesh-plane chains are sensitive to vertex-key rounding: try a small ladder so a single
    # dangling endpoint (common on aileron/hinge cuts) does not kill the whole loft.
    base = max(1e-7 * ext, 1e-9)
    ladders = [base, 1e-9, 1e-8 * ext, 1e-6 * ext, 1e-5 * ext, 1e-4 * ext, 0.5 * tol, tol]
    loops, bad = [], 0
    seen = set()
    for kt in ladders:
        kt = float(kt)
        if kt <= 0 or kt in seen:
            continue
        seen.add(kt)
        loops, bad = _loops(segs, key_tol=kt)
        if not bad and loops:
            break
    if bad or not loops:
        raise LoftError(f"open cross-section at h={h:.4f} ({bad} dangling points)")
    L2 = []
    for lp in loops:
        Q = np.c_[(lp - d * h) @ u, (lp - d * h) @ v]
        Q = _dedupe(Q, 1e-6 * ext)
        if _area2(Q) < 0:
            Q = Q[::-1]
        Q = simplify_loop(Q, 1e-3 * tol)
        if len(Q) >= 3 and abs(_area2(Q)) > (0.5 * tol) ** 2:
            L2.append(Q)
    if L2:
        areas = [abs(_area2(q)) for q in L2]
        amax = max(areas)
        L2 = [q for q, a in zip(L2, areas) if a > 0.02 * amax]
    depth = [sum(_inside(L2[i][0], L2[j]) for j in range(len(L2)) if j != i) for i in range(len(L2))]
    return _Section(h, L2, depth)


def _match_loops(ref, sec, tol):
    """Reorder sec's loops to correspond to ref's (centroid distance, same containment depth)."""
    if len(sec.loops) != len(ref.loops):
        # Hinge cutouts / spar tubes appear mid-span (aileron). Keep the loops that match
        # ref by depth+centroid; drop extras. Fewer than ref → hard fail (hole vanished).
        if len(sec.loops) > len(ref.loops):
            cr = [q.mean(0) for q in ref.loops]
            cs = [q.mean(0) for q in sec.loops]
            used, keep = set(), []
            for i, c in enumerate(cr):
                js = sorted((j for j in range(len(cs)) if j not in used
                             and sec.depth[j] % 2 == ref.depth[i] % 2),
                            key=lambda j: np.linalg.norm(cs[j] - c))
                if not js:
                    raise LoftError(
                        f"cross-section at h={sec.h:.4f} has {len(sec.loops)} loops, "
                        f"{len(ref.loops)} at h={ref.h:.4f}")
                used.add(js[0]); keep.append(js[0])
            sec.loops = [sec.loops[j] for j in keep]
            sec.depth = [sec.depth[j] for j in keep]
        else:
            raise LoftError(
                f"cross-section at h={sec.h:.4f} has {len(sec.loops)} loops, "
                f"{len(ref.loops)} at h={ref.h:.4f}")
    cr = [q.mean(0) for q in ref.loops]
    cs = [q.mean(0) for q in sec.loops]
    used, order = set(), []
    for i, c in enumerate(cr):
        js = sorted((j for j in range(len(cs)) if j not in used and sec.depth[j] % 2 == ref.depth[i] % 2),
                    key=lambda j: np.linalg.norm(cs[j] - c))
        if not js:
            raise LoftError(f"loop {i} has no counterpart at h={sec.h:.4f}")
        a0, a1 = abs(_area2(ref.loops[i])), abs(_area2(sec.loops[js[0]]))
        # Allow taper along the sweep (wing tips, cones); reject only pathological collapses.
        if not (0.005 < a1 / max(a0, 1e-300) < 50.0):
            raise LoftError(f"loop {i} area changes {a0:.4g} -> {a1:.4g} between h={ref.h:.4f} and h={sec.h:.4f}")
        used.add(js[0])
        order.append(js[0])
    sec.loops = [sec.loops[j] for j in order]
    sec.depth = [sec.depth[j] for j in order]


def _corners_like(Q, prev_pts, prev_sign, th):
    """Corner indices of Q matching the corners of the neighbouring section (positions prev_pts, turn
    signs prev_sign) in the same cyclic order; hysteresis: strong corners > th must all be matched,
    a matched corner may turn down to th/2."""
    t = _turn(Q)
    m = len(prev_pts)
    strong = np.where(np.abs(t) > th)[0]
    if m == 0:
        # Corner-free neighbour: polyline tessellation turns are not geometric corners.
        return np.zeros(0, int)
    if len(strong) > m:
        # Faceted CAD can sprout extra tessellation spikes; keep the m strongest turns.
        strong = np.array(sorted(strong, key=lambda i: -abs(float(t[i])))[:m], int)
        strong.sort()
    cand = list(strong)
    if len(cand) < m:   # hysteresis: a corner may soften to th/2 (tessellation)
        weak = [i for i in np.argsort(-np.abs(t)) if 0.5 * th < abs(t[i]) <= th]
        if len(weak) < m - len(cand):
            # Allow a slab-local drop in tessellation corners: keep strongest available
            need = m - len(cand)
            extra = [int(i) for i in np.argsort(-np.abs(t)) if i not in cand][:need]
            cand += extra
            if len(cand) < m:
                raise LoftError(f"{m} corners expected, {len(cand)} found")
        else:
            cand += [int(i) for i in weak[:m - len(cand)]]
    cand = sorted(int(i) for i in cand)
    # cyclic order is kept; prefer sign-consistent rotation, else best positional match
    # (tessellation can flip a soft corner's turn sign across neighbouring sections).
    best = None
    best_any = None
    for r in range(m):
        pick = [cand[(k + r) % m] for k in range(m)]
        cost = float(sum(np.linalg.norm(Q[i] - p) for i, p in zip(pick, prev_pts)))
        if best_any is None or cost < best_any[0]:
            best_any = (cost, pick)
        if any(np.sign(t[i]) != sg for i, sg in zip(pick, prev_sign)):
            continue
        if best is None or cost < best[0]:
            best = (cost, pick)
    if best is None:
        best = best_any
    if best is None:
        raise LoftError("corner turning signs do not match the neighbouring section")
    pick = best[1]
    return np.array(pick, int)



def _stable_corners(Q, th, n_keep=None, min_sep=None):
    """Corner indices on Q: turns above th, optionally exactly the n_keep strongest.

    min_sep: drop a weaker corner within this vertex-count of a stronger one so open
    corner-to-corner runs stay long enough for MakeWire (faceted CAD clusters turns).
    """
    t = _turn(Q)
    strong = np.where(np.abs(t) > th)[0]
    n = len(Q)
    sep = int(min_sep) if min_sep is not None else max(2, n // 40)

    def _thin(idxs, keep):
        # Greedy keep strongest first, reject neighbours within sep along the ring.
        order = sorted(idxs, key=lambda i: -abs(float(t[i])))
        chosen = []
        for i in order:
            if any(min((i - j) % n, (j - i) % n) < sep for j in chosen):
                continue
            chosen.append(int(i))
            if keep is not None and len(chosen) >= int(keep):
                break
        return chosen

    if n_keep is None:
        return np.array(sorted(_thin(list(strong), None)), int)
    pick = _thin(list(strong), int(n_keep))
    if len(pick) < int(n_keep):
        order = list(np.argsort(-np.abs(t)))
        for i in order:
            if any(min((int(i) - j) % n, (j - int(i)) % n) < sep for j in pick):
                continue
            if int(i) in pick:
                continue
            pick.append(int(i))
            if len(pick) >= int(n_keep):
                break
    return np.array(sorted(pick[:int(n_keep)]), int)


def _slab_corner_count(mesh, d, u, v, a, b, tol, ext, th_deg=35.0, samples=7):
    """Median sharp-corner count of the outer loop across a slab (stable vs a single mid hit)."""
    th = math.radians(th_deg)
    eps = max(tol, 1e-3 * (b - a))
    hs = np.linspace(a + eps, b - eps, int(samples))
    ns = []
    for h in hs:
        try:
            sec = _section(mesh, d, u, v, float(h), tol, ext)
            if not sec.loops:
                continue
            Q = max(sec.loops, key=lambda q: abs(_area2(q)))
            ns.append(int(np.sum(np.abs(_turn(Q)) > th)))
        except LoftError:
            continue
    if not ns:
        return 0
    return int(np.median(ns))


def _label_corners(ref, th, n_keep=None):
    ref.corners = []
    for Q in ref.loops:
        ref.corners.append(_stable_corners(Q, th, n_keep=n_keep))


def _follow(prev, sec, th):
    """Corners and closed-loop start of sec, following the neighbouring (already labelled) prev."""
    sec.corners, sec.start = [], []
    for li, Q in enumerate(sec.loops):
        cp = prev.corners[li]
        Qp = prev.loops[li]
        if len(cp):
            tp = _turn(Qp)
            c = _corners_like(Q, Qp[cp], np.sign(tp[cp]), th)
            sec.corners.append(c)
            sec.start.append(None)
        else:
            _corners_like(Q, np.zeros((0, 2)), [], th)
            sec.corners.append(np.zeros(0, int))
            sec.start.append(None)


def _closed_start(Q, p):
    """Q rotated to start at the point nearest p (inserted exactly on the polyline)."""
    i, P = _project_on_loop(Q, p)
    Q2 = np.r_[P[None], Q[i + 1:], Q[:i + 1]]
    return _dedupe(Q2, 1e-12)


def _runs(sec, li, Ms, start_pt=None, native_closed=False, by_curvature=False,
         densify_closed=False, le_local=False):
    """Resampled corner-to-corner runs of loop li (closed loop: one run from the start point).
    native_closed: keep the section vertices (faceted CAD) instead of arc-length resampling.
    by_curvature: curvature-weighted arc resample (closed smooth skins with concave belly).
    densify_closed: keep natives and only split long edges to reach M (preferred for smooth skins).
    le_local: mild LE-band densify on TE-seamed closed tip skins with absolute mid/TE
    sample floors. Guard with zone rms floors — dense_at=0.5 / insert densify regressed.
    """
    Q = sec.loops[li]
    c = sec.corners[li]
    if len(c) == 0:
        Qs = _closed_start(Q, start_pt)
        if native_closed:
            return [Qs]
        if densify_closed:
            return [_densify_closed(Qs, int(Ms[0]))]
        run = np.r_[Qs, Qs[:1]]
        if le_local:
            # TE-seamed closed tip: protect TE ends + mild LE densify so mid/TE absolute
            # sample counts stay ≥ uniform-96 while LE gets extra (M=160).
            return [_resample_te_le_protect(run, Ms[0] + 1)[:-1]]
        return [_resample(run, Ms[0] + 1, by_curvature=by_curvature)[:-1]]
    out = []
    n = len(Q)
    for k in range(len(c)):
        a, b = int(c[k]), int(c[(k + 1) % len(c)])
        idx = [(a + i) % n for i in range(((b - a) % n or n) + 1)]
        out.append(_resample(Q[idx], Ms[k]))
    return out


def _counts(ref, li, spacing):
    Q = ref.loops[li]
    c = ref.corners[li]
    n = len(Q)
    L = np.linalg.norm(np.roll(Q, -1, 0) - Q, axis=1)
    if len(c) == 0:
        return [int(np.clip(math.ceil(L.sum() / spacing), 12, 240))]
    Ms = []
    for k in range(len(c)):
        a, b = int(c[k]), int(c[(k + 1) % len(c)])
        ln = sum(L[(a + i) % n] for i in range((b - a) % n or n))
        nv = (b - a) % n or n
        Ms.append(int(np.clip(max(math.ceil(ln / spacing), nv + 1), 2, 160)))
    return Ms


def _loft_slab(mesh, d, u, v, a, b, tol, ext, corner_deg, max_sections, t_stop, smooth=False,
              tip_apexes=None):
    """Lofted solid between heights a < b; returns (shape, info).

    smooth=True: every loop is corner-free (faceted CAD skins / continuous airfoils).
    tip_apexes: optional (lo_pt, hi_pt) 3D points for ThruSections AddVertex tip closure
    when the skin has no planar caps (pointed wingtips / spindles).
    """
    th = math.radians(max(abs(corner_deg), 1.0))
    eps = min(1e-3 * (b - a), max(0.05 * tol, 1e-4 * (b - a)))
    hm = 0.5 * (a + b)
    ref = _section(mesh, d, u, v, hm, tol, ext)
    if not ref.loops:
        raise LoftError(f"no loops at h={hm:.4f}")
    if smooth:
        ref.corners = [np.zeros(0, int) for _ in ref.loops]
    else:
        n_keep = _slab_corner_count(mesh, d, u, v, a, b, tol, ext, th_deg=max(abs(corner_deg), 1.0))
        if n_keep < 2:
            # Fall back to smooth if the slab cannot sustain corners
            ref.corners = [np.zeros(0, int) for _ in ref.loops]
            smooth = True
        else:
            _label_corners(ref, th, n_keep=n_keep)
    # start points of closed (corner-free) loops. Pointed tip skins: lock the seam at the
    # sharp trailing-edge crease so ThruSections keeps a matched TE generator across
    # sections (C0 crease alignment). Otherwise: farthest along the loop major axis.
    ref.start = []
    te_hint = None
    for li, Q in enumerate(ref.loops):
        if len(ref.corners[li]):
            ref.start.append(None)
            continue
        if tip_apexes is not None and smooth:
            try:
                _c, te_pt = _te_le_corners(Q, te_hint=te_hint)
                te_hint = te_pt
                ref.start.append(te_pt)
                continue
            except LoftError:
                pass
        C = Q - Q.mean(0)
        w = np.linalg.eigh(C.T @ C)[1][:, -1]
        ref.start.append(Q[int(np.argmax(C @ w))])
    edge = np.median(np.concatenate([np.linalg.norm(np.roll(Q, -1, 0) - Q, axis=1) for Q in ref.loops]))
    spacing = max(0.5 * float(edge), 2 * tol)
    Ms = [_counts(ref, li, spacing) for li in range(len(ref.loops))]
    # Pointed tip skins: slightly higher M so mild LE densify does not starve mid/TE.
    # Tip: M=160 with le_frac=0.25 → ~40 LE + ~120 mid/TE (mid/TE ≥ prior uniform-96).
    m_cap = 160 if (smooth and tip_apexes is not None) else 96
    if smooth:
        Ms = [[min(m, m_cap) for m in row] for row in Ms]
    else:
        # Cornered multi-slab: keep edge counts modest — dense corner runs make
        # BRepAlgoAPI_Fuse hang minutes on shared planar caps (WING 2 L).
        Ms = [[int(np.clip(m, 2, 24)) for m in row] for row in Ms]
    secs = {_hk(hm): ref}

    def get(h, nb):
        k = _hk(h)
        if k in secs:
            return secs[k]
        s = _section(mesh, d, u, v, float(k), tol, ext)
        _match_loops(nb, s, tol)
        _follow(nb, s, th)
        s.start = []
        for li in range(len(s.loops)):
            if len(s.corners[li]):
                s.start.append(None)
            elif tip_apexes is not None and smooth and nb.start[li] is not None:
                # Keep the TE seam: re-detect TE near the neighbour's start point
                try:
                    _c, te_pt = _te_le_corners(s.loops[li], te_hint=nb.start[li])
                    s.start.append(te_pt)
                except LoftError:
                    s.start.append(_project_on_loop(s.loops[li], nb.start[li])[1])
            else:
                s.start.append(_project_on_loop(s.loops[li], nb.start[li])[1])
        secs[k] = s
        return s

    def pts(s):
        # ThruSections needs equal edge counts across sections. native_closed kept unequal
        # CAD vertex counts on tapers and produced shells / underfilled solids; resample.
        return [_runs(s, li, Ms[li], s.start[li], native_closed=False,
                       le_local=(tip_apexes is not None and smooth))
                for li in range(len(s.loops))]

    # initial sections: centre outwards so every section follows a close neighbour.
    # Prismatic: few samples. Pointed/tapered: denser Chebyshev seed (cluster near tips).
    # Cornered multi-slab: seed denser than 5 — corner follow drops many stations.
    pointed = tip_apexes is not None
    if smooth or pointed:
        n0 = int(np.clip(max_sections, 2, min(40, max_sections)))
    else:
        n0 = int(np.clip(max(8, max_sections // 2), 4, min(24, max_sections)))
    if (smooth or pointed) and n0 > 2:
        t = np.linspace(0.0, 1.0, n0)
        t = 0.5 - 0.5 * np.cos(np.pi * t)  # denser near a,b
        hs0 = list(a + eps + t * ((b - eps) - (a + eps)))
    else:
        hs0 = list(np.linspace(a + eps, b - eps, n0))
    hs_sorted = sorted(hs0, key=lambda x: abs(x - hm))
    for h in hs_sorted:
        if time.time() > t_stop:
            raise LoftError("section loft over its time budget")
        nb = secs[min(secs, key=lambda x: abs(x - _hk(h)))]
        try:
            get(h, nb)
        except LoftError:
            continue
    hs = sorted(secs)
    # Cornered slab that could not keep a consistent corner set across the span: fall back
    # to smooth (TE-seamed) rather than a 1-section prism that overfills tapered wings.
    if (not smooth) and len(hs) < 2 and tip_apexes is None:
        return _loft_slab(mesh, d, u, v, a, b, tol, ext, corner_deg, max_sections, t_stop,
                         smooth=True, tip_apexes=None)

    def pts_aligned(h_list):
        """Resample every section to Ms, then align each loop/run to the previous section."""
        out = {}
        prev_runs = None
        for h in h_list:
            raw = [_runs(secs[h], li, Ms[li], secs[h].start[li], native_closed=False,
                         le_local=(tip_apexes is not None and smooth))
                   for li in range(len(secs[h].loops))]
            if prev_runs is not None:
                aligned = []
                for li, runs in enumerate(raw):
                    # Closed (0-corner) loops: cyclic align. Cornered open runs: reverse-only.
                    if len(secs[h].corners[li]) == 0:
                        aligned.append([_align_closed(runs[k], prev_runs[li][k]) for k in range(len(runs))])
                    else:
                        aligned.append([_align_open(runs[k], prev_runs[li][k]) for k in range(len(runs))])
                raw = aligned
            out[h] = raw
            prev_runs = raw
        return out

    cache = pts_aligned(hs)
    # refine: insert mid-sections where linear interpolation misses. Safe with equal-count
    # polylines (native unequal counts + refine previously ballooned ThruSections).
    changed = True
    while changed and len(hs) < max_sections:
        if time.time() > t_stop:
            raise LoftError("section loft over its time budget")
        changed = False
        new = []
        for h0, h1 in zip(hs[:-1], hs[1:]):
            if len(hs) + len(new) >= max_sections:
                break
            hmid = 0.5 * (h0 + h1)
            try:
                s = get(hmid, secs[h0])
            except LoftError:
                continue
            P = pts(s)
            P = [[(_align_closed if len(secs[h0].corners[li]) == 0 else _align_open)(
                    P[li][k], cache[h0][li][k]) for k in range(len(P[li]))]
                 for li in range(len(P))]
            cache[_hk(hmid)] = P
            err = 0.0
            for li in range(len(P)):
                for r0, r1, rm in zip(cache[h0][li], cache[h1][li], P[li]):
                    err = max(err, float(np.linalg.norm(0.5 * (r0 + r1) - rm, axis=1).max()))
            if err > (0.15 * tol if (smooth or pointed) else 0.25 * tol):
                new.append(_hk(hmid))
            else:
                del secs[_hk(hmid)]
                del cache[_hk(hmid)]
        if new:
            hs = sorted(hs + new)
            changed = True
    hs = sorted(secs)
    cache = pts_aligned(hs)
    info = {"from": round(a, 4), "to": round(b, 4), "sections": len(hs), "loops": len(ref.loops),
            "corners": [int(len(c)) for c in ref.corners], "edges_per_loop": [len(m) for m in Ms]}
    # straight runs (every section) become lines
    straight = []
    for li in range(len(ref.loops)):
        st = []
        for k in range(len(Ms[li])):
            ok = len(ref.corners[li]) > 0
            for h in hs:
                r = cache[h][li][k]
                c = r[-1] - r[0]
                nc = np.linalg.norm(c)
                if nc <= 0 or np.abs((r - r[0]) @ np.array([-c[1], c[0]]) / nc).max() > 0.1 * tol:
                    ok = False
                    break
            st.append(ok)
        straight.append(st)
    info["lines"] = int(sum(map(sum, straight)))
    # Near-prismatic slab: section areas and extents match -> true prism (avoids ThruSections twist).
    # A single surviving section is never prismatic (would extrude a mid hit across a taper).
    href = hs[len(hs) // 2]
    prismatic = len(hs) >= 2
    for h in hs:
        for li in range(len(ref.loops)):
            a0 = abs(_area2(secs[href].loops[li]))
            a1 = abs(_area2(secs[h].loops[li]))
            if not (0.98 < a1 / max(a0, 1e-300) < 1.02):
                prismatic = False
                break
            e0 = np.ptp(secs[href].loops[li], axis=0)
            e1 = np.ptp(secs[h].loops[li], axis=0)
            if float(np.linalg.norm(e0 - e1)) > 2 * tol:
                prismatic = False
                break
        if not prismatic:
            break
    info["prismatic"] = bool(prismatic)
    solids = []
    if prismatic:
        from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
        from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
        from OCP.gp import gp_Vec
        for li in range(len(ref.loops)):
            # Native section vertices for the prism face (faithful to faceted CAD).
            native = _runs(secs[href], li, Ms[li], secs[href].start[li], native_closed=True)
            w = _wire3(native, straight[li], d * float(a), u, v,
                       closed=len(ref.corners[li]) == 0, tol=tol)
            face = BRepBuilderAPI_MakeFace(w, True).Face()
            pr = BRepPrimAPI_MakePrism(face, gp_Vec(*(float(x) * (b - a) for x in d)))
            if not pr.IsDone():
                raise LoftError("prism failed")
            solids.append((ref.depth[li], pr.Shape()))
    else:
        for li in range(len(ref.loops)):
            wires = []
            for j, h in enumerate(hs):
                # With tip apexes, keep wires at true section heights; apexes close the tips.
                if tip_apexes is None:
                    hh = a if j == 0 else (b if j == len(hs) - 1 else float(h))
                else:
                    hh = float(h)
                wires.append(_wire3(cache[h][li], straight[li], d * hh, u, v,
                                    closed=len(ref.corners[li]) == 0, tol=tol))
            # Ruled-vs-smooth pick target: full mesh volume when this slab covers most of
            # the sweep (capless skins / single slab); otherwise trapezoid of section areas.
            full = float(np.ptp(np.asarray(mesh.vertices) @ d))
            if (b - a) > 0.85 * full:
                expect = float(mesh.volume)
            else:
                areas = [abs(_area2(secs[h].loops[li])) for h in hs]
                expect = float(np.trapezoid(areas, x=[float(h) for h in hs]))
            if time.time() > t_stop:
                raise LoftError("section loft over its time budget")
            apex = tip_apexes if (tip_apexes is not None and ref.depth[li] % 2 == 0) else None
            solids.append((ref.depth[li], _thru(wires, tol, target_vol=expect, tip_apexes=apex)))
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
    outer = [s for dp, s in solids if dp % 2 == 0]
    inner = [s for dp, s in solids if dp % 2 == 1]
    shape = outer[0]
    for s in outer[1:]:
        shape = _bop(BRepAlgoAPI_Fuse, shape, s, ext)
    for s in inner:
        shape = _bop(BRepAlgoAPI_Cut, shape, s, ext)
    return shape, info


def _ensure_solid(shape):
    """If `shape` is (or contains) a closed shell and no solid, wrap shell(s) as solid(s)."""
    from OCP.BRep import BRep_Builder
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeSolid
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.TopAbs import TopAbs_SHELL, TopAbs_SOLID
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS, TopoDS_Compound

    def _count(sh, kind):
        ex = TopExp_Explorer(sh, kind)
        n = 0
        while ex.More():
            n += 1
            ex.Next()
        return n

    if _count(shape, TopAbs_SOLID) >= 1:
        return shape
    shells = []
    ex = TopExp_Explorer(shape, TopAbs_SHELL)
    while ex.More():
        shells.append(TopoDS.Shell(ex.Current()))
        ex.Next()
    if not shells:
        return shape
    solids = []
    for sh in shells:
        ms = BRepBuilderAPI_MakeSolid()
        ms.Add(sh)
        if not ms.IsDone():
            continue
        sol = ms.Solid()
        if BRepCheck_Analyzer(sol).IsValid():
            solids.append(sol)
    if not solids:
        return shape
    if len(solids) == 1:
        return solids[0]
    comp = TopoDS_Compound()
    BRep_Builder().MakeCompound(comp)
    for s in solids:
        BRep_Builder().Add(comp, s)
    return comp


def _bop(cls, s1, s2, ext):
    # Shared planar end caps at slab breaks. Keep this simple: GlueFull/large fuzzy has
    # hung OCC for minutes on WING 2 L cornered (and even smooth) multi-slab fuses.
    # One small fuzzy + optional subprocess timeout is applied by `_fuse_slabs`.
    last = None
    for fz in (1e-7 * ext, 1e-6 * ext, 5e-6 * ext):
        op = cls(s1, s2)
        op.SetFuzzyValue(float(fz))
        op.Build()
        if op.IsDone():
            return _ensure_solid(op.Shape())
        last = f"{cls.__name__} failed (fuzzy={fz:.3g})"
    raise LoftError(last or f"{cls.__name__} failed")


def _brep_write(sh, path):
    from OCP.BRepTools import BRepTools
    if not BRepTools.Write_s(sh, path):
        raise LoftError(f"failed to write temp BREP {path}")


def _brep_read(path):
    from OCP.BRep import BRep_Builder
    from OCP.BRepTools import BRepTools
    from OCP.TopoDS import TopoDS_Shape
    sh = TopoDS_Shape()
    if not BRepTools.Read_s(sh, path, BRep_Builder()):
        raise LoftError(f"failed to read temp BREP {path}")
    return sh



def _thru_shell(wires, tol):
    """Open shell through closed section wires (no end caps). Smooth then ruled."""
    from OCP.BRepOffsetAPI import BRepOffsetAPI_ThruSections
    from OCP.BRepCheck import BRepCheck_Analyzer
    last = None
    for ruled in (False, True):
        ts = BRepOffsetAPI_ThruSections(False, ruled, max(1e-7, 0.1 * tol))
        ts.CheckCompatibility(False)
        for w in wires:
            ts.AddWire(w)
        ts.Build()
        if not ts.IsDone():
            last = f"shell ThruSections failed (ruled={ruled})"
            continue
        sh = ts.Shape()
        if BRepCheck_Analyzer(sh).IsValid():
            return sh
        last = f"shell ThruSections invalid (ruled={ruled})"
    raise LoftError(last or "shell ThruSections failed")



def _section_xyz(mesh, d, u, v, h, tol, ext):
    """Outer-loop section points in 3D (and the 2D loop)."""
    sec = _section(mesh, d, u, v, float(h), tol, ext)
    if not sec.loops:
        raise LoftError(f"empty cross-section at h={h:.4f}")
    # Prefer outermost loop (depth even, largest area)
    li = max(range(len(sec.loops)), key=lambda i: abs(_area2(sec.loops[i])))
    Q = sec.loops[li]
    P = np.asarray(d, float) * float(h) + Q[:, 0:1] * u + Q[:, 1:2] * v
    return Q, P


def _track_guides(mesh, d, u, v, lo, hi, tol, ext, th_deg=20.0, samples=61):
    """Spanwise guide curves: TE/LE + soft corners + belly/top extrema.

    Mesh dihedral creases on faceted CAD skins are mostly chordwise (ringing the
    section); spanwise feature tracks come from matching section corners / extrema.
    Soft guides (min-y belly, max-y crown) pin concave mid-chord lower skin that has
    almost no turning and is missed by TE/LE-only lofts.
    """
    hs = np.linspace(float(lo), float(hi), int(np.clip(samples, 21, 121)))
    # Inset empty outer AABB hits
    eps = min(1e-3 * (hi - lo), max(0.05 * tol, 1e-4 * (hi - lo)))
    hs = np.array([lo + eps if i == 0 else (hi - eps if i == len(hs) - 1 else h)
                   for i, h in enumerate(hs)], float)
    hm = float(hs[len(hs) // 2])
    Q0, P0 = _section_xyz(mesh, d, u, v, hm, tol, ext)
    th = math.radians(th_deg)
    seed = []
    try:
        te_le, _ = _te_le_corners(Q0)
        seed.extend(int(x) for x in te_le)
    except LoftError:
        pass
    soft = _stable_corners(Q0, th, n_keep=None)
    for ci in list(map(int, soft)) + [int(np.argmin(P0[:, 1])), int(np.argmax(P0[:, 1]))]:
        if all(min((ci - s) % len(Q0), (s - ci) % len(Q0)) > max(3, len(Q0) // 40) for s in seed):
            seed.append(ci)
    if len(seed) < 2:
        raise LoftError("strip loft: fewer than 2 guide seeds")
    tracks = [[] for _ in seed]
    for h in hs:
        try:
            Q, P = _section_xyz(mesh, d, u, v, float(h), tol, ext)
        except LoftError:
            continue
        if not tracks[0]:
            for ti, ci in enumerate(seed):
                # nearest to mid seed position
                ii = int(np.argmin(np.linalg.norm(P - P0[ci], axis=1)))
                tracks[ti].append(P[ii].copy())
            continue
        prev = np.array([t[-1] for t in tracks])
        used = set()
        for ti, ppt in enumerate(prev):
            d2 = np.linalg.norm(P - ppt, axis=1)
            for i in np.argsort(d2):
                i = int(i)
                if i not in used:
                    tracks[ti].append(P[i].copy())
                    used.add(i)
                    break
    tracks = [np.asarray(t, float) for t in tracks if len(t) >= 8]
    if len(tracks) < 2:
        raise LoftError("strip loft: guide tracking collapsed")
    # Drop tracks that jumped too far between stations (unstable match)
    stable = []
    for t in tracks:
        step = np.linalg.norm(np.diff(t, axis=0), axis=1)
        if float(np.median(step)) < 0.15 * float(np.linalg.norm(mesh.extents)):
            stable.append(t)
    if len(stable) < 2:
        stable = tracks
    # Angular order in the section (u,v) frame at mid — stable for non-convex airfoils
    # where atan2(y,z) can scramble TE/belly/LE/top.
    cent_uv = Q0.mean(0)
    angs = []
    for t in stable:
        p = t[len(t) // 2]
        # project p into section uv at mid height
        # p ≈ d*hm + u*qu + v*qv  →  qu = (p-d*hm)·u etc. Use mid hm from first sample h.
        hm_t = float(t[len(t) // 2] @ np.asarray(d, float))  # wrong if d not unit... d is unit
        # better: nearest mid-section point
        ii = int(np.argmin(np.linalg.norm(P0 - p, axis=1)))
        qu, qv = Q0[ii]
        angs.append(math.atan2(qv - cent_uv[1], qu - cent_uv[0]))
    order = list(np.argsort(angs))
    return [stable[i] for i in order], {
        "n_guides": len(stable),
        "guide_y": [round(float(stable[i][:, 1].mean()), 3) for i in order],
        "guide_z": [round(float(stable[i][:, 2].mean()), 3) for i in order],
    }


def _resample_rail(pts, n):
    d = pts[1:] - pts[:-1]
    seg = np.linalg.norm(d, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    if cum[-1] < 1e-12:
        return np.repeat(pts[:1], n, axis=0)
    t = np.linspace(0.0, cum[-1], n)
    out = []
    for tv in t:
        k = int(np.searchsorted(cum, tv))
        if k <= 0:
            out.append(pts[0])
        elif k >= len(cum):
            out.append(pts[-1])
        else:
            a = (tv - cum[k - 1]) / max(cum[k] - cum[k - 1], 1e-12)
            out.append((1 - a) * pts[k - 1] + a * pts[k])
    return np.asarray(out, float)


def _short_arc(P, a_pt, b_pt, n):
    """Polyline arc a→b on section loop P choosing the geometrically shorter way."""
    ia = int(np.argmin(np.linalg.norm(P - a_pt, axis=1)))
    ib = int(np.argmin(np.linalg.norm(P - b_pt, axis=1)))
    nQ = len(P)
    fwd = [(ia + k) % nQ for k in range(((ib - ia) % nQ) + 1)]
    bak = [(ia - k) % nQ for k in range(((ia - ib) % nQ) + 1)]

    def plen(idx):
        pts = P[np.asarray(idx)]
        if len(pts) < 2:
            return 0.0
        return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())

    arc = fwd if plen(fwd) <= plen(bak) else bak
    arcP = P[np.asarray(arc)]
    if len(arcP) < 2:
        return np.linspace(a_pt, b_pt, n)
    return _resample_rail(np.vstack([a_pt, arcP, b_pt]), n)


def _open_poly_wire(pts, tol):
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire
    from OCP.gp import gp_Pnt
    mw = BRepBuilderAPI_MakeWire()
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        if np.linalg.norm(a - b) < 1e-9 * max(1.0, float(np.linalg.norm(a) + 1.0)):
            continue
        mw.Add(BRepBuilderAPI_MakeEdge(
            gp_Pnt(float(a[0]), float(a[1]), float(a[2])),
            gp_Pnt(float(b[0]), float(b[1]), float(b[2]))).Edge())
        if not mw.IsDone():
            raise LoftError("strip open wire not connected")
    return mw.Wire()


def _fit_strip_surface(grid, tol, ext):
    """(nu, nv, 3) point grid → B-spline surface with bbox guard (impeller-style)."""
    from OCP.collections import Array2_gp_Pnt
    from OCP.GeomAPI import GeomAPI_PointsToBSplineSurface
    from OCP.GeomAbs import GeomAbs_C2, GeomAbs_C1
    from OCP.gp import gp_Pnt
    nu, nv, _ = grid.shape
    arr = Array2_gp_Pnt(1, nu, 1, nv)
    for i in range(nu):
        for j in range(nv):
            p = grid[i, j]
            arr.SetValue(i + 1, j + 1, gp_Pnt(float(p[0]), float(p[1]), float(p[2])))
    lo, hi = grid.reshape(-1, 3).min(0), grid.reshape(-1, 3).max(0)
    margin = max(0.5, 0.05 * float(np.linalg.norm(hi - lo)))
    last = None
    for dmin, dmax, cont, tf in (
            (3, 8, GeomAbs_C2, 1.0), (3, 5, GeomAbs_C2, 2.0), (3, 3, GeomAbs_C1, 5.0)):
        sf = GeomAPI_PointsToBSplineSurface(arr, dmin, dmax, cont, max(tol, 1e-4) * tf).Surface()
        last = sf
        u1, u2, v1, v2 = sf.Bounds()
        Q = np.array([[sf.Value(uu, vv).X(), sf.Value(uu, vv).Y(), sf.Value(uu, vv).Z()]
                      for uu in np.linspace(u1, u2, max(6, 2 * nu))
                      for vv in np.linspace(v1, v2, max(6, 2 * nv))])
        if np.all(Q.min(0) > lo - margin) and np.all(Q.max(0) < hi + margin):
            return sf
    return last


def _strip_loft_skin(mesh, d, u, v, lo, hi, tol, ext, max_sections, t_stop,
                     th_deg=20.0):
    """Generic strip loft: spanwise guides → one B-spline face per sector → sew + caps.

    Guides = TE/LE + soft section corners + belly/top extrema (mesh dihedral creases on
    faceted CAD are mostly chordwise; spanwise tracks come from section matching).
    Soft belly guide targets concave lower mid-chord (y≈12–18) which has almost no turn.
    Corner-count changes become guides that drop out; sectors re-close angularly.
    """
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeSolid, BRepBuilderAPI_Sewing
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    from OCP.ShapeFix import ShapeFix_Shell, ShapeFix_Solid, ShapeFix_Shape
    from OCP.TopAbs import TopAbs_FACE, TopAbs_SHELL, TopAbs_SOLID
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    if time.time() > t_stop:
        raise LoftError("section loft over its time budget")
    guides, ginfo = _track_guides(mesh, d, u, v, lo, hi, tol, ext, th_deg=th_deg,
                                  samples=int(np.clip(max_sections * 2, 41, 101)))
    nspan = int(np.clip(max_sections, 24, 96))
    n_across = 14
    rails = [_resample_rail(g, nspan) for g in guides]
    mesh_area = float(mesh.area)
    faces = []
    strip_info = []
    for i in range(len(rails)):
        if time.time() > t_stop:
            raise LoftError("section loft over its time budget")
        A, B = rails[i], rails[(i + 1) % len(rails)]
        rows = []
        for j in range(nspan):
            h = float(A[j] @ d)
            try:
                _, P = _section_xyz(mesh, d, u, v, h, tol, ext)
                pts = _short_arc(P, A[j], B[j], n=n_across)
            except LoftError:
                pts = np.linspace(A[j], B[j], n_across)
            pts[0], pts[-1] = A[j], B[j]
            rows.append(pts)
        grid = np.transpose(np.asarray(rows, float), (1, 0, 2))  # (n_across, nspan, 3)
        sf = _fit_strip_surface(grid, tol, ext)
        face = BRepBuilderAPI_MakeFace(sf, 1e-6).Face()
        if not BRepCheck_Analyzer(face).IsValid():
            raise LoftError(f"strip {i} face invalid")
        p = GProp_GProps()
        BRepGProp.SurfaceProperties_s(face, p)
        area = float(p.Mass())
        if not math.isfinite(area) or area > 20.0 * mesh_area:
            raise LoftError(f"strip {i} ballooned area={area:.3g}")
        faces.append(face)
        strip_info.append({"strip": i, "area": round(area, 2)})

    sew = BRepBuilderAPI_Sewing(max(10 * tol, 1e-3 * ext))
    for f in faces:
        sew.Add(f)
    # End caps: walk short arcs between consecutive rail endpoints on the end section
    # so cap boundary matches strip free ends (full section polyline often misses rails).
    for end_i in (0, -1):
        # Rails already sit on inset end stations; build cap in that same plane.
        P_ref = []
        # rebuild section at this rail station
        h_end = float(rails[0][end_i] @ d)
        _, P = _section_xyz(mesh, d, u, v, h_end, tol, ext)
        ring = []
        for i in range(len(rails)):
            a = rails[i][end_i]
            b = rails[(i + 1) % len(rails)][end_i]
            arc = _short_arc(P, a, b, n=12)
            arc[0], arc[-1] = a, b
            if i == 0:
                ring.extend(list(arc))
            else:
                ring.extend(list(arc[1:]))
        ring.append(ring[0])
        w = _open_poly_wire(np.asarray(ring, float), tol)
        sew.Add(BRepBuilderAPI_MakeFace(w, True).Face())
    sew.Perform()
    sewn = sew.SewedShape()
    ex = TopExp_Explorer(sewn, TopAbs_SHELL)
    shells_out = []
    while ex.More():
        shells_out.append(TopoDS.Shell(ex.Current()))
        ex.Next()
    if not shells_out:
        raise LoftError("strip sew produced no shell")

    def nfaces(sh):
        e = TopExp_Explorer(sh, TopAbs_FACE)
        n = 0
        while e.More():
            n += 1
            e.Next()
        return n

    shell = max(shells_out, key=nfaces)
    fx = ShapeFix_Shell(shell)
    fx.Perform()
    shell = fx.Shell()
    solid = None
    for rev in (False, True):
        s = shell.Reversed() if rev else shell
        ms = BRepBuilderAPI_MakeSolid()
        ms.Add(TopoDS.Shell(s))
        if not ms.IsDone():
            continue
        sol = ms.Solid()
        sf = ShapeFix_Solid(sol)
        sf.Perform()
        sol = sf.Solid()
        fs = ShapeFix_Shape(sol)
        fs.SetPrecision(tol)
        fs.SetMaxTolerance(max(20 * tol, 2e-3 * ext))
        fs.Perform()
        sol = fs.Shape()
        p = GProp_GProps()
        BRepGProp.VolumeProperties_s(sol, p)
        vol = float(p.Mass())
        if vol > 0 and BRepCheck_Analyzer(sol).IsValid():
            e = TopExp_Explorer(sol, TopAbs_SOLID)
            n = 0
            while e.More():
                n += 1
                e.Next()
            if n == 1 and 0.5 < vol / max(mesh.volume, 1e-12) < 1.5:
                solid = sol
                break
    if solid is None:
        raise LoftError(f"strip sew MakeSolid failed (free={sew.NbFreeEdges()})")
    info = {
        "from": round(float(lo), 4), "to": round(float(hi), 4),
        "sections": nspan, "corners": [ginfo["n_guides"]],
        "smooth": False, "corner_deg": float(th_deg),
        "shared_wire": False, "strip_loft": True,
        "n_strips": len(faces), "guides": ginfo,
        "strips": strip_info, "sew_free": int(sew.NbFreeEdges()),
    }
    return solid, [info]


def _sew_slabs_shared_wire(mesh, d, u, v, lv, tol, ext, max_sections, t_stop,
                            th_deg=35.0):
    """Join multi-slab loft without BOP: shared interface wires + shell ThruSections + sew.

    Per-slab corner intents (e.g. 10/6/2 on WING 2 L). Break wires use
    n_break = max(n_left, n_right) (union); slabs upgrade to match end-break counts so
    ThruSections edge counts stay consistent. One TopoDS wire object is reused at each
    break so Sewing closes. Modes tried in order:
      1) cornered with upgraded n_use (when follow stays stable / shells sane)
      2) TE/LE 2-corner shared wires (stable pins; beats smooth mid-skin on WING 2 L)
      3) smooth TE-seamed shared wires (2l baseline)
    Planar faces close the outer ends.
    """
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeSolid, BRepBuilderAPI_Sewing
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    from OCP.ShapeFix import ShapeFix_Shell, ShapeFix_Solid, ShapeFix_Shape
    from OCP.TopAbs import TopAbs_FACE, TopAbs_SHELL, TopAbs_SOLID
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    lv = [float(x) for x in lv]
    th = math.radians(th_deg)
    n_slabs = len(lv) - 1
    intents = []
    for a, b in zip(lv[:-1], lv[1:]):
        n_c = _slab_corner_count(mesh, d, u, v, a, b, tol, ext, th_deg=th_deg)
        intents.append(max(2, int(n_c)) if n_c >= 2 else 0)
    n_use = list(intents)
    for _ in range(n_slabs + 2):
        breaks_n = [max(n_use[i], n_use[i + 1]) for i in range(n_slabs - 1)]
        new_use = []
        for i in range(n_slabs):
            n = n_use[i]
            if i > 0:
                n = max(n, breaks_n[i - 1])
            if i < n_slabs - 1:
                n = max(n, breaks_n[i])
            new_use.append(n)
        if new_use == n_use:
            break
        n_use = new_use
    breaks_n = [max(n_use[i], n_use[i + 1]) for i in range(n_slabs - 1)] if n_slabs > 1 else []

    eps_h = min(1e-3 * (lv[-1] - lv[0]), max(0.05 * tol, 1e-4 * (lv[-1] - lv[0])))

    def sample_h(h):
        h = float(h)
        if h <= lv[0] + 1e-12:
            return lv[0] + eps_h
        if h >= lv[-1] - 1e-12:
            return lv[-1] - eps_h
        return h

    def nfaces(sh):
        e = TopExp_Explorer(sh, TopAbs_FACE)
        n = 0
        while e.More():
            n += 1
            e.Next()
        return n

    def shell_area(sh):
        p = GProp_GProps()
        BRepGProp.SurfaceProperties_s(sh, p)
        return float(p.Mass())

    def make_solid_from_shells(shells, wires_at, ref, mode_tag, infos):
        # area sanity — ballooned ThruSections still pass BRepCheck
        mesh_area = float(mesh.area)
        for i, sh in enumerate(shells):
            a = shell_area(sh)
            if not math.isfinite(a) or a > 50.0 * mesh_area:
                raise LoftError(f"shared-wire shell {i} ballooned area={a:.3g}")
        sew = BRepBuilderAPI_Sewing(max(tol, 1e-4 * ext))
        for sh in shells:
            sew.Add(sh)
        for li in range(len(ref.loops)):
            if ref.depth[li] % 2 != 0:
                continue
            for h in (lv[0], lv[-1]):
                sew.Add(BRepBuilderAPI_MakeFace(wires_at[_hk(h)][li], True).Face())
        sew.Perform()
        sewn = sew.SewedShape()
        ex = TopExp_Explorer(sewn, TopAbs_SHELL)
        shells_out = []
        while ex.More():
            shells_out.append(TopoDS.Shell(ex.Current()))
            ex.Next()
        if not shells_out:
            raise LoftError("shared-wire sew produced no shell")
        shell = max(shells_out, key=nfaces)
        fx = ShapeFix_Shell(shell)
        fx.Perform()
        shell = fx.Shell()
        solid = None
        for rev in (False, True):
            s = shell.Reversed() if rev else shell
            ms = BRepBuilderAPI_MakeSolid()
            ms.Add(TopoDS.Shell(s))
            if not ms.IsDone():
                continue
            sol = ms.Solid()
            sf = ShapeFix_Solid(sol)
            sf.Perform()
            sol = sf.Solid()
            fs = ShapeFix_Shape(sol)
            fs.SetPrecision(tol)
            fs.SetMaxTolerance(max(10 * tol, 1e-3 * ext))
            fs.Perform()
            sol = fs.Shape()
            p = GProp_GProps()
            BRepGProp.VolumeProperties_s(sol, p)
            vol = float(p.Mass())
            if vol > 0 and BRepCheck_Analyzer(sol).IsValid():
                e = TopExp_Explorer(sol, TopAbs_SOLID)
                n = 0
                while e.More():
                    n += 1
                    e.Next()
                if n == 1:
                    # reject absurd volumes
                    if abs(vol / max(mesh.volume, 1e-12)) > 5.0:
                        continue
                    solid = sol
                    break
        if solid is None:
            raise LoftError(f"shared-wire sew MakeSolid failed ({mode_tag})")
        for info in infos:
            info["shared_wire_mode"] = mode_tag
        return solid, infos

    def run_mode(mode):
        """mode: 'smooth' | 'te_le' | 'cornered' (uses upgraded n_use globally)."""
        if time.time() > t_stop:
            raise LoftError("section loft over its time budget")
        hm = 0.5 * (lv[0] + lv[-1])
        ref = _section(mesh, d, u, v, sample_h(hm), tol, ext)
        if not ref.loops:
            raise LoftError("no loops at mid-span for shared-wire sew")
        closed = (mode == "smooth")
        n_keep = 0
        if mode == "te_le":
            n_keep = 2
            closed = False
        elif mode == "cornered":
            n_keep = max(n_use) if n_use else 0
            if n_keep < 2:
                raise LoftError("cornered mode needs n_use>=2")
            closed = False

        if mode == "smooth":
            ref.corners = [np.zeros(0, int) for _ in ref.loops]
            te_hint = None
            ref.start = []
            for Q in ref.loops:
                try:
                    _c, te_pt = _te_le_corners(Q, te_hint=te_hint)
                    te_hint = te_pt
                    ref.start.append(te_pt)
                except LoftError:
                    C = Q - Q.mean(0)
                    w = np.linalg.eigh(C.T @ C)[1][:, -1]
                    ref.start.append(Q[int(np.argmax(C @ w))])
        elif mode == "te_le":
            ref.corners, ref.start = [], []
            for Q in ref.loops:
                c, _te = _te_le_corners(Q)
                ref.corners.append(c)
                ref.start.append(None)
        else:
            _label_corners(ref, th, n_keep=n_keep)
            ref.start = [None] * len(ref.loops)

        edge = np.median(np.concatenate(
            [np.linalg.norm(np.roll(Q, -1, 0) - Q, axis=1) for Q in ref.loops]))
        # Section-fit (WING 2 L 2o): dominant error is chordal undersample of closed
        # sections (poly160 belly~0.22; span@dh<=5~0; sweep‖pca1 OK). Densify-only
        # zeros section fit but breaks spanwise homology → ThruSections balloon.
        # Fix: uniform arc at high M (poly400 belly~0.06) so generators stay matched.
        spacing = max(0.2 * float(edge), 1.5 * tol)
        # M=320: poly belly~0.07 (was 0.22@160); M>=400×46 sect risk OCC hang.
        cap = 320 if closed else 160
        n_max = [len(Q) for Q in ref.loops]
        if closed:
            for h_probe in np.linspace(lv[0], lv[-1], 9):
                try:
                    sp = _section(mesh, d, u, v, sample_h(float(h_probe)), tol, ext)
                    for li, Q in enumerate(sp.loops):
                        if li < len(n_max):
                            n_max[li] = max(n_max[li], len(Q))
                except LoftError:
                    pass
        Ms = []
        for li in range(len(ref.loops)):
            raw = _counts(ref, li, spacing)
            n_nat = n_max[li] if li < len(n_max) else len(ref.loops[li])
            if closed:
                Ms.append([int(np.clip(max(m, n_nat, 320), 12, cap)) for m in raw])
            else:
                raw2 = [int(np.clip(m, 4, cap)) for m in raw]
                if sum(raw2) < n_nat and sum(raw2) > 0:
                    scale = n_nat / sum(raw2)
                    raw2 = [int(np.clip(math.ceil(m * scale), 4, cap)) for m in raw2]
                Ms.append(raw2)
        straight = [[False] * len(Ms[li]) for li in range(len(ref.loops))]

        def section_at(h, nb):
            s = _section(mesh, d, u, v, float(sample_h(h)), tol, ext)
            _match_loops(nb, s, tol)
            if mode == "smooth":
                s.corners = [np.zeros(0, int) for _ in s.loops]
                s.start = []
                for li in range(len(s.loops)):
                    try:
                        _c, te_pt = _te_le_corners(s.loops[li], te_hint=nb.start[li])
                        s.start.append(te_pt)
                    except LoftError:
                        s.start.append(_project_on_loop(s.loops[li], nb.start[li])[1])
            elif mode == "te_le":
                s.corners, s.start = [], []
                for li, Q in enumerate(s.loops):
                    te_hint = nb.loops[li][nb.corners[li][0]]
                    c, _te = _te_le_corners(Q, te_hint=te_hint)
                    s.corners.append(c)
                    s.start.append(None)
            else:
                try:
                    _follow(nb, s, th)
                    s.start = [None] * len(s.loops)
                except LoftError:
                    _label_corners(s, th, n_keep=n_keep)
                    s.start = [None] * len(s.loops)
                if any(len(c) != n_keep for c in s.corners):
                    _label_corners(s, th, n_keep=n_keep)
                    s.start = [None] * len(s.loops)
                # reject unstable corner jumps vs neighbour
                for li in range(len(s.loops)):
                    prev_pts = nb.loops[li][nb.corners[li]]
                    cur_pts = s.loops[li][s.corners[li]]
                    # cyclic best shift
                    best = np.inf
                    m = len(prev_pts)
                    for shft in range(m):
                        dlt = np.linalg.norm(cur_pts - np.roll(prev_pts, shft, axis=0), axis=1).max()
                        best = min(best, float(dlt))
                    if best > 0.25 * ext:
                        raise LoftError(f"corner jump {best:.3g} at h={h}")
            return s

        def wires_for(h_place, sec):
            # Uniform arc (homology for ThruSections). densify_closed balloons smooth loft.
            raw = [_runs(sec, li, Ms[li], sec.start[li] if closed else None,
                         native_closed=False, by_curvature=False, densify_closed=False)
                   for li in range(len(sec.loops))]
            ref_runs = [_runs(ref, li, Ms[li], ref.start[li] if closed else None,
                              native_closed=False, by_curvature=False, densify_closed=False)
                        for li in range(len(ref.loops))]
            aligned = []
            for li, runs in enumerate(raw):
                if closed:
                    aligned.append([_align_closed(runs[k], ref_runs[li][k]) for k in range(len(runs))])
                else:
                    aligned.append([_align_open(runs[k], ref_runs[li][k]) for k in range(len(runs))])
            return [_wire3(aligned[li], straight[li], d * float(h_place), u, v,
                           closed=closed, tol=tol) for li in range(len(ref.loops))]

        secs = {_hk(sample_h(hm)): ref}
        wires_at = {}
        for h in lv:
            if time.time() > t_stop:
                raise LoftError("section loft over its time budget")
            hs = sample_h(h)
            if _hk(hs) not in secs:
                nb = secs[min(secs, key=lambda x: abs(x - _hk(hs)))]
                secs[_hk(hs)] = section_at(hs, nb)
            wires_at[_hk(h)] = wires_for(h, secs[_hk(hs)])

        shells = []
        infos = []
        span = lv[-1] - lv[0]
        for si, (a, b) in enumerate(zip(lv[:-1], lv[1:])):
            if time.time() > t_stop:
                raise LoftError("section loft over its time budget")
            frac = (b - a) / max(span, 1e-12)
            n_int = int(np.clip(round(max_sections * max(frac, 0.15)), 3, max_sections))
            t = np.linspace(0.0, 1.0, n_int + 2)[1:-1]
            t = 0.5 - 0.5 * np.cos(np.pi * t)
            hs_int = [a + (b - a) * float(x) for x in t]
            slab_wires = [[wires_at[_hk(a)][li]] for li in range(len(ref.loops))]
            for h in hs_int:
                if time.time() > t_stop:
                    raise LoftError("section loft over its time budget")
                if _hk(h) not in secs:
                    nb = secs[min(secs, key=lambda x: abs(x - _hk(h)))]
                    try:
                        secs[_hk(h)] = section_at(h, nb)
                    except LoftError:
                        continue
                ww = wires_for(h, secs[_hk(h)])
                for li in range(len(ref.loops)):
                    slab_wires[li].append(ww[li])
            for li in range(len(ref.loops)):
                slab_wires[li].append(wires_at[_hk(b)][li])
            for li in range(len(ref.loops)):
                if ref.depth[li] % 2 != 0:
                    continue
                if len(slab_wires[li]) < 2:
                    raise LoftError(f"shared-wire slab {a:.4f}->{b:.4f} has <2 sections")
                shells.append(_thru_shell(slab_wires[li], tol))
            used_c = 0 if mode == "smooth" else (2 if mode == "te_le" else int(n_keep))
            infos.append({
                "from": round(a, 4), "to": round(b, 4), "sections": len(slab_wires[0]),
                "corners": [used_c], "smooth": bool(closed),
                "corner_deg": float(0.0 if closed else th_deg),
                "shared_wire": True,
                "corner_count_intent": int(intents[si]),
                "corner_count_used": int(used_c if mode != "cornered" else n_use[si]),
            })
        return make_solid_from_shells(shells, wires_at, ref, mode, infos)

    # Dense smooth first: WING 2 L diagnosis showed section-resample (M≤96) dominates
    # belly error; te_le pins help less than keeping native-density closed sections.
    # TE/LE and modest cornered remain as fallbacks.
    modes = ["smooth"]
    if n_use and max(n_use) >= 2:
        modes.append("te_le")
        if max(n_use) <= 4:
            modes.append("cornered")

    errors = []
    for mode in modes:
        try:
            return run_mode(mode)
        except LoftError as ex:
            errors.append(f"{mode}:{ex}")
            if time.time() > t_stop:
                break
            continue
    raise LoftError("shared-wire sew failed: " + " | ".join(errors)[:240])



def _fuse_pair_worker(path_a, path_b, path_out, ext_, q):
    """Child-process entry: fuse two BREPs to path_out (picklable args only)."""
    try:
        from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse
        from meshsolid.loft import _bop, _brep_read, _brep_write
        out = _bop(BRepAlgoAPI_Fuse, _brep_read(path_a), _brep_read(path_b), ext_)
        _brep_write(out, path_out)
        q.put(("ok", path_out))
    except Exception as ex:
        q.put(("err", str(ex)))


def _fuse_slabs(shapes, ext, timeout_s=12.0):
    """Fuse slab solids; each BOP runs in a child with a hard wall-clock timeout.

    Cornered ThruSections slabs hang inside BRepAlgoAPI_Fuse (minutes+). Shapes are
    passed via temp BREP files because TopoDS handles are not picklable.
    """
    import multiprocessing as mp
    import tempfile

    if not shapes:
        raise LoftError("no slabs to fuse")
    if len(shapes) == 1:
        return _ensure_solid(shapes[0])

    with tempfile.TemporaryDirectory(prefix="loft_fuse_") as td:
        solid = shapes[0]
        for i, sh in enumerate(shapes[1:]):
            pa, pb, po = f"{td}/a.brep", f"{td}/b.brep", f"{td}/out_{i}.brep"
            _brep_write(solid, pa)
            _brep_write(sh, pb)
            q = mp.Queue()
            proc = mp.Process(target=_fuse_pair_worker, args=(pa, pb, po, float(ext), q))
            proc.start()
            proc.join(timeout_s)
            if proc.is_alive():
                proc.terminate()
                proc.join(2.0)
                if proc.is_alive():
                    proc.kill()
                raise LoftError(f"slab fuse exceeded {timeout_s:.0f}s (OCC hang)")
            if q.empty():
                raise LoftError("slab fuse produced no result")
            status, payload = q.get()
            if status != "ok":
                raise LoftError(payload)
            solid = _brep_read(payload)
        solid = _ensure_solid(solid)
        # Reject "successful" BOPs that leave crumbs (WING 2 L smooth multi-slab → 25 solids).
        from OCP.TopAbs import TopAbs_SOLID
        from OCP.TopExp import TopExp_Explorer
        from OCP.BRepGProp import BRepGProp
        from OCP.GProp import GProp_GProps
        from OCP.BRepCheck import BRepCheck_Analyzer
        ex = TopExp_Explorer(solid, TopAbs_SOLID)
        n = 0
        while ex.More():
            n += 1
            ex.Next()
        p = GProp_GProps()
        BRepGProp.VolumeProperties_s(solid, p)
        vol = float(p.Mass())
        if n != 1 or vol <= 0 or not BRepCheck_Analyzer(solid).IsValid():
            raise LoftError(f"slab fuse produced n_solids={n} vol={vol:.4g} valid={BRepCheck_Analyzer(solid).IsValid()}")
        return solid


def _wire3(runs, straight, base, u, v, closed, tol):
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire
    from OCP.GeomAPI import GeomAPI_Interpolate
    from OCP.collections import HArray1_double, HArray1_gp_Pnt
    from OCP.gp import gp_Pnt

    def p3(q):
        x = base + q[0] * u + q[1] * v
        return gp_Pnt(float(x[0]), float(x[1]), float(x[2]))

    mw = BRepBuilderAPI_MakeWire()
    for k, r in enumerate(runs):
        if straight[k]:
            e = BRepBuilderAPI_MakeEdge(p3(r[0]), p3(r[-1])).Edge()
        else:
            n = len(r)
            # Polyline through resampled points (faithful to faceted CAD). B-spline
            # interpolate shrinks concave airfoils / balloons under refine — including open TE runs.
            idxs = range(n) if closed else range(n - 1)
            for i in idxs:
                j = (i + 1) % n if closed else i + 1
                e = BRepBuilderAPI_MakeEdge(p3(r[i]), p3(r[j])).Edge()
                mw.Add(e)
                if not mw.IsDone():
                    raise LoftError("section wire is not connected")
            continue
        mw.Add(e)
        if not mw.IsDone():
            raise LoftError("section wire is not connected")
    return mw.Wire()


def _thru(wires, tol, target_vol=None, tip_apexes=None):
    """Loft closed section wires to a solid. Tries smooth then ruled.

    When target_vol is set, pick the valid solid whose volume is closest to it
    (smooth loft can balloon polyline sections; ruled can underfill tapers).
    tip_apexes: optional (lo_pt, hi_pt) added via AddVertex before/after wires so pointed
    tips are closed instead of planar end caps (which leave tip mesh verts far from the solid).
    """
    from OCP.BRepOffsetAPI import BRepOffsetAPI_ThruSections
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    from OCP.TopAbs import TopAbs_SOLID
    from OCP.TopExp import TopExp_Explorer
    from OCP.gp import gp_Pnt
    last = None
    cands = []
    for ruled in (False, True):
        ts = BRepOffsetAPI_ThruSections(True, ruled, max(1e-7, 0.1 * tol))
        ts.CheckCompatibility(False)
        if not ruled:
            ts.SetSmoothing(False)
        if tip_apexes is not None:
            lo_pt, hi_pt = tip_apexes
            ts.AddVertex(BRepBuilderAPI_MakeVertex(
                gp_Pnt(float(lo_pt[0]), float(lo_pt[1]), float(lo_pt[2]))).Vertex())
        for w in wires:
            ts.AddWire(w)
        if tip_apexes is not None:
            lo_pt, hi_pt = tip_apexes
            ts.AddVertex(BRepBuilderAPI_MakeVertex(
                gp_Pnt(float(hi_pt[0]), float(hi_pt[1]), float(hi_pt[2]))).Vertex())
        ts.Build()
        if not ts.IsDone():
            last = f"ThruSections failed (ruled={ruled})"
            continue
        sh = ts.Shape()
        ns = 0
        ex = TopExp_Explorer(sh, TopAbs_SOLID)
        while ex.More():
            ns += 1
            ex.Next()
        if ns != 1 or not BRepCheck_Analyzer(sh).IsValid():
            last = f"ThruSections produced non-solid/invalid (ruled={ruled})"
            continue
        gp = GProp_GProps()
        BRepGProp.VolumeProperties_s(sh, gp)
        vol = float(gp.Mass())
        if target_vol is None:
            return sh
        cands.append((abs(vol - abs(target_vol)), sh, vol, ruled))
    if cands:
        # Prefer a smooth loft when its volume is within 2% of target (ruled wins on raw
        # |dV| alone but leaves larger chordal skin error between sections). Fall back to
        # closest volume when no candidate is inside the band.
        if target_vol is not None:
            ok = [c for c in cands if abs(c[2] - abs(target_vol)) <= 0.02 * abs(target_vol)]
            pool = ok or cands
            pool.sort(key=lambda x: (x[3], x[0]))  # prefer smooth (ruled=False), then |dV|
            return pool[0][1]
        cands.sort(key=lambda x: x[0])
        return cands[0][1]
    raise LoftError(last or "ThruSections failed")


def build_loft(mesh, tol, min_frac=0.9, corner_deg=35.0, max_slabs=12, max_sections=33, time_budget=120.0,
               self_check=True, accuracy=None):
    t0 = time.time()
    t_stop = t0 + time_budget
    res = LoftResult(mesh_volume=float(mesh.volume))
    d, frac, capf = sweep_axis(mesh)
    res.sweep_area_frac = round(frac, 5)
    if d is None or frac < min_frac:
        res.reason = f"not a swept skin: cap/skin area {100 * frac:.1f}% < {100 * min_frac:.1f}%"
        return res
    res.sweep_dir = np.round(d, 6).tolist()
    ext = float(np.linalg.norm(mesh.extents))
    u = np.cross(d, [1.0, 0, 0] if abs(d[0]) < 0.9 else [0, 1.0, 0])
    u /= np.linalg.norm(u)
    v = np.cross(d, u)
    lv = _cap_levels(mesh, d, merge=max(2 * tol, 1e-3 * ext), min_area=1e-4 * float(mesh.area))
    tip_apexes = None
    # Skins without planar caps: trim to non-empty sections; close pointed tips with apexes
    # only when AABB tips extend past the section span (planar-capped slabs must not get apexes).
    if capf < 0.05:
        lo, hi = _area_span(mesh, d, u, v, tol, ext, min_frac=0.0)
        if hi - lo > 2 * tol:
            lv = [lo, hi]
        hv = np.asarray(mesh.vertices) @ d
        tip_gap = max(float(lv[0]) - float(hv.min()), float(hv.max()) - float(lv[-1]))
        # Pointed tip only when end-section area collapses vs mid (ailerons have AABB gap
        # but near-constant section area — apexes there balloon ThruSections volume).
        pointed = False
        if tip_gap > tol and (hi - lo) > 4 * tol:
            try:
                a_lo = abs(_area2(max(_section(mesh, d, u, v, float(lv[0]) + tol, tol, ext).loops,
                                      key=lambda q: abs(_area2(q)))))
                a_hi = abs(_area2(max(_section(mesh, d, u, v, float(lv[-1]) - tol, tol, ext).loops,
                                      key=lambda q: abs(_area2(q)))))
                a_mid = abs(_area2(max(_section(mesh, d, u, v, 0.5 * (lv[0] + lv[-1]), tol, ext).loops,
                                       key=lambda q: abs(_area2(q)))))
                # Both ends must collapse (spindle/wingtip). One thin end (aileron) is not pointed.
                pointed = a_mid > 0 and (a_lo / a_mid < 0.15) and (a_hi / a_mid < 0.15)
            except LoftError:
                pointed = False
        if pointed:
            tip_apexes = _tip_apexes(mesh, d)
            # Degenerate tip section wires: push the loft span to the outermost valid
            # micro-sections near the AABB, then close with AddVertex apexes.
            lo2, hi2 = _outer_section_span(mesh, d, u, v, tol, ext, lv[0], lv[-1])
            if hi2 - lo2 > 2 * tol:
                lv = [lo2, hi2]
            # Pointed tips need dense span samples; ruled/smooth chord error dominates otherwise.
            max_sections = max(int(max_sections), 80)
    # Auto slab breaks where spanwise sharp-corner count changes (wings with spars/hinges).
    if tip_apexes is None and len(lv) >= 2:
        br = _corner_count_breaks(mesh, d, u, v, tol, ext, lv[0], lv[-1],
                                  th_deg=35.0 if corner_deg <= 0 else abs(corner_deg))
        merge_h = max(2 * tol, 1e-3 * ext)
        merged = sorted({float(x) for x in list(lv) + list(br)})
        out = [merged[0]]
        for x in merged[1:]:
            if x - out[-1] > merge_h:
                out.append(x)
        out[0], out[-1] = float(lv[0]), float(lv[-1])
        lv = out
    res.levels = [round(x, 4) for x in lv]
    if len(lv) - 1 > max_slabs:
        res.reason = f"{len(lv) - 1} loft slabs (limit {max_slabs})"
        return res
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
    from OCP.TopAbs import TopAbs_FACE, TopAbs_SOLID
    from OCP.TopExp import TopExp_Explorer
    solid = None
    try:
        # Multi-slab without tip apexes: shared-wire shell sew (no BOP). Falls back to a
        # single full-span loft if sew fails. Single-slab / tip-apex path unchanged.
        if tip_apexes is None and len(lv) > 2:
            strip_err = None
            try:
                solid, infos = _strip_loft_skin(
                    mesh, d, u, v, float(lv[0]), float(lv[-1]), tol, ext,
                    max_sections, t_stop,
                    th_deg=20.0 if corner_deg <= 0 else min(abs(corner_deg), 25.0))
                res.slabs = infos
            except LoftError as ex:
                strip_err = str(ex)[:120]
                solid = None
            if solid is None:
                try:
                    solid, infos = _sew_slabs_shared_wire(
                        mesh, d, u, v, lv, tol, ext, max(int(max_sections), 64), t_stop,
                        th_deg=35.0 if corner_deg <= 0 else abs(corner_deg))
                    if strip_err:
                        for inf in infos:
                            inf["strip_fallback"] = strip_err
                    res.slabs = infos
                except LoftError as sew_ex:
                    if time.time() > t_stop:
                        raise LoftError("section loft over its time budget")
                    sh, info = _loft_slab(
                        mesh, d, u, v, float(lv[0]), float(lv[-1]), tol, ext,
                        corner_deg if corner_deg > 0 else 35.0, max_sections, t_stop,
                        smooth=(corner_deg <= 0), tip_apexes=None)
                    info = dict(info)
                    info["smooth"] = bool(corner_deg <= 0)
                    info["corner_deg"] = float(0.0 if corner_deg <= 0 else abs(corner_deg))
                    info["sew_fallback"] = str(sew_ex)[:160]
                    if strip_err:
                        info["strip_fallback"] = strip_err
                    res.slabs = [info]
                    solid = sh
        else:
            span = float(lv[-1] - lv[0])
            slab_shapes = []
            for a, b in zip(lv[:-1], lv[1:]):
                slab_smooth = (corner_deg <= 0)
                slab_corner = corner_deg if corner_deg > 0 else 35.0
                if time.time() > t_stop:
                    raise LoftError("section loft over its time budget")
                frac = (float(b) - float(a)) / max(span, 1e-12)
                slab_ms = int(np.clip(round(max_sections * max(frac, 0.15)), 4, max_sections))
                sh, info = _loft_slab(mesh, d, u, v, a, b, tol, ext, slab_corner, slab_ms, t_stop,
                                     smooth=slab_smooth, tip_apexes=tip_apexes)
                info = dict(info)
                info["smooth"] = bool(slab_smooth)
                info["corner_deg"] = float(0.0 if slab_smooth else slab_corner)
                res.slabs.append(info)
                slab_shapes.append(sh)
            solid = _fuse_slabs(slab_shapes, ext, timeout_s=min(45.0, max(5.0, t_stop - time.time())))
    except LoftError as ex:
        res.reason = str(ex)
        res.seconds = time.time() - t0
        return res
    if solid is not None:
        solid = _ensure_solid(solid)
    if time.time() > t_stop:
        # Fuse finished but unify/self_check would overrun: still expose solid count/volume.
        res.shape = solid
        from OCP.TopAbs import TopAbs_SOLID as _S
        from OCP.TopExp import TopExp_Explorer as _E
        from OCP.BRepGProp import BRepGProp as _GP
        from OCP.GProp import GProp_GProps as _G
        ex = _E(solid, _S); n = 0
        while ex.More():
            n += 1; ex.Next()
        res.n_solids = n
        p = _G(); _GP.VolumeProperties_s(solid, p); res.volume = float(p.Mass())
        res.brepcheck_valid = bool(BRepCheck_Analyzer(solid).IsValid())
        if n == 1 and res.volume > 0 and abs(res.volume - res.mesh_volume) <= 0.02 * abs(res.mesh_volume) + 1e-9:
            res.ok = True
            res.reason = "section loft over its time budget (before unify); solid kept"
        else:
            res.reason = f"section loft over its time budget (before unify); n_solids={n}"
        res.seconds = time.time() - t0
        return res
    un = ShapeUpgrade_UnifySameDomain(solid, True, True, True)
    un.Build()
    solid = un.Shape()
    if time.time() > t_stop:
        res.reason = "section loft over its time budget (before shapefix)"
        res.shape = solid
        res.seconds = time.time() - t0
        return res
    from OCP.ShapeFix import ShapeFix_Shape
    fx = ShapeFix_Shape(solid)
    fx.SetPrecision(tol)
    fx.SetMaxTolerance(max(10 * tol, 1e-3 * ext))
    fx.Perform()
    solid = fx.Shape()
    # UnifySameDomain / ShapeFix can demote a fused multi-slab solid to a closed shell
    # (still has volume via BRepGProp). Promote any free closed shells back to solids so
    # the single-solid gate sees the fused result.
    solid = _ensure_solid(solid)
    res.brepcheck_valid = bool(BRepCheck_Analyzer(solid).IsValid())
    for kind, attr in ((TopAbs_SOLID, "n_solids"), (TopAbs_FACE, "n_faces")):
        ex = TopExp_Explorer(solid, kind)
        while ex.More():
            setattr(res, attr, getattr(res, attr) + 1)
            ex.Next()
    p = GProp_GProps()
    BRepGProp.VolumeProperties_s(solid, p)
    res.volume = float(p.Mass())
    res.shape = solid
    if not res.brepcheck_valid:
        res.reason = "BRepCheck_Analyzer reports invalid shape"
    elif res.n_solids != 1:
        res.reason = f"{res.n_solids} solids"
    elif res.volume <= 0:
        res.reason = "non-positive volume"
    elif abs(res.volume - res.mesh_volume) > 0.02 * abs(res.mesh_volume) + 1e-9:
        res.reason = f"volume {res.volume:.4g} differs from mesh {res.mesh_volume:.4g} by more than 2%"
    else:
        res.ok = True
    if res.ok and self_check:
        # quick pre-gate: a loft that misses the mesh must not displace the analytic builders.
        # Faceted CAD (exports): vertices lie on the true surface, so vertex->solid stays at
        # 1.5/3*tol. Face interiors sit a chord sag off that surface (facets.sag ~ w*theta/8), so
        # solid->mesh is limited by 3*tol + sag_p90 — same basis as pipeline.accept (accept_p95 +
        # sag95). Non-faceted keeps the prior 4*tol solid->mesh cap. Not tuned per part.
        if time.time() > t_stop:
            res.ok = False
            res.reason = "section loft over its time budget (before self_check)"
            res.seconds = time.time() - t0
            return res
        from .deviation import deviation
        from .facets import facet_stats
        # Coarse tessellation if little budget remains — still a valid reject/accept signal;
        # fine lin was hanging multi-minute on large smooth skins (WING 2 L).
        remain = t_stop - time.time()
        # Shared-wire sew shells can have thousands of faces; fine OCC tess hangs minutes.
        sewn = any(s.get("shared_wire") for s in res.slabs)
        if sewn:
            lin, n_samp, ang = max(tol, 0.05), 2000, 0.25
        elif remain <= 20.0 or res.n_faces > 2000:
            lin, n_samp, ang = max(tol, 0.02), 3000, 0.1
        else:
            lin, n_samp, ang = max(tol / 4, 0.002), 8000, 0.1
        dev = deviation(mesh, solid, None, n_samples=n_samp, lin=lin, ang=ang)[0]
        vd, s2m = dev["mesh_vertices_to_solid"], dev["solid_to_mesh"]
        st = facet_stats(mesh)
        faceted = (st.get("coplanar_len_frac", 0) > 0.02
                   and st.get("turning_smooth_len_frac", 0) > 0.15)
        sag = float(st.get("sag_p90", 0.0)) if faceted else 0.0
        from .accuracy import resolve_gates
        gates = resolve_gates(accuracy, tol)
        lim_rms, lim_p95 = gates["rms_limit"], gates["p95_limit"]
        lim_s2m = (lim_p95 + sag) if faceted else (max(4.0 * tol, lim_p95 * 4 / 3))
        res.self_check = {"mesh_vertices_to_solid_p95": round(vd["p95"], 5), "rms": round(vd["rms"], 5),
                          "solid_to_mesh_p95": round(s2m["p95"], 5), "hausdorff": round(dev["hausdorff"], 5),
                          "faceted": bool(faceted), "facet_sag_p90": round(sag, 5),
                          "solid_to_mesh_p95_limit": round(lim_s2m, 5),
                          "accuracy": gates["class"], "rms_limit": round(lim_rms, 5),
                          "p95_limit": round(lim_p95, 5)}
        if vd["rms"] > lim_rms or vd["p95"] > lim_p95 or s2m["p95"] > lim_s2m:
            # Pocket recovery (printed skins): high residual often = on-skin cavities the
            # outer loft bridges. Cut non-splitting pocket tools and re-gate once.
            pocket_tried = False
            if (gates["class"] == "printed" and solid is not None
                    and time.time() + 5.0 < t_stop and vd["rms"] < 1.5):
                try:
                    from . import loft_features as _LF
                    from .deviation import deviation as _dev2
                    from .occ_util import shape_to_trimesh as _s2t
                    from .deviation import _on_surface
                    try:
                        loft_tm = _s2t(solid, lin=max(tol, 0.03), ang=0.2)
                    except Exception:
                        loft_tm = None
                    if loft_tm is not None:
                        cp, dvec, _ = _on_surface(loft_tm, np.asarray(mesh.vertices, float))
                        dists = np.asarray(dvec, float)
                        clusters, thr, link = _LF.cluster_high_residual(
                            np.asarray(mesh.vertices, float), dists, thr=max(0.25, lim_rms * 4),
                            extents_norm=float(np.linalg.norm(mesh.extents)))
                        if clusters:
                            pocket_tried = True
                            new_sh, plog = _LF.cut_pockets_from_loft(
                                mesh, solid, dists, cp, clusters, top_k=3,
                                crease_deg=45.0, t_stop=t_stop)
                            res.pockets = {"log": plog, "n_clusters": len(clusters), "thr": thr}
                            if new_sh is not None and any(x.get("ok") for x in plog):
                                solid = new_sh
                                res.shape = solid
                                try:
                                    from OCP.TopAbs import TopAbs_FACE, TopAbs_SOLID
                                    from OCP.TopExp import TopExp_Explorer
                                    from OCP.BRepGProp import BRepGProp
                                    from OCP.GProp import GProp_GProps
                                    from OCP.BRepCheck import BRepCheck_Analyzer
                                    ex = TopExp_Explorer(solid, TopAbs_SOLID)
                                    nsol = 0
                                    while ex.More():
                                        nsol += 1; ex.Next()
                                    pg = GProp_GProps(); BRepGProp.VolumeProperties_s(solid, pg)
                                    res.volume = float(pg.Mass())
                                    res.n_solids = nsol
                                    nfaces = 0
                                    exf = TopExp_Explorer(solid, TopAbs_FACE)
                                    while exf.More():
                                        nfaces += 1; exf.Next()
                                    res.n_faces = nfaces
                                    res.brepcheck_valid = bool(BRepCheck_Analyzer(solid).IsValid())
                                except Exception:
                                    pass
                                if res.n_solids == 1 and res.brepcheck_valid:
                                    dev2 = _dev2(mesh, solid, None, n_samples=n_samp, lin=lin, ang=ang)[0]
                                    vd2, s2m2 = dev2["mesh_vertices_to_solid"], dev2["solid_to_mesh"]
                                    res.self_check = {
                                        "mesh_vertices_to_solid_p95": round(vd2["p95"], 5),
                                        "rms": round(vd2["rms"], 5),
                                        "solid_to_mesh_p95": round(s2m2["p95"], 5),
                                        "hausdorff": round(dev2["hausdorff"], 5),
                                        "faceted": bool(faceted), "facet_sag_p90": round(sag, 5),
                                        "solid_to_mesh_p95_limit": round(lim_s2m, 5),
                                        "accuracy": gates["class"], "rms_limit": round(lim_rms, 5),
                                        "p95_limit": round(lim_p95, 5),
                                        "after_pockets": True,
                                    }
                                    if (vd2["rms"] <= lim_rms and vd2["p95"] <= lim_p95
                                            and s2m2["p95"] <= lim_s2m):
                                        res.ok = True
                                        res.reason = ""
                                        res.method = "section_loft+pockets"
                                        res.seconds = time.time() - t0
                                        return res
                                    vd, s2m = vd2, s2m2
                except Exception as ex:  # noqa: BLE001
                    res.pockets = {"error": f"{type(ex).__name__}: {ex}"}
            res.ok = False
            res.reason = (f"loft deviates from the mesh: vertices->solid rms {vd['rms']:.4g} / p95 {vd['p95']:.4g}, "
                          f"solid->mesh p95 {s2m['p95']:.4g} (accuracy={gates['class']}, "
                          f"rms≤{lim_rms:.4g}, p95≤{lim_p95:.4g}, lim_s2m {lim_s2m:.4g})"
                          + ("; pockets tried" if pocket_tried else ""))
    res.seconds = time.time() - t0
    return res
