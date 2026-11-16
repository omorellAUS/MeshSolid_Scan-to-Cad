"""Optional GPU acceleration (M8b).  Engine-side only; the add-ins are unchanged.

Device choice: --device auto|cpu|cuda (or MESHSOLID_DEVICE).  "auto" uses CUDA when NVIDIA Warp is
installed and sees a CUDA device, else the CPU.  "cuda" without a usable GPU is an error, never a silent
CPU run.  The device used is recorded in report.json.

Accelerated so far:
  closest_points  point -> triangle-mesh closest point (deviation stage; the heaviest query).  Warp's BVH
                  finds the closest face in float32 on centred coordinates; the distance and closest
                  point are then recomputed exactly in float64 on that face, so results match the CPU
                  path (libigl) except where two faces are within float32 noise of equally close (then
                  the distance is still the same to ~1e-6 mm).
The same Warp kernel also runs on Warp's CPU device: that is the equivalence test path on machines
without a GPU (the box)."""
from __future__ import annotations

import os

import numpy as np

_STATE = {"pref": None, "resolved": None, "info": None}


def _warp():
    try:
        import warp as wp
    except Exception:  # noqa: BLE001
        return None
    try:
        wp.config.quiet = True
        wp.config.verbose = False
        wp.init()
    except Exception:  # noqa: BLE001
        return None
    return wp


def _driver_present():
    """Cheap NVIDIA driver probe without importing Warp (importing it maps several GB of virtual
    memory for the CUDA/NVRTC libraries, which broke memory-capped CPU runs)."""
    import ctypes
    import ctypes.util
    names = ["nvcuda.dll"] if os.name == "nt" else ["libcuda.so.1", "libcuda.so"]
    if os.name != "nt" and ctypes.util.find_library("cuda"):
        return True
    for n in names:
        try:
            ctypes.CDLL(n)
            return True
        except OSError:
            continue
    return False


def cuda_info():
    """(available, description)."""
    if not _driver_present():
        return False, "no NVIDIA CUDA driver found"
    wp = _warp()
    if wp is None:
        return False, "NVIDIA Warp not installed (pip install warp-lang)"
    try:
        n = wp.get_cuda_device_count()
    except Exception as ex:  # noqa: BLE001
        return False, f"Warp present but CUDA unusable: {ex}"
    if n < 1:
        return False, "Warp present, no CUDA device (no NVIDIA GPU or driver)"
    d = wp.get_cuda_device(0)
    return True, f"{d.name} (Warp {wp.__version__})"


def set_device(pref: str | None):
    """Resolve and remember the device for this process.  Returns the resolved name ('cpu' or 'cuda')."""
    pref = (pref or os.environ.get("MESHSOLID_DEVICE") or "auto").lower()
    if pref not in ("auto", "cpu", "cuda", "warp-cpu"):
        raise ValueError(f"device must be auto, cpu or cuda (got {pref!r})")
    ok, desc = cuda_info() if pref in ("auto", "cuda") else (False, "")
    if pref == "cuda" and not ok:
        raise RuntimeError(f"--device cuda requested but no usable GPU: {desc}")
    if pref == "warp-cpu" and _warp() is None:
        raise RuntimeError("--device warp-cpu needs NVIDIA Warp (pip install warp-lang)")
    res = "cuda" if (pref in ("auto", "cuda") and ok) else ("warp-cpu" if pref == "warp-cpu" else "cpu")
    _STATE.update(pref=pref, resolved=res, info=desc or None)
    return res


def device():
    if _STATE["resolved"] is None:
        set_device(None)
    return _STATE["resolved"]


def report():
    device()
    return {"requested": _STATE["pref"], "used": _STATE["resolved"], "gpu": _STATE["info"],
            "accelerated": ["deviation closest-point queries"] if _STATE["resolved"] != "cpu" else []}


# ----------------------------------------------------------------------------- closest points
def _closest_on_tri(P, A, B, C):
    """Exact closest points on triangles (A, B, C) for points P (all (n, 3), float64; Ericson 5.1.5)."""
    ab, ac, ap = B - A, C - A, P - A
    d1 = np.einsum("ij,ij->i", ab, ap); d2 = np.einsum("ij,ij->i", ac, ap)
    bp = P - B; d3 = np.einsum("ij,ij->i", ab, bp); d4 = np.einsum("ij,ij->i", ac, bp)
    cp_ = P - C; d5 = np.einsum("ij,ij->i", ab, cp_); d6 = np.einsum("ij,ij->i", ac, cp_)
    va = d3 * d6 - d5 * d4; vb = d5 * d2 - d1 * d6; vc = d1 * d4 - d3 * d2
    out = np.empty_like(P)
    done = np.zeros(len(P), bool)

    def put(mask, val):
        m = mask & ~done
        out[m] = val[m] if val.ndim == 2 else val
        done[m] = True
    put((d1 <= 0) & (d2 <= 0), A)
    put((d3 >= 0) & (d4 <= d3), B)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = d1 / (d1 - d3); put((vc <= 0) & (d1 >= 0) & (d3 <= 0), A + v[:, None] * ab)
        put((d6 >= 0) & (d5 <= d6), C)
        w = d2 / (d2 - d6); put((vb <= 0) & (d2 >= 0) & (d6 <= 0), A + w[:, None] * ac)
        w2 = (d4 - d3) / ((d4 - d3) + (d5 - d6)); put((va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0), B + w2[:, None] * (C - B))
        den = 1.0 / (va + vb + vc); v3 = vb * den; w3 = vc * den
        put(np.ones(len(P), bool), A + v3[:, None] * ab + w3[:, None] * ac)
    return out


_KERNEL = {}


def _kernel(wp):
    if "k" in _KERNEL:
        return _KERNEL["k"]

    @wp.func
    def cpt(p: wp.vec3d, a: wp.vec3d, b: wp.vec3d, c: wp.vec3d):
        # Ericson's closest point on a triangle, in float64 (float32 is off by ~1e-2 mm on slivers)
        z = wp.float64(0.0)
        ab = b - a; ac = c - a; ap = p - a
        d1 = wp.dot(ab, ap); d2 = wp.dot(ac, ap)
        if d1 <= z and d2 <= z:
            return a
        bp = p - b; d3 = wp.dot(ab, bp); d4 = wp.dot(ac, bp)
        if d3 >= z and d4 <= d3:
            return b
        vc = d1 * d4 - d3 * d2
        if vc <= z and d1 >= z and d3 <= z:
            return a + ab * (d1 / (d1 - d3))
        cp = p - c; d5 = wp.dot(ab, cp); d6 = wp.dot(ac, cp)
        if d6 >= z and d5 <= d6:
            return c
        vb = d5 * d2 - d1 * d6
        if vb <= z and d2 >= z and d6 <= z:
            return a + ac * (d2 / (d2 - d6))
        va = d3 * d6 - d5 * d4
        if va <= z and (d4 - d3) >= z and (d5 - d6) >= z:
            return b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)))
        den = wp.float64(1.0) / (va + vb + vc)
        return a + ab * (vb * den) + ac * (vc * den)

    @wp.kernel
    def q(mesh: wp.uint64, X: wp.array(dtype=wp.vec3d), I: wp.array(dtype=wp.int32), pts: wp.array(dtype=wp.vec3),
          pts64: wp.array(dtype=wp.vec3d), max_d: float, slack: float, face: wp.array(dtype=wp.int32)):
        i = wp.tid()
        p = pts[i]
        r = wp.mesh_query_point_no_sign(mesh, p, max_d)
        bd = max_d
        if r.result:
            bd = wp.length(p - wp.mesh_eval_position(mesh, r.face, r.u, r.v))
        # float32 BVH: every face whose box is within the found distance (+ slack) is a candidate,
        # decided in float64
        rad = bd + slack
        p64 = pts64[i]
        best = int(-1)
        be = wp.float64(1e30)
        qa = wp.mesh_query_aabb(mesh, p - wp.vec3(rad, rad, rad), p + wp.vec3(rad, rad, rad))
        fi = int(0)
        while wp.mesh_query_aabb_next(qa, fi):
            x = cpt(p64, X[I[3 * fi]], X[I[3 * fi + 1]], X[I[3 * fi + 2]])
            dd = wp.length(p64 - x)
            if dd < be:
                be = dd
                best = fi
        face[i] = best
    _KERNEL["k"] = q
    return q


def closest_points(V, F, P, dev=None):
    """(closest points, distances, face ids) of points P on the triangle mesh (V, F), on `dev`
    ('cuda' or 'warp-cpu').  float64 exact on the face Warp's BVH picked."""
    wp = _warp()
    if wp is None:
        raise RuntimeError("NVIDIA Warp not installed")
    dev = dev or device()
    wdev = "cuda:0" if dev == "cuda" else "cpu"
    V = np.asarray(V, float); F = np.asarray(F, np.int64); P = np.asarray(P, float)
    c = V.mean(0); scale = float(np.ptp(V, axis=0).max()) or 1.0
    with wp.ScopedDevice(wdev):
        mv = wp.array((V - c).astype(np.float32), dtype=wp.vec3)
        mi = wp.array(F.astype(np.int32).ravel(), dtype=wp.int32)
        mesh = wp.Mesh(points=mv, indices=mi)
        pts = wp.array((P - c).astype(np.float32), dtype=wp.vec3)
        face = wp.zeros(len(P), dtype=wp.int32)
        X64 = wp.array(V - c, dtype=wp.vec3d)
        P64 = wp.array(P - c, dtype=wp.vec3d)
        wp.launch(_kernel(wp), dim=len(P), inputs=[mesh.id, X64, mi, pts, P64, 1e3 * scale, 4e-6 * scale + 1e-6, face])
        f = face.numpy().astype(np.int64)
    if (f < 0).any():
        raise RuntimeError("closest-point query found no face for some points")
    # distance and closest point recomputed on the host in float64 (same formula as the CPU reference)
    T = V[F[f]]
    cp = _closest_on_tri(P, T[:, 0], T[:, 1], T[:, 2])
    return cp, np.linalg.norm(P - cp, axis=1), f
