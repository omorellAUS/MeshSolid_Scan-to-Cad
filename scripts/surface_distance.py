"""Accurate surface-to-surface distances between tessellated models (point-to-plane against a dense oriented
sample; point-to-point nearest-sample distances are dominated by sample spacing).
Usage: python scripts/surface_distance.py a.stl b.stl [T_b_to_a.npy|none] out.json"""
import json, sys
import numpy as np
import trimesh
from scipy.spatial import cKDTree


def p2s(P, m, n=3000000, seed=0):
    Q, fi = trimesh.sample.sample_surface(m, n, seed=seed)
    N = m.face_normals[fi]
    dd, j = cKDTree(Q).query(P)
    ds = np.einsum("ij,ij->i", P - Q[j], N[j])
    return np.where(np.abs(ds) <= dd, ds, np.sign(ds) * dd), Q


def stats(d):
    a = np.abs(d)
    return {"rms": round(float(np.sqrt(np.mean(a ** 2))), 4), "p95": round(float(np.percentile(a, 95)), 4),
            "p99": round(float(np.percentile(a, 99)), 4), "max": round(float(a.max()), 4), "median": round(float(np.median(a)), 4)}


if __name__ == "__main__":
    a = trimesh.load(sys.argv[1], process=False); b = trimesh.load(sys.argv[2], process=False)
    if sys.argv[3] != "none":
        b.apply_transform(np.load(sys.argv[3]))
    Pb = trimesh.sample.sample_surface(b, 200000, seed=5)[0]
    Pa = trimesh.sample.sample_surface(a, 200000, seed=6)[0]
    dba, _ = p2s(Pb, a, seed=1); dab, _ = p2s(Pa, b, seed=2)
    out = {"b_to_a": stats(dba), "a_to_b": stats(dab), "hausdorff_sampled": round(float(max(np.abs(dba).max(), np.abs(dab).max())), 4),
           "area_a": round(float(a.area), 1), "area_b": round(float(b.area), 1)}
    # where are the big differences (> 0.1 mm)? axial histogram along the longest bbox axis of a
    ax = int(np.argmax(a.extents))
    for key, P, d in (("b_to_a", Pb, dba), ("a_to_b", Pa, dab)):
        k = np.abs(d) > 0.1
        h, e = np.histogram(P[k, ax], bins=np.arange(np.floor(a.bounds[0, ax]), a.bounds[1, ax] + 5, 5))
        out[key + "_gt0.1mm_by_5mm"] = {str(int(x)): int(c) for x, c in zip(e[:-1], h) if c > 50}
    json.dump(out, open(sys.argv[4], "w"), indent=1)
    print(json.dumps(out, indent=1))
