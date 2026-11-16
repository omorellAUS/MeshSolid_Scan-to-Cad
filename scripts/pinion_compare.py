"""Per-zone scan deviation of a meshsolid pinion STEP vs the user's STEP A / B, with the gear sub-zones.

Usage: python scripts/pinion_compare.py scan.stl pinion_dir ours.step gear.pkl out.json [label]
Zones are the STEP-frame axial zones of scripts/pinion_zones.py; the gear zone is additionally split into
mid-flank / tip & edge rounding / root (gear.zones, from the recognised gear) and the same masks are applied
to A, B and ours.  Deviation = distance of each scan vertex to the model surface (signed for ours by the
outward normal; A/B from their saved signed arrays).  With a gear pickle, two extra zones repeat the left
neck and the whole part without the scan's capture-cap vertices (meshsolid.internal.capture_caps)."""
import json, pickle, sys
from pathlib import Path
import numpy as np
import trimesh
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from meshsolid import occ_util as U  # noqa: E402
from meshsolid import gear as G  # noqa: E402

ZONES = {"left journal 46-96": (46, 96), "neck + collar L 96-117": (96, 117), "gear teeth 117-164": (117, 164),
         "collar R + neck 164-185": (164, 185), "journal 185-223": (185, 223), "seat 223-233": (223, 233),
         "233-249 (groove 241.7-243.7)": (233, 249), "D18.41 249-268": (249, 268), "D14.46 end 268-325": (268, 325)}


def stats(d):
    if len(d) == 0:
        return None
    a = np.abs(d)
    return {"n": int(len(d)), "rms": round(float(np.sqrt(np.mean(d ** 2))), 3), "p95": round(float(np.percentile(a, 95)), 3),
            "median_abs": round(float(np.median(a)), 3), "max": round(float(a.max()), 3), "signed_median": round(float(np.median(d)), 3)}


def signed_to(shape, V, lin=0.002):
    m = U.shape_to_trimesh(shape, lin=lin, ang=0.05)
    out = np.empty(len(V))
    for i in range(0, len(V), 4000):          # chunked: the candidate lists of a fine tessellation are large
        Q = V[i:i + 4000]
        cp, d, tri = trimesh.proximity.closest_point(m, Q)
        s = np.sign(np.einsum("ij,ij->i", Q - cp, m.face_normals[tri]))
        out[i:i + 4000] = d * np.where(s == 0, 1, s)
    return out


def main():
    scan = trimesh.load(sys.argv[1], process=True); d0 = Path(sys.argv[2])
    ours = sys.argv[3]; g, _ = pickle.load(open(sys.argv[4], "rb")) if sys.argv[4].endswith(".pkl") else (None, None)
    out_path = Path(sys.argv[5]); label = sys.argv[6] if len(sys.argv) > 6 else "ours"
    V = np.asarray(scan.vertices)
    T = np.load(d0 / "T_stepB_to_scan.npy")
    x = ((V - T[:3, 3]) @ T[:3, :3])[:, 0]
    dev = {"A": np.load(d0 / "scan_vs_stepA.signed.npy"), "B": np.load(d0 / "scan_vs_stepB.signed.npy")}
    cache = Path(str(out_path) + f".{label}.signed.npy")
    if cache.exists():
        dev[label] = np.load(cache)
    else:
        dev[label] = signed_to(U.read_step(ours), V); np.save(cache, dev[label])
    masks = {n: (x >= a) & (x < b) for n, (a, b) in ZONES.items()}
    if g is not None:
        zm = G.zones(V, g)
        for k in ("mid_flank", "tip_edge_rounding", "root"):
            masks[f"gear: {k.replace('_', ' ')}"] = zm[k]
        masks["gear zone (recognised, all)"] = zm["gear"]
    allm = (x >= 46) & (x < 325)
    if g is not None:   # the scan's internal capture caps (scan-only faces at the capture seam), same mask for all
        from meshsolid.internal import capture_caps
        cf, _ = capture_caps(scan, np.asarray(g.axis_point), np.asarray(g.axis_dir))
        cv = np.zeros(len(V), bool); cv[np.asarray(scan.faces)[cf].ravel()] = True
        masks["neck + collar L 96-117, capture caps excluded"] = masks["neck + collar L 96-117"] & ~cv
        masks["all 46-325, capture caps excluded"] = allm & ~cv
    res = {k: {n: stats(dv[m]) for n, m in masks.items()} for k, dv in dev.items()}
    res["all scanned"] = {k: stats(dv[allm]) for k, dv in dev.items()}
    prev = json.loads(out_path.read_text()) if out_path.exists() else {}
    prev.update(res)
    out_path.write_text(json.dumps(prev, indent=1))
    keys = ["A", "B", label]
    print(f"{'zone':38s}" + "".join(f"{k:>22s}" for k in keys))
    for n in masks:
        print(f"{n:38s}" + "".join(f"{(res[k][n]['rms'] if res[k][n] else float('nan')):>10.3f}/{(res[k][n]['p95'] if res[k][n] else float('nan')):<10.3f}" + " " for k in keys))
    print(f"{'all 46-325':38s}" + "".join(f"{res['all scanned'][k]['rms']:>10.3f}/{res['all scanned'][k]['p95']:<10.3f} " for k in keys))


if __name__ == "__main__":
    main()
