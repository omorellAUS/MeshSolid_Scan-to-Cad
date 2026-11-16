"""Scan-to-STEP signed deviation per axial zone (STEP frame). Usage: python scripts/pinion_zones.py scan.stl pinion_dir"""
import json, sys
from pathlib import Path
import numpy as np
import trimesh
scan = trimesh.load(sys.argv[1], process=True); d0 = Path(sys.argv[2]); V = np.asarray(scan.vertices)
zones = {"left journal 46-96": (46, 96), "neck + collar L 96-117": (96, 117), "gear teeth 117-164": (117, 164),
         "collar R + neck 164-185": (164, 185), "journal 185-223": (185, 223), "seat 223-233": (223, 233),
         "233-249 (groove 241.7-243.7)": (233, 249), "D18.41 249-268": (249, 268), "D14.46 end 268-325": (268, 325)}
out = {}
for k in "AB":
    T = np.load(d0 / f"T_step{k}_to_scan.npy"); d = np.load(d0 / f"scan_vs_step{k}.signed.npy")
    x = ((V - T[:3, 3]) @ T[:3, :3])[:, 0]
    zs = {}
    for n, (a, b) in zones.items():
        dd = d[(x >= a) & (x < b)]
        zs[n] = {"n": int(len(dd)), "rms": round(float(np.sqrt(np.mean(dd ** 2))), 3), "p95": round(float(np.percentile(np.abs(dd), 95)), 3),
                 "max": round(float(np.abs(dd).max()), 3), "signed_median": round(float(np.median(dd)), 3)}
    big = np.abs(d) > 1.0
    out[k] = {"zones": zs, "gt1mm_points": int(big.sum()),
              "gt1mm_by_5mm": {int(e): int(h) for e, h in zip(np.arange(45, 330, 5), np.histogram(x[big], bins=np.arange(45, 335, 5))[0]) if h > 10}}
json.dump(out, open(d0 / "scan_vs_step_zones.json", "w"), indent=1)
for k in "AB":
    print(k, out[k]["gt1mm_points"], out[k]["gt1mm_by_5mm"])
    for n, z in out[k]["zones"].items(): print("  ", n, z)
