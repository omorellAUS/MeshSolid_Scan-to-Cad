"""Unrolled deviation map (axial position x angle) of the scan against each user STEP, plus the axial
radius profile. Usage: python scripts/pinion_devmap.py scan.stl pinion_dir out.png"""
import json, sys
from pathlib import Path
import numpy as np
import trimesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
scan = trimesh.load(sys.argv[1], process=True); d0 = Path(sys.argv[2])
V = np.asarray(scan.vertices)
fig, ax = plt.subplots(3, 1, figsize=(15, 11), gridspec_kw={"height_ratios": [1, 1, 0.8]})
for i, k in enumerate("AB"):
    T = np.load(d0 / f"T_step{k}_to_scan.npy"); d = np.load(d0 / f"scan_vs_step{k}.signed.npy")
    Vi = (V - T[:3, 3]) @ T[:3, :3]
    x = Vi[:, 0]; th = np.degrees(np.arctan2(Vi[:, 2], Vi[:, 1]))
    sc = ax[i].scatter(x, th, c=np.clip(d, -0.3, 0.3), s=0.15, cmap="coolwarm", vmin=-0.3, vmax=0.3, rasterized=True)
    ax[i].set_ylabel("angle (deg)"); ax[i].set_xlim(45, 326)
    ax[i].set_title(f"scan minus STEP {k} (signed mm, red = scan outside the model), STEP frame x")
    fig.colorbar(sc, ax=ax[i], pad=0.01)
pr = json.load(open(d0 / "pinion_profiles.json"))
P = np.array([[np.nan if v is None else v for v in row] for row in pr["profile"]], float)
for j, lab in enumerate(["scan", "STEP A", "STEP B"], start=1):
    ax[2].plot(P[:, 0], P[:, j], lw=0.9, label=lab)
ax[2].set_xlim(45, 326); ax[2].set_xlabel("x along shaft (mm, STEP frame)"); ax[2].set_ylabel("D99 (mm)"); ax[2].legend()
ax[2].set_title("outer diameter profile (99th percentile radius x 2 per 0.5 mm slice)")
fig.tight_layout(); fig.savefig(sys.argv[3], dpi=110)
print("saved", sys.argv[3])
