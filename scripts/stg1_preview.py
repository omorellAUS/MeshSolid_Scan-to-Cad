"""Stage 1 impeller preview: scan coloured by signed deviation to each user model (top and side views)."""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
d0 = sys.argv[1]; out = sys.argv[2]
names = ["brep_v1", "circ_pattern", "solid_body"]
fig, ax = plt.subplots(2, 3, figsize=(18, 10.5))
for i, n in enumerate(names):
    P = np.load(f"{d0}/scan_vs_{n}.pts.npy"); d = np.load(f"{d0}/scan_vs_{n}.signed.npy")
    k = np.argsort(np.abs(d))
    sc = ax[0, i].scatter(P[k, 0], P[k, 1], c=np.clip(d[k], -0.4, 0.4), s=0.08, cmap="coolwarm", vmin=-0.4, vmax=0.4, rasterized=True)
    ax[0, i].set_aspect("equal"); ax[0, i].set_title(f"scan vs {n}: top view\n(signed mm, red = scan outside model)", fontsize=9)
    r = np.hypot(P[:, 0], P[:, 1])
    ax[1, i].scatter(r[k], P[k, 2], c=np.clip(d[k], -0.4, 0.4), s=0.08, cmap="coolwarm", vmin=-0.4, vmax=0.4, rasterized=True)
    ax[1, i].set_xlabel("radius (mm)"); ax[1, i].set_ylabel("z, model frame (mm)"); ax[1, i].set_title(f"scan vs {n}: radius-z view", fontsize=9)
fig.colorbar(sc, ax=ax.ravel().tolist(), shrink=0.6)
fig.savefig(out, dpi=100)
print("saved", out)
