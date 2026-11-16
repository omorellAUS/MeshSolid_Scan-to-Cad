"""Preview of an impeller STEP: shaded solid (two views) + scan deviation map.
Usage: python scripts/impeller_preview.py model.step out.png [scan_vs_prefix] [title]"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt  # noqa: E401
from meshsolid import occ_util as U
from meshsolid.render import _draw, _shade

shape = U.read_step(sys.argv[1])
m = U.shape_to_trimesh(shape, lin=0.05, ang=0.35)
pre = sys.argv[3] if len(sys.argv) > 3 else None
fig = plt.figure(figsize=(21, 7), dpi=100)
b = m.bounds
# the eye (small-radius nose) end of the z axis: the end whose slab of the part has the smaller radial extent
V = m.vertices; c0 = V[:, :2].mean(0); rr = np.linalg.norm(V[:, :2] - c0, axis=1); zz = V[:, 2]
lo, hi = np.percentile(zz, [3, 97])
eye_up = np.percentile(rr[zz > hi], 90) < np.percentile(rr[zz < lo], 90)
el = 1 if eye_up else -1
ax = fig.add_subplot(131, projection="3d"); _draw(ax, m, _shade(m, np.array([0.45, 0.6, 0.85])), f"solid ({len(m.faces)} tris), eye side oblique", b); ax.view_init(25 * el, -60)
ax = fig.add_subplot(132, projection="3d"); _draw(ax, m, _shade(m, np.array([0.45, 0.6, 0.85])), "eye side", b); ax.view_init(70 * el, -60)
if pre:
    P = np.load(pre + ".pts.npy"); d = np.load(pre + ".signed.npy")
    ax = fig.add_subplot(133); k = np.abs(d) < 2
    c = m.bounds.mean(0)
    sc = ax.scatter(P[k, 0], P[k, 1], c=np.clip(d[k], -0.3, 0.3), s=0.05, cmap="coolwarm")
    ax.set_aspect("equal"); ax.set_title("scan deviation to the solid, top view (mm, clipped ±0.3)")
    fig.colorbar(sc, ax=ax, shrink=0.7)
fig.suptitle(sys.argv[4] if len(sys.argv) > 4 else Path(sys.argv[1]).name)
fig.tight_layout(); fig.savefig(sys.argv[2]); print("wrote", sys.argv[2])
