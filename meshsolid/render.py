"""Offscreen matplotlib preview: input mesh | reconstructed solid | deviation map."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def _shade(m, base):
    light = np.array([0.4, -0.5, 0.75]); light /= np.linalg.norm(light)
    s = 0.35 + 0.65 * np.abs(m.face_normals @ light)
    return np.c_[np.outer(s, base), np.ones(len(s))]


def _draw(ax, m, colors, title, bounds):
    pc = Poly3DCollection(m.vertices[m.faces], facecolors=colors, edgecolors="none", linewidths=0)
    ax.add_collection3d(pc)
    lo, hi = bounds
    c, r = (lo + hi) / 2, (hi - lo).max() / 2
    for set_, i in ((ax.set_xlim, 0), (ax.set_ylim, 1), (ax.set_zlim, 2)):
        set_(c[i] - r, c[i] + r)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(28, -60)
    ax.set_title(title, fontsize=9)
    ax.set_axis_off()


def render_preview(mesh, labels, solid_mesh, dv, path, title=""):
    fig = plt.figure(figsize=(22, 6.5), dpi=110)
    b = mesh.bounds
    _draw(fig.add_subplot(141, projection="3d"), mesh, _shade(mesh, np.array([0.7, 0.7, 0.7])),
          f"input mesh ({len(mesh.faces)} tris)", b)
    rng = np.random.default_rng(3)
    pal = rng.uniform(0.15, 0.95, (max(labels.max() + 1, 1), 3))
    base = np.where(labels[:, None] >= 0, pal[np.maximum(labels, 0)], 0.0)
    light = np.array([0.4, -0.5, 0.75]); light /= np.linalg.norm(light)
    s = 0.45 + 0.55 * np.abs(mesh.face_normals @ light)
    seg = np.c_[base * s[:, None], np.ones(len(s))]
    _draw(fig.add_subplot(142, projection="3d"), mesh, seg,
          f"segmentation ({labels.max() + 1} regions, black=unassigned)", b)
    ax3 = fig.add_subplot(143, projection="3d")
    ax4 = fig.add_subplot(144, projection="3d")
    if solid_mesh is None:
        for ax, t in ((ax3, "NO SOLID (build failed)"), (ax4, "no deviation map")):
            ax.set_axis_off()
            ax.text2D(0.5, 0.5, t, transform=ax.transAxes, ha="center", color="red", fontsize=13)
    else:
        if len(solid_mesh.faces) > 400000:   # preview only: matplotlib cannot draw millions of triangles
            import trimesh
            keep = np.random.default_rng(0).choice(len(solid_mesh.faces), 400000, replace=False)
            solid_mesh = trimesh.Trimesh(solid_mesh.vertices, solid_mesh.faces[np.sort(keep)], process=False)
        _draw(ax3, solid_mesh, _shade(solid_mesh, np.array([0.45, 0.6, 0.85])), "reconstructed B-rep (tessellated)", b)
        vmax = max(float(np.percentile(np.abs(dv), 99)), 1e-6)
        fd = dv[mesh.faces].mean(1)
        cm = matplotlib.colormaps["coolwarm"]
        _draw(ax4, mesh, cm(np.clip((fd / vmax + 1) / 2, 0, 1)), f"mesh->solid signed deviation (±{vmax:.3f} mm)", b)
        sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(-vmax, vmax))
        fig.colorbar(sm, ax=ax4, shrink=0.6, label="mm")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
