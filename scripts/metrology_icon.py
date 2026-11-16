"""Main MeshSolid icon from the stage-1 impeller metrology: the user's scan coloured by its signed
deviation to the built solid (turbo map, +-0.25 mm, green = on size), seen end-on from the blade side.

Inputs are the user's private scan and its built STEP; the full-size render and the deviation array go
to out/icon_src/ (git-ignored).  Only the small icon PNGs (<= 64 px) are written into the add-in.

    python scripts/metrology_icon.py [scan.stl] [solid.step]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "out" / "icon_src"
SCAN = ROOT / "results" / "regression_loft_guard_2s" / "prepped" / "stg1_scale01.stl"
STEP = ROOT / "results" / "baseline_rerun" / "stg1" / "stg1.step"
RENDER = SRC / "stg1_metrology_end.png"
VMAX = 0.25


def deviations(scan, step):
    import trimesh
    from scipy.spatial import cKDTree
    sys.path.insert(0, str(ROOT))
    from meshsolid import occ_util as O
    sol = O.shape_to_trimesh(O.read_step(str(step)), lin=0.02, ang=0.1)
    Q, fi = trimesh.sample.sample_surface(sol, 6_000_000, seed=0)
    d, j = cKDTree(Q).query(scan.vertices, workers=-1)
    sgn = np.sign(np.einsum("ij,ij->i", scan.vertices - Q[j], sol.face_normals[fi[j]]))
    return (d * sgn).astype(np.float32)


def render(scan, dv, size=1200, elev=180.0, azim=0.0):
    """Painter's-algorithm render, per-face colour = turbo(signed deviation) x Lambert shade."""
    from matplotlib import colormaps
    V = scan.vertices - scan.bounds.mean(0)
    a, e = np.radians(azim), np.radians(elev)
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0, np.sin(e), np.cos(e)]])
    P = V @ Rz.T @ Rx.T
    F = scan.faces
    c = colormaps["turbo"](np.clip(0.5 + 0.5 * dv[F].mean(1) / VMAX, 0, 1))[:, :3]
    n = np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]])
    n /= np.linalg.norm(n, axis=1)[:, None] + 1e-12
    L = np.array([-0.35, 0.45, 0.82]); L /= np.linalg.norm(L)
    col = (np.clip(c * (0.35 + 0.65 * np.abs(n @ L))[:, None], 0, 1) * 255).astype(np.uint8)
    ext = np.abs(P[:, :2]).max() * 1.01
    xy = (P[:, :2] / ext * 0.5 + 0.5) * size
    xy[:, 1] = size - xy[:, 1]
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0)); d = ImageDraw.Draw(img)
    for i in np.argsort(P[F].mean(1)[:, 2]):
        f = F[i]
        d.polygon([tuple(xy[f[0]]), tuple(xy[f[1]]), tuple(xy[f[2]])], fill=tuple(col[i]) + (255,))
    return img


def square_crop(img, pad=0.02):
    x0, y0, x1, y1 = img.getbbox()
    side = int(max(x1 - x0, y1 - y0) * (1 + 2 * pad))
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    out = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    out.paste(img.crop((cx - side // 2, cy - side // 2, cx - side // 2 + side, cy - side // 2 + side)), (0, 0))
    return out


def icon(src, px):
    """Downsample with pre-blur for small sizes, then boost saturation/contrast; 16 px gets a stronger
    blur (blade detail averaged out into broad colour sectors) and a dark rim so the disc reads."""
    blur = {16: 6.0, 32: 2.5, 64: 1.0}.get(px, 0.6)
    k = src.width / px
    im = src.filter(ImageFilter.GaussianBlur(blur * k / 12)) if blur else src
    rgb, a = im.convert("RGB"), im.getchannel("A")
    rgb = ImageEnhance.Color(rgb).enhance(1.5 if px <= 16 else 1.35)
    rgb = ImageEnhance.Contrast(rgb).enhance(1.35 if px <= 16 else 1.2)
    im = Image.merge("RGBA", (*rgb.split(), a)).resize((px, px), Image.LANCZOS)
    if px >= 32:
        im = im.filter(ImageFilter.UnsharpMask(radius=1, percent=60, threshold=2))
    S = px * 8
    rim = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    ImageDraw.Draw(rim).ellipse([S * .015, S * .015, S * .985, S * .985], outline=(20, 22, 26, 230),
                                width=max(8, int(S * (0.06 if px <= 16 else 0.03))))
    return Image.alpha_composite(im, rim.resize((px, px), Image.LANCZOS))


def source(scan_path=SCAN, step_path=STEP, force=False):
    if RENDER.exists() and not force:
        return Image.open(RENDER).convert("RGBA")
    import trimesh
    SRC.mkdir(parents=True, exist_ok=True)
    scan = trimesh.load(scan_path)
    dv_path = SRC / "stg1_dev.npy"
    dv = np.load(dv_path) if dv_path.exists() and not force else deviations(scan, step_path)
    np.save(dv_path, dv)
    img = square_crop(render(scan, dv))
    img.save(RENDER)
    return img


def write_main_icons(dst: Path, src=None):
    src = src or source()
    dst.mkdir(parents=True, exist_ok=True)
    out = []
    for name, px in (("16x16", 16), ("16x16@2x", 32), ("32x32", 32), ("32x32@2x", 64), ("64x64", 64)):
        p = dst / f"{name}.png"; icon(src, px).save(p); out.append(p)
    return out


if __name__ == "__main__":
    a = sys.argv[1:]
    img = source(Path(a[0]) if a else SCAN, Path(a[1]) if len(a) > 1 else STEP, force=bool(a))
    print("render", RENDER, img.size)
