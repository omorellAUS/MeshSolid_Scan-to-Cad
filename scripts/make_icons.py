"""Generate the MeshSolid Fusion add-in icons (PNG) programmatically with PIL.

Only one icon: the main MeshSolid button.  It is the stage-1 impeller deviation heatmap
(scripts/metrology_icon.py) when that render or its scan is on this machine, else a procedural impeller
(left half triangulated mesh, right half shaded solid).  Sizes 16x16, 16x16@2x, 32x32, 32x32@2x, 64x64,
the file names Fusion looks for (UserInterface_UM, "Command Icons").  Drawn at 8x and downsampled
(LANCZOS) for antialiasing; small sizes use thicker relative strokes.

    python scripts/make_icons.py            # writes fusion_addin/MeshSolid/resources/*
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "fusion_addin" / "MeshSolid" / "resources"
SS = 8                                    # supersampling factor

BLUE = (38, 120, 214, 255)
BLUE_D = (20, 70, 140, 255)
BLUE_L = (120, 185, 245, 255)
GREY = (150, 156, 164, 255)
GREY_D = (70, 76, 84, 255)
ORANGE = (240, 140, 30, 255)
CLEAR = (0, 0, 0, 0)


def _blade_poly(cx, cy, r_in, r_out, a0, sweep, width, n=14):
    """Curved (backswept) blade as a polygon between two offset spirals."""
    pts_a, pts_b = [], []
    for i in range(n + 1):
        t = i / n
        r = r_in + (r_out - r_in) * t
        a = a0 + sweep * t
        w = width * (1.0 - 0.55 * t)
        pts_a.append((cx + r * math.cos(a - w / r), cy + r * math.sin(a - w / r)))
        pts_b.append((cx + r * math.cos(a + w / r), cy + r * math.sin(a + w / r)))
    return pts_a + pts_b[::-1]


def _impeller_mask(size, blades, detail):
    """L-mode mask of the impeller silhouette at size*SS."""
    S = size * SS
    m = Image.new("L", (S, S), 0)
    d = ImageDraw.Draw(m)
    cx = cy = S / 2
    r_out = S * 0.47
    r_hub = S * (0.24 if detail else 0.28)
    for k in range(blades):
        a0 = 2 * math.pi * k / blades
        d.polygon(_blade_poly(cx, cy, r_hub * 0.8, r_out, a0, 0.9, S * (0.07 if detail else 0.10)), fill=255)
    d.ellipse([cx - r_hub, cy - r_hub, cx + r_hub, cy + r_hub], fill=255)
    return m


def main_icon(size):
    S = size * SS
    detail = size >= 32
    blades = 7 if detail else 5
    mask = _impeller_mask(size, blades, detail)
    img = Image.new("RGBA", (S, S), CLEAR)
    # right half: shaded solid (radial-ish gradient by x)
    solid = Image.new("RGBA", (S, S), CLEAR)
    sd = ImageDraw.Draw(solid)
    for x in range(S // 2, S):
        t = (x - S / 2) / (S / 2)
        c = tuple(int(BLUE_L[i] * (1 - t) + BLUE_D[i] * t) for i in range(3)) + (255,)
        sd.line([(x, 0), (x, S)], fill=c)
    # left half: grey mesh with triangle wireframe
    mesh = Image.new("RGBA", (S, S), CLEAR)
    md = ImageDraw.Draw(mesh)
    md.rectangle([0, 0, S // 2, S], fill=GREY)
    step = S / (6 if detail else 3)
    lw = max(SS, int(S * (0.018 if detail else 0.05)))
    i = -S
    while i < 2 * S:
        md.line([(i, 0), (i + S, S)], fill=GREY_D, width=lw)
        i += step
    y = 0.0
    while y <= S:
        md.line([(0, y), (S // 2, y)], fill=GREY_D, width=lw)
        y += step
    x = 0.0
    while x <= S / 2:
        md.line([(x, 0), (x, S)], fill=GREY_D, width=lw)
        x += step
    half_l = Image.new("L", (S, S), 0); ImageDraw.Draw(half_l).rectangle([0, 0, S // 2, S], fill=255)
    half_r = Image.new("L", (S, S), 0); ImageDraw.Draw(half_r).rectangle([S // 2, 0, S, S], fill=255)
    from PIL import ImageChops
    img.paste(mesh, (0, 0), ImageChops.multiply(mask, half_l))
    img.paste(solid, (0, 0), ImageChops.multiply(mask, half_r))
    d = ImageDraw.Draw(img)
    # bore
    rb = S * (0.09 if detail else 0.11)
    d.ellipse([S / 2 - rb, S / 2 - rb, S / 2 + rb, S / 2 + rb], fill=CLEAR)
    # divider: orange "conversion" line
    d.line([(S / 2, S * 0.02), (S / 2, S * 0.98)], fill=ORANGE, width=max(SS, int(S * (0.035 if detail else 0.07))))
    return img.resize((size, size), Image.LANCZOS)




def write_all(res: Path = RES, procedural_main=False) -> list[Path]:
    """Main icon = stage-1 impeller metrology render (scripts/metrology_icon.py) when its render or the
    scan + STEP are on this machine; otherwise the procedural mesh-to-solid impeller below."""
    out = []
    d = res / "MeshSolid"; d.mkdir(parents=True, exist_ok=True)
    try:
        import metrology_icon as MI
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parent)); import metrology_icon as MI
    if not procedural_main and (MI.RENDER.exists() or (MI.SCAN.exists() and MI.STEP.exists())):
        out += MI.write_main_icons(d)
    else:
        for name, px in (("16x16", 16), ("16x16@2x", 32), ("32x32", 32), ("32x32@2x", 64), ("64x64", 64)):
            p = d / f"{name}.png"; main_icon(px).save(p); out.append(p)
    return out


def preview_sheet(path: Path, res: Path = RES):
    """One PNG: every icon enlarged (top), then the main icon at its real pixel sizes on a dark and a
    light strip (Fusion's dark and light themes)."""
    tiles = [Image.open(res / "MeshSolid" / "64x64.png"), Image.open(res / "MeshSolid" / "32x32.png"),
             Image.open(res / "MeshSolid" / "16x16.png")]
    W = max(140 * len(tiles), 300)
    sheet = Image.new("RGBA", (W, 330), (60, 63, 68, 255))
    ImageDraw.Draw(sheet).rectangle([0, 245, W, 330], fill=(236, 236, 236, 255))
    for i, t in enumerate(tiles):
        big = t.resize((128, 128), Image.NEAREST if t.width < 64 else Image.LANCZOS)
        sheet.paste(big, (i * 140 + 6, 16), big)
    x = 10
    for name in ("16x16", "32x32", "64x64", "16x16@2x", "32x32@2x"):
        t = Image.open(res / "MeshSolid" / f"{name}.png")
        for y0 in (170, 255):
            sheet.paste(t, (x, y0), t)
        x += t.width + 24
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


if __name__ == "__main__":
    files = write_all()
    preview_sheet(ROOT / "dist" / "icons_preview.png")
    print(f"{len(files)} icon files under {RES}; preview dist/icons_preview.png")
