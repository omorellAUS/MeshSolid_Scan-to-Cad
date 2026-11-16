"""Pure-Python helpers for the Metrology dialog (no numpy, no meshsolid package — Fusion's Python)."""
from __future__ import annotations

import json
import os


# ---- band colours: every default band clearly distinct; red is reserved for "beyond outlier cap" ----
# Side palettes run from next-to-tolerance outwards (plus: yellow, orange, magenta, purple, brown;
# minus: cyan, teal, light blue, blue, navy). Grey = not inspected (face/zone exclusions).
GREEN = (36, 170, 65)
PLUS_COLORS = ((245, 215, 35), (245, 140, 25), (215, 40, 190), (110, 30, 180), (110, 60, 25))
MINUS_COLORS = ((30, 210, 225), (0, 120, 120), (60, 140, 255), (25, 60, 210), (15, 20, 105))
_SIDE_SETS = {1: (2,), 2: (0, 2), 3: (0, 1, 2), 4: (0, 1, 2, 3), 5: (0, 1, 2, 3, 4)}
_MINUS_SETS = {1: (3,), 2: (0, 4), 3: (0, 3, 4), 4: (0, 2, 3, 4), 5: (0, 1, 2, 3, 4)}
PLUS_EXTRA = ((250, 150, 200), (128, 128, 0), (60, 15, 80), (210, 180, 120), (190, 160, 230))
MINUS_EXTRA = ((170, 235, 240), (75, 40, 160), (90, 110, 130), (25, 25, 30), (70, 130, 180))
CAP_RGB = (255, 0, 0)              # beyond the outlier cap (default; >= 90 from every palette colour)
GRAY = (150, 150, 154)             # not inspected (face / zone / hand-entered exclusions)
MIN_COLOR_DIST = 45.0
RESERVED_DIST = 70.0               # band colours keep well away from beyond-cap red and not-inspected grey


def side_colors(n, plus=True):
    """n distinct colours for one side of the pass band, next-to-tolerance first."""
    pal = PLUS_COLORS if plus else MINUS_COLORS
    sets = _SIDE_SETS if plus else _MINUS_SETS
    if n <= 0:
        return []
    if n in sets:
        return [pal[i] for i in sets[n]]
    ext = list(pal) + list(PLUS_EXTRA if plus else MINUS_EXTRA)     # more than 5 on one side
    out = [ext[k % len(ext)] for k in range(n)]
    return make_distinct(out)[0]


def color_dist(a, b):
    """Rough perceptual RGB distance ('redmean')."""
    r = 0.5 * (a[0] + b[0])
    dr, dg, db = a[0] - b[0], a[1] - b[1], a[2] - b[2]
    return ((2 + r / 256.0) * dr * dr + 4 * dg * dg + (2 + (255 - r) / 256.0) * db * db) ** 0.5 / 3.0


def make_distinct(rgbs, reserved=(CAP_RGB, GRAY), min_dist=MIN_COLOR_DIST, reserved_dist=RESERVED_DIST):
    """Adjust colours that are too close to their neighbour or to a reserved colour (beyond-cap red,
    not-inspected grey) by lightening or darkening them. Returns (new_rgbs, changed_indices)."""
    out = [tuple(int(c) for c in x[:3]) for x in rgbs]
    changed = []
    for i in range(len(out)):
        def worst(c):
            # margin to the requirement: neighbours need min_dist, reserved colours reserved_dist
            m = [color_dist(c, out[j]) - min_dist for j in (i - 1, i + 1) if 0 <= j < len(out)]
            m += [color_dist(c, tuple(r)) - reserved_dist for r in reserved]
            return min(m or [1e9])
        if worst(out[i]) >= 0:
            continue
        best, best_d = out[i], worst(out[i])
        for t in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
            for target in ((255, 255, 255), (0, 0, 0)):
                c = tuple(int(round(out[i][k] + (target[k] - out[i][k]) * t)) for k in range(3))
                d = worst(c)
                if d > best_d + 1e-9:
                    best, best_d = c, d
            if best_d >= 0:
                break
        if best != out[i]:
            out[i] = best
            changed.append(i)
    return out, changed


def _linspace(a, b, n):
    if n <= 1:
        return [a]
    return [a + (b - a) * i / (n - 1) for i in range(n)]


def default_bands(vmin, vmax, pass_tol, n_bands):
    vmin, vmax = float(vmin), float(vmax)
    if vmax <= vmin:
        vmax = vmin + 1e-6
    p = abs(float(pass_tol))
    lo, hi = max(-p, vmin), min(p, vmax)
    n_bands = max(3, int(n_bands))
    has_pass = hi > lo + 1e-12
    n_pass = 1 if has_pass else 0
    rest = n_bands - n_pass
    w_lo, w_hi = max(0.0, lo - vmin), max(0.0, vmax - hi)
    if rest <= 0 or w_lo + w_hi <= 0:
        n_pass, n_lo, n_hi = (n_bands, 0, 0)
        if w_lo + w_hi <= 0:
            lo, hi = vmin, vmax
    else:
        n_lo = int(round(rest * w_lo / (w_lo + w_hi)))
        n_lo = min(max(n_lo, 1 if w_lo > 0 else 0), rest - (1 if w_hi > 0 else 0))
        n_hi = rest - n_lo

    def edges(a, b, n):
        return _linspace(a, b, n + 1) if n > 0 and b > a else []

    ev = []
    if n_lo:
        ev += edges(vmin, lo, n_lo)[:-1]
    if n_pass:
        ev += edges(lo, hi, n_pass)[:-1]
    else:
        ev.append(lo if has_pass else vmin)
    ev += edges(hi, vmax, n_hi) if n_hi else [vmax]
    if ev[-1] < vmax:
        ev.append(vmax)
    uniq = [ev[0]]
    for x in ev[1:]:
        if x - uniq[-1] > 1e-12:
            uniq.append(float(x))
    if uniq[-1] < vmax:
        uniq.append(vmax)
    bands, i_lo, i_hi = [], 0, 0
    minus, plus = side_colors(n_lo, plus=False), side_colors(n_hi, plus=True)
    n = len(uniq) - 1
    for i in range(n):
        a, b = float(uniq[i]), float(uniq[i + 1])
        mid = 0.5 * (a + b)
        if mid < lo - 1e-12 and n_lo:
            col = minus[n_lo - 1 - i_lo]; i_lo += 1
        elif mid > hi + 1e-12 and n_hi:
            col = plus[i_hi]; i_hi += 1
        else:
            col = GREEN
        bands.append({"lo": round(a, 6), "hi": round(b, 6), "rgb": list(col),
                      "color": "#%02x%02x%02x" % col,
                      "in_tol": bool(lo - 1e-12 <= mid <= hi + 1e-12)})
    return bands


def parse_xyz(text):
    parts = [p.strip() for p in (text or "").replace(";", ",").split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError(f"expected x,y,z (got {text!r})")
    return [float(p) for p in parts]


def parse_xyz_list(text):
    out = []
    for part in (text or "").split(";"):
        if part.strip():
            out.append(parse_xyz(part))
    return out


def parse_section(origin, normal):
    return parse_xyz(origin), parse_xyz(normal)


def band_json_dump(path, bands):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(bands, f, indent=2)
    return path


def find_job_files(folder):
    folder = os.path.abspath(folder)
    out = {"folder": folder, "scan": None, "solid": None, "report": None}
    names = os.listdir(folder)
    for n in names:
        if n.endswith(".report.json") or n == "report.json":
            out["report"] = os.path.join(folder, n)
            try:
                with open(out["report"], encoding="utf-8") as f:
                    rep = json.load(f)
                for key in ("step", "output", "solid"):
                    p = rep.get(key)
                    if p and os.path.isfile(p):
                        out["solid"] = p
                        break
                mesh = (rep.get("mesh") or {}).get("path") or rep.get("input") or rep.get("mesh_path")
                if mesh and os.path.isfile(mesh):
                    out["scan"] = mesh
            except (OSError, ValueError):
                pass
            break
    if out["solid"] is None:
        for n in sorted(names):
            if n.lower().endswith((".step", ".stp")):
                out["solid"] = os.path.join(folder, n)
                break
    if out["scan"] is None:
        prefer = [n for n in names if n.lower() == "mesh_body.stl"] or \
            [n for n in names if n.lower().endswith((".stl", ".obj", ".ply"))]
        if prefer:
            out["scan"] = os.path.join(folder, prefer[0])
    return out


def parse_zones(text):
    """Typed exclusion zones, checked in the dialog before the engine runs (same rules as
    meshsolid.metrology_exclude.parse_zones):
        box(xmin,ymin,zmin, xmax,ymax,zmax)      cyl(px,py,pz, dx,dy,dz, r[, t0,t1])
    several separated by ';'. Returns a list of (kind, numbers); raises ValueError."""
    import re
    num = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
    out = []
    for part in (text or "").split(";"):
        part = part.strip()
        if not part:
            continue
        m = re.match(r"\s*([A-Za-z]+)", part)
        if not m:
            raise ValueError(f"zone '{part}' must start with box or cyl")
        kind = m.group(1).lower()
        nums = [float(x) for x in num.findall(part[m.end():])]
        if kind == "box":
            if len(nums) != 6:
                raise ValueError(f"box needs 6 numbers (xmin,ymin,zmin, xmax,ymax,zmax), got {len(nums)} in '{part}'")
        elif kind in ("cyl", "cylinder"):
            if len(nums) not in (7, 9):
                raise ValueError(f"cyl needs 7 or 9 numbers (px,py,pz, dx,dy,dz, r[, t0,t1]), got {len(nums)} in '{part}'")
            if abs(nums[3]) + abs(nums[4]) + abs(nums[5]) < 1e-12:
                raise ValueError(f"cyl direction is zero in '{part}'")
            if nums[6] <= 0:
                raise ValueError(f"cyl radius must be > 0 in '{part}'")
            kind = "cyl"
        else:
            raise ValueError(f"unknown zone type '{kind}' (use box or cyl)")
        out.append((kind, nums))
    return out


# ---- honest verdict (spec v1/v2) next to the 1.0.1 engine-limit result ---------------------------------
def _verdict_colour(res):
    return "#1f8a3b" if res == "PASS" else "#b3261e" if res in ("FAIL", "REFUSED") else "#333333"


def verdict_parts(st):
    """(target_result, target_label, engine_result, engine_label, rule_text) from a metrology stats dict.
    Reports from engines older than spec v1 have no verdict: their PASS/FAIL is shown as the engine-limit result
    and the target verdict is '—' (never a bare PASS)."""
    st = st or {}
    vd = st.get("verdict") or {}
    tgt, eng = vd.get("target") or {}, vd.get("engine_limit") or {}
    ex = st.get("pct_excluded")
    ex_txt = f"{float(ex):.1f} % excluded" if ex is not None else "exclusion % unknown"
    if tgt:
        t_res, t_lab = tgt.get("result", "—"), tgt.get("label") or f"{tgt.get('result')} · {ex_txt}"
    else:
        t_res, t_lab = "—", "no target verdict (old report)"
    e_res = eng.get("result") or st.get("result") if not tgt else eng.get("result", "—")
    e_lab = eng.get("label") or f"{e_res} · {ex_txt}"
    rule = tgt.get("rule") or "rms <= 0.03 mm, >= 90 % within ±0.05 mm (beyond-cap = fail), <= 5 % excluded"
    return t_res, t_lab, e_res, e_lab, rule


def verdict_html(st, big="20pt", small="12pt"):
    """Target verdict large, the 1.0.1 engine-limit result beside it; the exclusion % is in every label."""
    import html as _h
    t_res, t_lab, e_res, e_lab, rule = verdict_parts(st)
    e_txt = "PASS refused" if e_res == "REFUSED" else e_res
    return (f"<span style='font-size:{big}; font-weight:bold; color:{_verdict_colour(t_res)}'>{_h.escape(str(t_res))}</span>"
            f"<span style='font-size:{small}'> &nbsp;target verdict · {_h.escape(t_lab.split('·', 1)[-1].strip())}</span>"
            f" &nbsp;&nbsp;|&nbsp;&nbsp; <span style='font-size:{small}; color:{_verdict_colour(e_res)}'><b>"
            f"{_h.escape(str(e_txt))}</b> engine-limit (1.0.1) · {_h.escape(e_lab.split('·', 1)[-1].strip())}</span>"
            f"<br><span style='font-size:10pt; color:#5b6470'>target: {_h.escape(rule)}</span>")
