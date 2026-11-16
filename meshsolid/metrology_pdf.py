"""Multi-page A4 metrology inspection report (PDF) from a metrology report dict.

matplotlib's PdfPages only (already an engine dependency; no reportlab). Pages (fonts: title 20 pt,
headings 14 pt, tables 11.5-12 pt; raster content at 200 dpi):
  1  header (title, part, date/time, scan, solid, MeshSolid version, operator, units), PASS/FAIL, summary
  2  deviation map: three shaded views (two per row) and the legend
  3  per-zone statistics and histogram
  4  colour band table (limits, colour, in-tolerance, % of inspected vertices)
  5  probes table (x, y, z, deviation)
  6  section plot (when enabled) and the regions not inspected, with reasons
  7  settings appendix and notes
Every page has a footer with the page number.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

A4 = (8.27, 11.69)
INK, MUTED, RULE = "#1b1f24", "#5b6470", "#c9ced6"
PASS_C, FAIL_C = "#1f8a3b", "#b3261e"


def _version():
    try:
        from . import __version__
        return __version__
    except Exception:  # noqa: BLE001
        return "?"


def _fmt(v, nd=4, unit=" mm", signed=False):
    if v is None:
        return "—"
    try:
        return (f"{float(v):+.{nd}f}" if signed else f"{float(v):.{nd}f}") + unit
    except (TypeError, ValueError):
        return str(v)


def _short(path, n=70):
    s = str(path or "—")
    return s if len(s) <= n else "…" + s[-(n - 1):]


def _hex(h):
    h = str(h).lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _table(ax, rows, col_widths, header=None, fontsize=8, row_h=1.25, colors=None, header_color="#e8ebef"):
    ax.axis("off")
    if not rows:
        ax.text(0, 0.95, "—", fontsize=fontsize, color=MUTED, va="top", transform=ax.transAxes)
        return None
    t = ax.table(cellText=rows, colLabels=header, colWidths=col_widths, loc="upper left", cellLoc="left")
    t.auto_set_font_size(False)
    t.set_fontsize(fontsize)
    t.scale(1, row_h)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor(RULE)
        cell.set_linewidth(0.5)
        if header and r == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(weight="bold", color=INK)
        elif colors and (r - (1 if header else 0), c) in colors:
            cell.set_facecolor(colors[(r - (1 if header else 0), c)])
    return t


def _page(fig_list, title, meta):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=A4, dpi=200)
    fig.text(0.06, 0.962, title, fontsize=20, fontweight="bold", color=INK)
    fig.text(0.94, 0.962, meta.get("part_name") or "", fontsize=13, color=MUTED, ha="right")
    fig.add_artist(plt.Line2D([0.06, 0.94], [0.955, 0.955], color=RULE, lw=0.8))
    fig_list.append(fig)
    return fig


def _section_title(fig, y, text, size=14):
    fig.text(0.06, y, text, fontsize=size, fontweight="bold", color=INK)


def write_pdf(report, out_dir, dv=None, section=None, path=None):
    """Write report.pdf into out_dir (or path). dv: inspected signed deviations for the histogram.
    section: the section dict from metrology.section_points (u_mm, v_mm, deviation_mm) or None."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.patches import Rectangle
    from PIL import Image

    out_dir = Path(out_dir)
    path = Path(path) if path else out_dir / "report.pdf"
    meta = dict(report.get("meta") or {})
    st = report.get("stats") or {}
    sc = report.get("scale") or {}
    bands = report.get("bands") or []
    pass_tol = float(sc.get("pass_tol_mm", st.get("pass_tol_mm", 0.05)) or 0.05)
    vmin, vmax = float(sc.get("vmin", -0.2)), float(sc.get("vmax", 0.2))
    title = meta.get("title") or "Metrology inspection report"
    part = meta.get("part_name") or Path(str(report.get("solid") or "part")).stem
    meta["part_name"] = part
    when = time.strftime("%Y-%m-%d %H:%M %Z") if not meta.get("datetime") else meta["datetime"]
    figs = []
    H, T, TS = 14, 11.5, 10.5          # section heading, table, small-table font sizes (pt)

    # ---- page 1: header, summary, PASS/FAIL ----------------------------------------------------------
    fig = _page(figs, title, meta)
    hdr = [["Part", part], ["Date / time (local)", when], ["Operator", meta.get("operator") or "—"],
           ["Scan", _short(report.get("scan"), 52)], ["Solid", _short(report.get("solid"), 52)],
           ["Software", f"MeshSolid {meta.get('version') or _version()}"], ["Units", "millimetres (mm)"]]
    ax = fig.add_axes([0.06, 0.66, 0.88, 0.26])
    _table(ax, hdr, [0.3, 0.7], fontsize=T, row_h=2.0)
    # two verdicts side by side: the honest target verdict (left, large) and the 1.0.1 engine-limit rule (right)
    vd = st.get("verdict") or {}
    tgt, eng = vd.get("target") or {}, vd.get("engine_limit") or {}
    result = st.get("result") or "—"
    def _c(r):
        return PASS_C if r == "PASS" else FAIL_C if r in ("FAIL", "REFUSED") else MUTED
    fig.patches.append(Rectangle((0.06, 0.53), 0.53, 0.10, transform=fig.transFigure, facecolor=_c(result),
                                 edgecolor="none"))
    fig.text(0.325, 0.603, result, fontsize=30, fontweight="bold", color="white", ha="center", va="center")
    ex = st.get("pct_excluded")
    fig.text(0.325, 0.568, "target verdict" + (f" · {ex:.1f} % excluded" if ex is not None else ""),
             fontsize=T, color="white", ha="center", va="center")
    fig.text(0.325, 0.545, (tgt.get("rule") or f"≥ {st.get('accept_pct', 95):g} % within ±{pass_tol:g} mm"),
             fontsize=7.5, color="white", ha="center", va="center")
    er = eng.get("result") or "—"
    fig.patches.append(Rectangle((0.61, 0.53), 0.33, 0.10, transform=fig.transFigure, facecolor="white",
                                 edgecolor=_c(er), linewidth=2.0))
    fig.text(0.775, 0.603, "PASS refused" if er == "REFUSED" else er, fontsize=18, fontweight="bold", color=_c(er),
             ha="center", va="center")
    fig.text(0.775, 0.568, "engine-limit result (1.0.1 rule)" + (f" · {ex:.1f} % excl." if ex is not None else ""),
             fontsize=8.5, color=INK, ha="center", va="center")
    fig.text(0.775, 0.545, f"≥ {st.get('accept_pct', 95):g} % within ±{pass_tol:g} mm, beyond-cap dropped",
             fontsize=7.5, color=MUTED, ha="center", va="center")
    _section_title(fig, 0.495, "Summary", H)
    nin = report.get("not_inspected") or []
    sm = st.get("spec_metrics") or {}
    rows = [["rms uncapped (verdict)", _fmt(sm.get("rms_uncapped"))],
            [f"within ±{pass_tol:g} mm, beyond-cap = fail (verdict)", f"{sm.get('pass_pct_v1', 0):.2f} %"],
            ["excluded (named reasons)", f"{sm.get('pct_excluded', 0):.2f} % of the scan"],
            ["rms" + (" (beyond-cap dropped)" if st.get("cap_mm") else ""), _fmt(st.get("rms"))],
            ["mean (signed)", _fmt(st.get("mean"), signed=True)],
            ["p95 |d|", _fmt(st.get("p95"))],
            ["max + / max − (inspected)", f"{_fmt(st.get('max_plus'), signed=True)} / {_fmt(st.get('max_minus'), signed=True)}"],
            [f"within ±{pass_tol:g} mm (beyond-cap dropped, 1.0.1)", f"{st.get('within_pass_pct', 0):.2f} %"]]
    if st.get("cap_mm"):
        rows += [["true max + / max − (all measured)",
                  f"{_fmt(st.get('true_max_plus'), signed=True)} / {_fmt(st.get('true_max_minus'), signed=True)}"],
                 [f"beyond outlier cap |d| > {st['cap_mm']:g} mm",
                  f"{st.get('n_beyond_cap', 0):,} points ({st.get('pct_beyond_cap', 0):.2f} %), not in the stats"]]
    rows += [["vertices inspected", f"{st.get('n', 0):,}"],
             ["vertices not inspected", f"{st.get('n_excluded', 0):,}"],
             ]
    for r_, n_ in (st.get("excluded_by_reason") or {}).items():
        rows.append([f"   – {r_}", f"{n_:,} not inspected"])
    ax = fig.add_axes([0.06, 0.13, 0.88, 0.345])
    _table(ax, rows, [0.42, 0.58], fontsize=12 if len(rows) <= 8 else 11, row_h=2.1 if len(rows) <= 8 else 1.75)
    fig.text(0.06, 0.09, "Signed deviation, scan to solid: positive = scan outside the solid (material the solid lacks);\n"
             "negative = scan inside the solid. Details of excluded regions: page 'Not inspected'.",
             fontsize=9, color=MUTED, va="top")

    # ---- page 2: deviation map (2 views per row) + legend ------------------------------------------
    fig = _page(figs, title, meta)
    _section_title(fig, 0.925, "Deviation map", H)
    vl = report.get("views") or {}
    views = [(k, vl.get(k, d)) for k, d in (("iso", "isometric"), ("end", "end view (along the rotation axis)"),
                                            ("side", "side view (across the axis)"))]
    boxes = [[0.04, 0.50, 0.46, 0.40], [0.50, 0.50, 0.46, 0.40], [0.04, 0.07, 0.46, 0.40]]
    for (k, lab), bx in zip(views, boxes):
        pth = out_dir / f"heatmap_{k}.png"
        a = fig.add_axes(bx)
        a.axis("off")
        if pth.is_file():
            a.imshow(Image.open(pth))
        a.set_title(lab, fontsize=12, color=INK)
    from .metrology import draw_legend
    a = fig.add_axes([0.58, 0.10, 0.38, 0.34])
    cap_rgb = _hex(sett_cap) if (sett_cap := (report.get("settings") or {}).get("cap_color")) else (255, 0, 0)
    draw_legend(a, report.get("bands") or [], st.get("pass_tol_mm", 0.05), fontsize=12,
                n_excluded=st.get("n_not_inspected", st.get("n_excluded")), cap=st.get("cap_mm"),
                n_cap=st.get("n_beyond_cap"), cap_rgb=cap_rgb)
    fig.text(0.06, 0.05, "Grey = not inspected (excluded). The views show the scan coloured by its deviation.",
             fontsize=10, color=MUTED)

    # ---- page 3: zones + histogram -----------------------------------------------------------------
    fig = _page(figs, title, meta)
    _section_title(fig, 0.925, "Per-zone statistics (mm)", H)
    z = st.get("zones") or report.get("zones") or {}
    zrows = [[k, f"{v['n']:,}", _fmt(v.get("rms"), unit=""), _fmt(v.get("mean"), unit="", signed=True),
              _fmt(v.get("p95"), unit=""), _fmt(v.get("max_plus"), unit="", signed=True),
              _fmt(v.get("max_minus"), unit="", signed=True)] for k, v in z.items() if v]
    ax = fig.add_axes([0.06, 0.62, 0.88, 0.28])
    if zrows:
        _table(ax, zrows, [0.18, 0.13, 0.13, 0.15, 0.13, 0.14, 0.14],
               header=["zone", "vertices", "rms", "mean", "p95", "max +", "max −"], fontsize=T, row_h=2.0)
    else:
        ax.axis("off")
        ax.text(0, 0.9, "Not available: the part was not recognised as an impeller or gear,\n"
                "or zone statistics were turned off.", fontsize=T, color=MUTED, va="top")
    _section_title(fig, 0.56, "Histogram (inspected vertices)", H)
    ax = fig.add_axes([0.12, 0.12, 0.82, 0.40])
    if dv is not None and len(dv):
        bins = np.linspace(vmin, vmax, 41)
        ax.hist(np.clip(dv, vmin, vmax), bins=bins, color="#4a63a8", edgecolor="white", linewidth=0.3)
        ax.axvspan(-pass_tol, pass_tol, color="#2e9e4f", alpha=0.18, label=f"pass band ±{pass_tol:g} mm")
        ax.axvline(0, color="0.2", lw=1.0)
        ax.set_xlim(vmin, vmax)
        ax.set_xlabel("signed deviation (mm), clipped to the colour scale", fontsize=12)
        ax.set_ylabel("vertices", fontsize=12)
        ax.tick_params(labelsize=11)
        ax.legend(fontsize=11, loc="upper right")
        tp, tm = st.get("true_max_plus", st.get("max_plus")), st.get("true_max_minus", st.get("max_minus"))
        if tp is not None and tm is not None:
            txt = f"true extremes {tp:+.3f} / {tm:+.3f} mm"
            if st.get("cap_mm"):
                txt += f"\n{st.get('n_beyond_cap', 0):,} beyond the {st['cap_mm']:g} mm cap (not in the histogram)"
            ax.text(0.01, 0.97, txt, transform=ax.transAxes, fontsize=10, va="top", ha="left",
                    bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.9))
    else:
        ax.axis("off")
        ax.text(0, 0.5, "no inspected vertices", color=MUTED, fontsize=T)

    # ---- page 4: colour bands ----------------------------------------------------------------------
    fig = _page(figs, title, meta)
    _section_title(fig, 0.925, "Colour bands", H)
    brows, bcols = [], {}
    for i, b in enumerate(bands):
        brows.append([str(i + 1), _fmt(b["lo"], 3, "", True), _fmt(b["hi"], 3, "", True), "",
                      "yes" if b.get("in_tol") else "", f"{b.get('pct', 0):.2f} %"])
        bcols[(i, 3)] = tuple(c / 255 for c in b["rgb"])
    if st.get("cap_mm"):
        brows.append(["cap", f"|d| > {st['cap_mm']:g}", "", "", "",
                      f"{st.get('n_beyond_cap', 0):,} ({st.get('pct_beyond_cap', 0):.2f} % of measured)"])
        bcols[(len(brows) - 1, 3)] = tuple(c / 255 for c in cap_rgb)
    n_ni = st.get("n_not_inspected", st.get("n_excluded"))
    if n_ni:
        brows.append(["—", "not inspected", "", "", "", f"{n_ni:,} vertices"])
        bcols[(len(brows) - 1, 3)] = (150 / 255, 150 / 255, 154 / 255)
    ax = fig.add_axes([0.06, 0.40, 0.88, 0.50])
    _table(ax, brows, [0.1, 0.18, 0.18, 0.14, 0.14, 0.26],
           header=["band", "from (mm)", "to (mm)", "colour", "in tol", "% of inspected"],
           fontsize=12, row_h=2.1, colors=bcols)

    # ---- page 5: probes ----------------------------------------------------------------------------
    fig = _page(figs, title, meta)
    _section_title(fig, 0.925, "Point probes", H)
    probes = report.get("probes") or []
    prows = [[str(i + 1), p["kind"], f"{p['xyz_mm'][0]:.3f}", f"{p['xyz_mm'][1]:.3f}", f"{p['xyz_mm'][2]:.3f}",
              f"{p['deviation_mm']:+.4f}"] for i, p in enumerate(probes[:20])]
    ax = fig.add_axes([0.06, 0.10, 0.88, 0.80])
    _table(ax, prows, [0.08, 0.14, 0.18, 0.18, 0.18, 0.24],
           header=["#", "kind", "x (mm)", "y (mm)", "z (mm)", "deviation (mm)"], fontsize=T, row_h=1.9)

    # ---- page 6: section + not inspected ------------------------------------------------------------
    fig = _page(figs, title, meta)
    _section_title(fig, 0.925, "Section", H)
    ax = fig.add_axes([0.10, 0.60, 0.84, 0.30])
    if section and section.get("n"):
        from .metrology import plot_section
        plot_section(fig, ax, section, vmin, vmax, fontsize=11, point_size=3, cap=st.get("cap_mm"), cap_rgb=cap_rgb)
        o, nn = section.get("origin"), section.get("normal")
        fig.text(0.06, 0.535, f"plane origin ({o[0]:.2f}, {o[1]:.2f}, {o[2]:.2f}) mm,  normal ({nn[0]:.3f}, "
                 f"{nn[1]:.3f}, {nn[2]:.3f})", fontsize=10, color=MUTED, va="top")
    else:
        ax.axis("off")
        ax.text(0, 0.9, "Section not requested.", fontsize=T, color=MUTED, va="top")
    _section_title(fig, 0.50, "Not inspected: where the excluded points come from", H)
    import textwrap
    if nin:
        n_scan = int(st.get("n", 0)) + int(st.get("n_excluded", 0))
        def _area(e):
            a = e.get("area_mm2")
            return f"{a:.1f}" if a is not None else "—"
        erows = [[e.get("reason", ""), _short(e.get("label") or e.get("reason", ""), 28),
                  _area(e), f"{e['n']:,}",
                  f"{e.get('n_new', e['n']):,}", f"{100.0 * e.get('n_new', e['n']) / max(n_scan, 1):.2f} %"]
                 for e in nin]
        tot_new = sum(e.get("n_new", e["n"]) for e in nin)
        tot_area = sum(e.get("area_mm2") or 0.0 for e in nin)
        erows.append(["total", "", f"{tot_area:.1f}" if tot_area else "",
                      f"{sum(e['n'] for e in nin):,}", f"{tot_new:,}",
                      f"{100.0 * tot_new / max(n_scan, 1):.2f} %"])
        hgt = min(0.031 * (len(erows) + 1), 0.30)
        ax = fig.add_axes([0.06, 0.465 - hgt, 0.88, hgt])
        _table(ax, erows, [0.24, 0.28, 0.10, 0.12, 0.12, 0.14],
               header=["reason", "what", "area mm²", "matched", "counted", "% of scan"], fontsize=8.5, row_h=1.7)
        y = 0.465 - hgt - 0.02
        for line in textwrap.wrap("'matched' = points the exclusion covers; 'counted' = points not already excluded "
                                  "by an earlier row, so 'counted' adds up to the total.", 100):
            fig.text(0.06, y, line, fontsize=9, color=MUTED, va="top")
            y -= 0.019
        if st.get("excluded_over_5pct") or (report.get("auto_exclusions_2mm") or {}).get("over_5pct"):
            flag = ((report.get("auto_exclusions_2mm") or {}).get("over_5pct_flag")
                    or f"excluded {st.get('pct_excluded', 0):.2f} % of the scan, above the 5 % limit")
            for line in textwrap.wrap("FLAG: " + flag, 100):
                fig.text(0.06, y, line, fontsize=9, color=INK, va="top", fontweight="bold")
                y -= 0.019
        y -= 0.012
        for e in nin:
            if y < 0.08:
                break
            for j, line in enumerate(textwrap.wrap(f"{e.get('label') or e['reason']}: {e['detail']}", 100)):
                fig.text(0.06 if j == 0 else 0.08, y, line, fontsize=9, color=INK, va="top")
                y -= 0.019
            y -= 0.006
    else:
        fig.text(0.06, 0.465, "Every scan vertex was inspected.", fontsize=T, color=MUTED, va="top")

    # ---- page 7: settings appendix + notes ----------------------------------------------------------
    fig = _page(figs, title, meta)
    _section_title(fig, 0.925, "Appendix: settings", H)
    sett = report.get("settings") or {}
    srows = [[k.replace("_", " "), _short(v if not isinstance(v, (list, dict)) else str(v), 56)]
             for k, v in sett.items()]
    srows += [["solid kind", str(report.get("solid_kind"))], ["device", str(report.get("device_resolved"))],
              ["conversion report", _short(report.get("convert_report"), 56)],
              ["computation time", f"{report.get('seconds', 0)} s"]]
    ax = fig.add_axes([0.06, 0.32, 0.88, 0.58])
    _table(ax, srows, [0.34, 0.66], fontsize=TS, row_h=1.6)
    _section_title(fig, 0.29, "Notes", H)
    notes = report.get("notes") or []
    txt = "\n".join("• " + "\n   ".join(textwrap.wrap(str(n), 95)) for n in notes[:8])
    fig.text(0.06, 0.265, txt or "—", fontsize=9, color=INK, va="top")

    n = len(figs)
    path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(path) as pdf:
        d = pdf.infodict()
        d["Title"] = f"{title}: {part}"
        d["Author"] = meta.get("operator") or "MeshSolid"
        d["Subject"] = "Scan-to-solid metrology"
        d["Creator"] = f"MeshSolid {meta.get('version') or _version()}"
        for i, f in enumerate(figs):
            f.add_artist(plt.Line2D([0.06, 0.94], [0.035, 0.035], color=RULE, lw=0.6))
            f.text(0.06, 0.02, f"MeshSolid {meta.get('version') or _version()}  ·  {title}  ·  {part}  ·  {when}",
                   fontsize=8, color=MUTED)
            f.text(0.94, 0.02, f"Page {i + 1} of {n}", fontsize=10, color=MUTED, ha="right")
            pdf.savefig(f, dpi=200)
            plt.close(f)
    return str(path)
