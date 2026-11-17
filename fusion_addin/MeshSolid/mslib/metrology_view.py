"""Zoomable metrology results page for a Fusion HTML palette (pure Python; no adsk import).

The command-dialog results window (TextBox + Image inputs) can only scroll vertically and cannot
zoom. This page is shown in a Palette instead: a fixed toolbar (zoom out / in, fit width, fit page,
100 %, zoom readout, Export PDF report..., Open full-size report, Open job folder) above a viewport
with overflow:auto (both scrollbars). Zooming scales the report image by width (so the scrollbars
track the real size) and the text/tables by font size; Ctrl + mouse wheel zooms at the cursor and a
left-drag on the image pans. Buttons talk to the add-in with adsk.fusionSendData(action, json)."""
from __future__ import annotations

import html as _H
import json
import os

ZOOM_MIN, ZOOM_MAX, ZOOM_STEP = 0.05, 4.0, 1.25
ACTIONS = ("ready", "exportPdf", "openFull", "openPng", "openFolder")


def _f(v, nd=4, signed=False):
    try:
        return f"{float(v):+.{nd}f}" if signed else f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return "—"


def _rgb(c):
    try:
        r, g, b = [int(x) for x in list(c)[:3]]
        return f"rgb({r},{g},{b})"
    except Exception:  # noqa: BLE001
        return "#999"


def summary_html(rep):
    """Headline + tables. Sizes are in em so the whole block scales with the zoom (font-size on #text)."""
    e = _H.escape
    st = rep.get("stats") or {}
    meta = rep.get("meta") or {}
    res = st.get("result") or ""
    cls = "pass" if res == "PASS" else "fail" if res == "FAIL" else ""
    out = []
    title = meta.get("title") or "Metrology inspection report"
    part = meta.get("part_name") or ""
    out.append(f"<h1>{e(title)}{(' — ' + e(part)) if part else ''}</h1>")
    from .metrology_ui_defaults import verdict_parts, warnings_html
    out.append(warnings_html(st))
    t_res, t_lab, e_res, e_lab, rule = verdict_parts(st)
    cls = "pass" if t_res == "PASS" else "fail" if t_res in ("FAIL", "REFUSED") else ""
    ecls = "pass" if e_res == "PASS" else "fail" if e_res in ("FAIL", "REFUSED") else ""
    sm = st.get("spec_metrics") or {}
    out.append(f"<div class='headline'><span class='result {cls}'>{e(str(t_res))}</span>"
               f"<span class='mid'>target verdict · {e(t_lab.split('·', 1)[-1].strip())}</span> &nbsp;|&nbsp; "
               f"<span class='mid {ecls}'><b>{e('PASS refused' if e_res == 'REFUSED' else str(e_res))}</b> engine-limit "
               f"(1.0.1) · {e(e_lab.split('·', 1)[-1].strip())}</span><br>"
               f"<span class='mid'>target: {e(rule)}</span><br>"
               + (f"<span class='big'>rms uncapped {_f(sm.get('rms_uncapped'))} mm &nbsp; p95 {_f(st.get('p95'))} mm</span><br>"
                  if sm else f"<span class='big'>rms {_f(st.get('rms'))} mm &nbsp; p95 {_f(st.get('p95'))} mm</span><br>")
               +
               f"<span class='mid'>{e(str(sm.get('pass_pct_v1', '—')))} % within ±{e(str(st.get('pass_tol_mm', '—')))} mm with "
               f"beyond-cap points as failures; {e(str(st.get('within_pass_pct', '—')))} % with them dropped"
               f" (1.0.1 needs ≥ {e(str(st.get('accept_pct', 95)))} %) &nbsp;·&nbsp; mean {_f(st.get('mean'), signed=True)} mm"
               f" &nbsp;·&nbsp; max {_f(st.get('max_plus'), 3, True)} / {_f(st.get('max_minus'), 3, True)} mm"
               + (f" (inspected), true max {_f(st.get('true_max_plus'), 3, True)} / {_f(st.get('true_max_minus'), 3, True)} mm"
                  if st.get("cap_mm") else "")
               + f" &nbsp;·&nbsp; {int(st.get('n') or 0):,} vertices inspected, {int(st.get('n_excluded') or 0):,} not inspected"
               "</span>"
               + (f"<br><span class='mid cap'><b>{int(st.get('n_beyond_cap') or 0):,}</b> points "
                  f"({_f(st.get('pct_beyond_cap'), 2)} %) beyond the {float(st['cap_mm']):g} mm outlier cap "
                  "— left out of the statistics, shown in the cap colour</span>" if st.get("cap_mm") else "")
               + "</div>")
    nin = rep.get("not_inspected") or []
    if nin:
        rows = "".join(f"<tr><td>{e(str(x.get('reason', '')))}</td><td>{e(str(x.get('label') or ''))}</td>"
                       f"<td class='n'>{int(x.get('n') or 0):,}</td><td class='n'>{int(x.get('n_new', x.get('n')) or 0):,}</td>"
                       f"<td class='detail'>{e(str(x.get('detail', '')))}</td></tr>" for x in nin)
        tot = sum(int(x.get("n_new", x.get("n")) or 0) for x in nin)
        out.append("<h2>Not inspected: where the excluded points come from</h2><table><tr><th>reason</th><th>what</th>"
                   "<th>matched</th><th>counted</th><th>detail</th></tr>"
                   f"{rows}<tr><th colspan='3'>total</th><th class='n'>{tot:,}</th><th></th></tr></table>"
                   "<div class='notes'>matched = points the exclusion covers; counted = not already excluded by an "
                   "earlier row (adds up to the total).</div>")
    z = st.get("zones") or rep.get("zones") or {}
    zrows = "".join(f"<tr><td>{e(k)}</td><td class='n'>{int(v.get('n') or 0):,}</td><td class='n'>{_f(v.get('rms'))}</td>"
                    f"<td class='n'>{_f(v.get('mean'), signed=True)}</td><td class='n'>{_f(v.get('p95'))}</td>"
                    f"<td class='n'>{_f(v.get('max_plus'), 3, True)}</td><td class='n'>{_f(v.get('max_minus'), 3, True)}</td></tr>"
                    for k, v in z.items() if isinstance(v, dict) and v)
    bands = rep.get("bands") or []
    brows = "".join(f"<tr><td><span class='sw' style='background:{_rgb(b.get('rgb'))}'></span></td>"
                    f"<td class='n'>{_f(b.get('lo'), 3, True)}</td><td class='n'>{_f(b.get('hi'), 3, True)}</td>"
                    f"<td>{'yes' if b.get('in_tol') else ''}</td><td class='n'>{_f(b.get('pct'), 2)} %</td></tr>"
                    for b in bands if b.get("lo") is not None)
    if st.get("cap_mm"):
        crgb = (rep.get("settings") or {}).get("cap_color") or "#ff0000"
        brows += (f"<tr><td><span class='sw' style='background:{e(crgb)}'></span></td><td colspan='3'>beyond cap "
                  f"|d| &gt; {float(st['cap_mm']):g}</td><td class='n'>{int(st.get('n_beyond_cap') or 0):,} "
                  f"({_f(st.get('pct_beyond_cap'), 2)} %)</td></tr>")
    out.append("<div class='cols'>")
    if zrows:
        out.append("<div><h2>Zones (mm)</h2><table><tr><th>zone</th><th>n</th><th>rms</th><th>mean</th><th>p95</th>"
                   f"<th>max +</th><th>max −</th></tr>{zrows}</table></div>")
    if brows:
        out.append("<div><h2>Colour bands (mm)</h2><table><tr><th></th><th>from</th><th>to</th><th>in tol</th>"
                   f"<th>% of inspected</th></tr>{brows}</table></div>")
    out.append("</div>")
    probes = rep.get("probes") or []
    if probes:
        prow = "".join(f"<tr><td class='n'>{i + 1}</td><td>{e(str(p.get('kind', '')))}</td>" +
                       "".join(f"<td class='n'>{float(x):.3f}</td>" for x in (p.get('xyz_mm') or [0, 0, 0])[:3]) +
                       f"<td class='n'>{_f(p.get('deviation_mm'), 4, True)}</td></tr>" for i, p in enumerate(probes[:20]))
        out.append("<h2>Worst points / probes</h2><table><tr><th>#</th><th>kind</th><th>x (mm)</th><th>y (mm)</th>"
                   f"<th>z (mm)</th><th>deviation (mm)</th></tr>{prow}</table>")
    notes = rep.get("notes") or []
    if notes:
        out.append("<h2>Notes</h2><ul class='notes'>" + "".join(f"<li>{e(str(n))}</li>" for n in notes) + "</ul>")
    return "\n".join(out)


PAGE = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>MeshSolid metrology result</title>
<style>
  html, body { margin:0; height:100%; overflow:hidden; font-family:"Segoe UI", Arial, sans-serif; background:#2b2d31; }
  #bar { position:absolute; left:0; right:0; top:0; height:46px; display:flex; align-items:center; gap:6px;
         padding:0 10px; background:#f3f4f6; border-bottom:1px solid #c9ccd1; font-size:14px; white-space:nowrap; overflow-x:auto; }
  #bar button { font-size:14px; padding:5px 11px; border:1px solid #9aa0a8; border-radius:4px; background:#fff; cursor:pointer; }
  #bar button:hover { background:#e7eefc; }
  #bar button.primary { background:#0b62d6; color:#fff; border-color:#0b62d6; }
  #zoomPct { display:inline-block; min-width:62px; text-align:center; font-weight:bold; font-size:15px; }
  #status { color:#555; margin-left:8px; overflow:hidden; text-overflow:ellipsis; }
  .sep { width:1px; height:26px; background:#c9ccd1; margin:0 6px; }
  #view { position:absolute; left:0; right:0; top:47px; bottom:0; overflow:auto; }
  #view.grab { cursor:grab; } #view.grabbing { cursor:grabbing; user-select:none; }
  #content { display:inline-block; padding:12px; min-width:calc(100% - 24px); }
  #text { background:#fff; color:#1b1d21; padding:0.8em 1.1em; border-radius:4px; font-size:15px; margin-bottom:12px; box-sizing:border-box; }
  #text h1 { font-size:1.45em; margin:0.1em 0 0.4em; } #text h2 { font-size:1.12em; margin:0.9em 0 0.3em; }
  .headline .result { font-size:2.0em; font-weight:bold; margin-right:0.5em; } .result.pass { color:#1f8a3b; } .result.fail { color:#b3261e; } .cap { color:#b3261e; }
  .headline .big { font-size:1.4em; font-weight:bold; } .headline .mid { font-size:1.0em; }
  #text table { border-collapse:collapse; font-size:0.95em; } #text th, #text td { border:1px solid #c9ccd1; padding:0.2em 0.55em; }
  #text th { background:#eef0f3; text-align:left; } #text td.n { text-align:right; font-variant-numeric:tabular-nums; }
  .cols { display:flex; flex-wrap:wrap; gap:1.6em; } .sw { display:inline-block; width:1.6em; height:0.95em; border:1px solid #555; }
  #text td.detail { font-size:0.85em; max-width:40em; }
  .notes { font-size:0.85em; color:#444; margin:0.2em 0; padding-left:1.3em; }
  #rep { display:block; image-rendering:auto; background:#fff; }
  #noimg { color:#ddd; font-size:15px; }
</style></head>
<body>
<div id="bar">
  <button id="zout" title="Zoom out (-)">&#8722;</button>
  <span id="zoomPct">100%</span>
  <button id="zin" title="Zoom in (+)">+</button>
  <button id="fitw" title="Fit width (0)">Fit width</button>
  <button id="fitp" title="Fit the whole report in the window">Fit page</button>
  <button id="z100" title="Actual pixels">100%</button>
  <span class="sep"></span>
  <button id="exportPdf" class="primary">Export PDF report...</button>
  <button id="openFull">Open full-size report</button>
  <button id="openFolder">Open job folder</button>
  <span id="status">Ctrl + wheel: zoom &nbsp;·&nbsp; drag the image: pan &nbsp;·&nbsp; Shift + wheel: left / right</span>
</div>
<div id="view" class="grab"><div id="content">
  <div id="text">/*__SUMMARY__*/</div>
  /*__IMAGE__*/
</div></div>
<script>
(function () {
  var Z_MIN = /*__ZMIN__*/, Z_MAX = /*__ZMAX__*/, STEP = /*__ZSTEP__*/;
  var view = document.getElementById('view'), img = document.getElementById('rep'), text = document.getElementById('text');
  var pct = document.getElementById('zoomPct'), status = document.getElementById('status');
  var natW = 1000, natH = 1400, z = 1, zFit = 1;
  function clamp(v, a, b) { return Math.max(a, Math.min(b, v)); }
  function fitWidthZ() { return Math.max(Z_MIN, (view.clientWidth - 30) / natW); }
  function fitPageZ() { var avail = view.clientHeight - 30 - text.offsetHeight - 12;
                        return Math.max(Z_MIN, Math.min((view.clientWidth - 30) / natW, (avail > 120 ? avail : view.clientHeight - 30) / natH)); }
  function apply() {
    var w = Math.round(natW * z);
    if (img) { img.style.width = w + 'px'; img.style.height = Math.round(natH * z) + 'px'; }
    var ts = clamp(z / zFit, 0.85, 2.5);                 // text scales with the zoom, never below readable
    text.style.fontSize = (15 * ts).toFixed(1) + 'px';
    text.style.width = Math.max(w, Math.round(view.clientWidth * 0.6 * Math.min(ts, 1.5))) + 'px';
    pct.textContent = Math.round(z * 100) + '%';
  }
  function setZoom(nz, cx, cy) {
    nz = clamp(nz, Z_MIN, Z_MAX);
    if (cx === undefined) { cx = view.clientWidth / 2; cy = view.clientHeight / 2; }
    var r = nz / z, sx = view.scrollLeft + cx, sy = view.scrollTop + cy;
    z = nz; apply();
    view.scrollLeft = sx * r - cx; view.scrollTop = sy * r - cy;
  }
  window.msZoom = { set: setZoom, get: function () { return z; } };   // for testing in a browser
  document.getElementById('zin').onclick = function () { setZoom(z * STEP); };
  document.getElementById('zout').onclick = function () { setZoom(z / STEP); };
  document.getElementById('fitw').onclick = function () { zFit = fitWidthZ(); setZoom(zFit, 0, 0); view.scrollLeft = 0; };
  document.getElementById('fitp').onclick = function () { setZoom(fitPageZ(), 0, 0); view.scrollLeft = 0;
                                                           if (img) view.scrollTop = img.offsetTop - 12; };
  document.getElementById('z100').onclick = function () { setZoom(1); };
  view.addEventListener('wheel', function (e) {
    if (!e.ctrlKey) return;                                   // plain wheel scrolls, Shift+wheel scrolls sideways
    e.preventDefault();
    var b = view.getBoundingClientRect();
    setZoom(z * (e.deltaY < 0 ? 1.12 : 1 / 1.12), e.clientX - b.left, e.clientY - b.top);
  }, { passive: false });
  document.addEventListener('keydown', function (e) {
    if (e.key === '+' || e.key === '=') { setZoom(z * STEP); e.preventDefault(); }
    else if (e.key === '-' || e.key === '_') { setZoom(z / STEP); e.preventDefault(); }
    else if (e.key === '0') { document.getElementById('fitw').onclick(); e.preventDefault(); }
  });
  var drag = null;                                            // drag to pan (on the image or empty space)
  view.addEventListener('mousedown', function (e) {
    if (e.button !== 0 || text.contains(e.target)) return;    // text stays selectable
    drag = { x: e.clientX, y: e.clientY, l: view.scrollLeft, t: view.scrollTop };
    view.className = 'grabbing'; e.preventDefault();
  });
  window.addEventListener('mousemove', function (e) {
    if (!drag) return;
    view.scrollLeft = drag.l - (e.clientX - drag.x); view.scrollTop = drag.t - (e.clientY - drag.y);
  });
  window.addEventListener('mouseup', function () { if (drag) { drag = null; view.className = 'grab'; } });
  window.addEventListener('resize', function () { var wasFit = Math.abs(z - zFit) < 1e-6; zFit = fitWidthZ();
                                                  if (wasFit) setZoom(zFit, 0, 0); else apply(); });
  function say(t) { if (t) status.textContent = String(t); }
  function send(action) {
    try {
      if (!(window.adsk && adsk.fusionSendData)) { say('Not connected to Fusion (open this page from the MeshSolid add-in).'); return; }
      var r = adsk.fusionSendData(action, JSON.stringify({ action: action }));
      if (r && typeof r.then === 'function') r.then(say); else say(r);
    } catch (err) { say('Could not reach Fusion: ' + err); }
  }
  ['exportPdf', 'openFull', 'openFolder'].forEach(function (id) {
    document.getElementById(id).onclick = function () { send(id); };
  });
  function start() { if (img && img.naturalWidth) { natW = img.naturalWidth; natH = img.naturalHeight; }
                     zFit = fitWidthZ(); z = zFit; apply(); }
  if (img && !img.complete) { img.onload = start; img.onerror = start; } else { start(); }
  if (!/*__HASPDF__*/) { document.getElementById('exportPdf').disabled = true; document.getElementById('exportPdf').title = 'No report.pdf in this job'; }
})();
</script>
</body></html>
"""


def page_html(rep):
    files = rep.get("files") or {}
    png = files.get("report_png") or ""
    if png and os.path.isfile(png):
        img = (f"<img id='rep' src='{_H.escape(os.path.basename(png), quote=True)}?t={int(os.path.getmtime(png))}' "
               "alt='metrology report' draggable='false'>")
    else:
        img = "<div id='noimg'>report.png was not written for this run.</div>"
    has_pdf = bool(files.get("report_pdf") and os.path.isfile(files["report_pdf"]))
    return (PAGE.replace("/*__SUMMARY__*/", summary_html(rep)).replace("/*__IMAGE__*/", img)
            .replace("/*__ZMIN__*/", repr(ZOOM_MIN)).replace("/*__ZMAX__*/", repr(ZOOM_MAX))
            .replace("/*__ZSTEP__*/", repr(ZOOM_STEP)).replace("/*__HASPDF__*/", "true" if has_pdf else "false"))


def write_page(rep, out_dir=None):
    """results_view.html in the job folder (next to report.png, which it references by a relative path)."""
    out_dir = out_dir or rep.get("out_dir") or os.path.dirname((rep.get("files") or {}).get("report_png") or "")
    path = os.path.join(out_dir, "results_view.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(page_html(rep))
    return path


def palette_url(path):
    """Fusion palettes take a local file path; Autodesk's own add-in template passes it with '/' separators."""
    return os.path.abspath(path).replace("\\", "/")


def parse_action(action, data):
    """HTMLEventArgs.action/.data -> action name ('' when it is not one of ours)."""
    a = (action or "").strip()
    if a not in ACTIONS:
        try:
            a = str((json.loads(data or "{}") or {}).get("action", ""))
        except (ValueError, AttributeError):
            a = ""
    return a if a in ACTIONS else ""
