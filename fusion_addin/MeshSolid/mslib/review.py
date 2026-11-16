"""Review dialog content built from the engine report.  No adsk import.

review_model() collects what the user should see before importing: the outcome, the deviation
against its limits, everything the engine left out or flagged (sharp LE, balance cuts, unscanned
areas, excluded faces, warnings) and the parameter groups on offer.  review_html() renders it for a
Fusion TextBoxCommandInput (which shows basic HTML)."""
import html

from .outcome import deviation_line
from .report_params import GROUP_LABELS, by_group, parameters_from_report


def review_model(rep, outcome):
    m = {"title": outcome["title"], "importable": outcome["importable"], "facts": [], "flags": [], "notes": []}
    b = (rep or {}).get("stages", {}).get("build", {})
    imp = (rep or {}).get("impeller") or {}
    if imp.get("detected"):
        ib = imp.get("build") or {}
        m["facts"].append(f"impeller, {imp.get('blades')} blades: {ib.get('n_solids')} solid, "
                          f"{ib.get('n_faces')} faces, volume {ib.get('volume', 0):.0f} mm3")
    elif b:
        m["facts"].append(f"build {b.get('method')}: {b.get('n_solids')} solid, {b.get('n_faces')} faces, "
                          f"volume {b.get('volume', 0):.1f} mm3 (mesh {((rep or {}).get('mesh') or {}).get('volume')})")
    dl = deviation_line(rep or {})
    if dl:
        m["facts"].append(dl)
    g = (rep or {}).get("gear") or {}
    if g.get("detected"):
        m["facts"].append(f"gear: {g.get('teeth')} teeth, {g.get('hand')} hand, built from {g.get('build_from')} values")
    for t in (rep or {}).get("threads", []) or []:
        std = (t.get("standard") or {}).get("designation")
        m["facts"].append(f"thread: {t.get('kind')} {std or ''} pitch {t.get('pitch', 0):.3f}, "
                          f"modelled as {t.get('modelled_as')}")
    le = imp.get("le_round_flag")
    if le is not None and not le.get("rounded"):
        m["flags"].append("leading edges are sharp: the LE round could not be built")
    bc = (rep or {}).get("balance_cuts")
    if isinstance(bc, dict) and bc.get("count"):
        m["flags"].append(f"{bc['count']} balance cut(s) found; built to the nominal surface, not cut in")
    isf = (rep or {}).get("internal_scan_faces")
    if isinstance(isf, dict) and isf.get("faces"):
        m["flags"].append(f"{isf['faces']} internal scan faces ({isf.get('area_mm2')} mm2) left out of the "
                          "deviation check")
    m["flags"] += [str(w) for w in (rep or {}).get("warnings", []) or []]
    m["notes"] = [str(n) for n in imp.get("notes", []) or []] + [str(n) for n in (rep or {}).get("notes", []) or []]
    specs = parameters_from_report(rep or {})
    m["param_groups"] = [{"group": k, "label": GROUP_LABELS[k], "count": len(v),
                          "default": any(s["default"] for s in v)} for k, v in by_group(specs).items()]
    m["params"] = specs
    if not outcome["importable"]:
        m["flags"].insert(0, outcome["message"])
    return m


def review_html(m):
    e = html.escape
    parts = [f"<b>{e(m['title'])}</b>"]
    if m["facts"]:
        parts.append("<br>".join(e(f) for f in m["facts"]))
    if m["flags"]:
        parts.append("<b>Check:</b><br>" + "<br>".join("- " + e(f).replace("\n", "<br>") for f in m["flags"]))
    if m["notes"]:
        parts.append("<b>Notes:</b><br>" + "<br>".join("- " + e(n) for n in m["notes"]))
    if m.get("param_groups"):
        parts.append("<b>User parameters on offer:</b><br>" +
                     "<br>".join(f"- {e(g['label'])}: {g['count']}" for g in m["param_groups"]))
    return "<br><br>".join(parts)
