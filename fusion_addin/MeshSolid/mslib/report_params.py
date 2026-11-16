"""report.json -> Fusion user parameters.  No adsk import (unit-testable outside Fusion).

The imported STEP is a plain B-rep: these parameters record the recognised design values (snapped
dimensions, gear, screw rotor, thread and impeller parameters) next to the body.  They do not drive its geometry.

Each spec: {"name", "expr", "units", "comment", "group", "default"}; `units` is a Fusion unit string
("mm", "deg", or "" for a count).  `default` is whether the review dialog ticks it initially."""
import re

GROUPS = ("gear", "screw", "spline", "keyway", "bearing", "seal", "thread", "impeller", "snapped", "fit")
GROUP_LABELS = {"gear": "Gear", "screw": "Screw rotor", "spline": "Splines", "keyway": "Keyways", "bearing": "Bearing", "seal": "Labyrinth seal", "thread": "Threads", "impeller": "Impeller", "snapped": "Snapped dimensions",
                "fit": "Other fitted dimensions"}
_ANGLE_TYPES = {"cone_half_angle_deg", "draft_angle_deg", "angle"}


def _pname(*parts):
    s = "_".join(re.sub(r"[^A-Za-z0-9]+", "_", str(p)).strip("_") for p in parts if p not in (None, ""))
    if not s or not s[0].isalpha():
        s = "p_" + s
    return s


def _num(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if x == x and abs(x) != float("inf") else None


def _spec(name, value, units, comment, group, default=True, digits=4):
    v = _num(value)
    if v is None:
        return None
    if units == "":
        expr = str(int(round(v))) if abs(v - round(v)) < 1e-9 else f"{v:.{digits}f}"
    else:
        expr = f"{v:.{digits}f} {units}"
    return {"name": name, "expr": expr, "units": units, "comment": comment, "group": group, "default": default}


def _dimensions(rep, prefix):
    out = []
    for d in rep.get("dimensions", []) or []:
        name, val, fit = d.get("name"), d.get("value"), d.get("fit_value")
        if name is None or _num(val) is None:
            continue
        kind = d.get("type", "dimension")
        units = "deg" if (d.get("units") == "deg" or kind in _ANGLE_TYPES) else "mm"
        snapped = _num(fit) is not None and abs(float(fit) - float(val)) > 1e-6
        comment = f"meshsolid {kind}" + (f", snapped from fit {float(fit):.4f}" if snapped else ", fitted value")
        # tori are blend radii: many and rarely design-driving, so they start unticked
        out.append(_spec(_pname(prefix + str(name)), val, units, comment, "snapped" if snapped else "fit",
                         default=snapped and not kind.startswith("torus")))
    return out


def _gear(rep, prefix):
    g = rep.get("gear") or {}
    if not g.get("detected"):
        return []
    src = g.get(g.get("build_from") or "snapped") or g.get("snapped") or g.get("free_fit") or {}
    built = g.get("build_from", "snapped")
    hand = g.get("hand")
    c = f"meshsolid gear ({built} values, {hand} hand)" if hand else f"meshsolid gear ({built} values)"
    rows = [("teeth", g.get("teeth"), "", "tooth count"),
            ("normal_DP", src.get("normal_DP"), "", "normal diametral pitch (1/in)"),
            ("normal_module", src.get("normal_module"), "mm", "normal module"),
            ("transverse_module", src.get("transverse_module"), "mm", "transverse module"),
            ("pressure_angle", src.get("normal_pressure_angle_deg", src.get("pressure_angle_normal_deg")), "deg",
             "normal pressure angle"),
            ("helix_angle", src.get("helix_angle_pitch_deg"), "deg", "helix angle at the pitch circle"),
            ("lead", src.get("lead_mm"), "mm", "lead"),
            ("pitch_dia", src.get("pitch_diameter"), "mm", "pitch diameter"),
            ("base_dia", src.get("base_diameter"), "mm", "base diameter"),
            ("tip_dia", src.get("tip_diameter_standard") if src.get("tip_snapped") else src.get("tip_diameter_crest"),
             "mm", "tip diameter"),
            ("root_dia", src.get("root_diameter_standard") if src.get("root_snapped") else
             src.get("root_diameter_measured", src.get("root_diameter")), "mm", "root diameter"),
            ("profile_shift_x", src.get("profile_shift_x"), "", "profile shift coefficient"),
            ("face_width", g.get("face_width_mm"), "mm", "face width"),
            ("span_W", (src.get("span") or {}).get("W_normal_mm"), "mm",
             f"span measurement over {(src.get('span') or {}).get('teeth_spanned')} teeth (normal)"),
            ("over_pins_M", (src.get("over_pins") or {}).get("M_over_pins_mm"), "mm",
             f"measurement over {(src.get('over_pins') or {}).get('pin_diameter_mm')} mm pins")]
    return [_spec(_pname(prefix + "gear", k), v, u, f"{c}: {t}", "gear") for k, v, u, t in rows]


def _screw(rep, prefix):
    sr = rep.get("screw_rotor") or {}
    if not sr.get("detected"):
        return []
    c = f"meshsolid screw rotor ({sr.get('kind')}, {sr.get('hand')} hand, fitted values)"
    lr = sr.get("lead_range_mm") or [None, None]
    rows = [("lobes", sr.get("lobes"), "", "lobe count"),
            ("lead", sr.get("lead_mm"), "mm", "lead (one full turn)"),
            ("lead_start", lr[0], "mm", "lead at the start end (variable pitch)"),
            ("lead_end", lr[1], "mm", "lead at the far end (variable pitch)"),
            ("wrap", sr.get("wrap_deg"), "deg", "wrap angle over the rotor length"),
            ("length", sr.get("rotor_length_mm"), "mm", "rotor (lobed) length"),
            ("tip_dia", sr.get("tip_diameter"), "mm", "outer (tip) diameter"),
            ("root_dia", sr.get("root_diameter"), "mm", "root diameter")]
    return [_spec(_pname(prefix + "screw", k), v, u, f"{c}: {t}", "screw") for k, v, u, t in rows]


def _keyways(rep, prefix):
    out = []
    for i, k in enumerate(rep.get("keyways", []) or [], 1):
        std = k.get("standard") or {}
        des = f"{std.get('standard')} {std.get('designation')}" if std else "no standard match"
        c = f"meshsolid keyway {i}: {k.get('kind')}, {k.get('ends')} ends, {des}"
        dkey = "depth_t1" if "depth_t1" in k else "depth_t2"
        snapped = bool(std) and std.get("depth_misfit_mm") is not None and std["depth_misfit_mm"] <= 0.1
        out += [_spec(_pname(prefix + f"key{i}", "width"), std.get("b") if std else k.get("width_b"), "mm", c + " (width)", "keyway"),
                _spec(_pname(prefix + f"key{i}", "depth"), std.get("depth_nominal") if snapped else k.get(dkey), "mm",
                      c + f" ({'nominal' if snapped else 'measured'} {dkey[-2:]} depth)", "keyway"),
                _spec(_pname(prefix + f"key{i}", "length"), k.get("length"), "mm", c + " (length, measured)", "keyway")]
    return out


def _splines(rep, prefix):
    out = []
    for i, sp in enumerate(rep.get("splines", []) or [], 1):
        des = f"{sp.get('standard')} {sp.get('designation')}" if sp.get("designation") else "no standard match"
        c = f"meshsolid spline {i}: {sp.get('side')} {sp.get('form')}, {des}"
        n = prefix + f"spl{i}"
        out += [_spec(_pname(n, "teeth"), sp.get("teeth"), "", c + " (teeth)", "spline"),
                _spec(_pname(n, "major_dia"), sp.get("major_diameter"), "mm", c + " (major diameter, measured)", "spline"),
                _spec(_pname(n, "minor_dia"), sp.get("minor_diameter"), "mm", c + " (minor diameter, measured)", "spline"),
                _spec(_pname(n, "length"), sp.get("length"), "mm", c + " (length, measured)", "spline")]
        if sp.get("module") is not None:
            out.append(_spec(_pname(n, "module"), sp.get("module"), "mm", c + " (module, standard)", "spline"))
        if sp.get("width_B") is not None:
            out.append(_spec(_pname(n, "width"), (sp.get("snap") or {}).get("B", sp.get("width_B")), "mm",
                             c + (" (tooth width, standard)" if sp.get("snap") else " (tooth width, measured)"), "spline"))
    return out


def _bearing_seal(rep, prefix):
    out = []
    b = rep.get("bearing") or {}
    m = b.get("match") or {}
    if m:
        c = f"meshsolid bearing boundary dimensions, matches {' / '.join(m.get('designations', []))}"
        out += [_spec(_pname(prefix + "brg", "d"), m.get("d"), "mm", c + " (bore)", "bearing"),
                _spec(_pname(prefix + "brg", "D"), m.get("D"), "mm", c + " (outside)", "bearing"),
                _spec(_pname(prefix + "brg", "B"), m.get("B"), "mm", c + " (width)", "bearing")]
    re_ = rep.get("rolling_elements") or {}
    if re_:
        c = f"meshsolid rolling elements ({re_.get('elements')}s, measured)"
        out += [_spec(_pname(prefix + "brg", "Z"), re_.get("count"), "", c + ": count", "bearing"),
                _spec(_pname(prefix + "brg", "Dw"), re_.get("diameter"), "mm", c + ": diameter", "bearing"),
                _spec(_pname(prefix + "brg", "Dpw"), re_.get("pitch_diameter"), "mm", c + ": pitch diameter", "bearing")]
    for i, l in enumerate(rep.get("labyrinth", []) or [], 1):
        c = f"meshsolid labyrinth seal {i} ({l.get('side')} teeth, measured)"
        rows = [("teeth", l.get("teeth"), "", "tooth count"), ("pitch", l.get("pitch"), "mm", "tooth pitch"),
                ("tip_dia", l.get("tip_diameter"), "mm", "tooth tip diameter"),
                ("groove_dia", l.get("groove_diameter"), "mm", "groove diameter"),
                ("tip_land", l.get("tip_land"), "mm", "tip land width"),
                ("flank", l.get("flank_angle_deg"), "deg", "flank angle from the radial")]
        out += [_spec(_pname(prefix + f"seal{i}", k), v, u_, f"{c}: {t}", "seal") for k, v, u_, t in rows]
    return out


def _threads(rep, prefix):
    out = []
    for i, t in enumerate(rep.get("threads", []) or [], 1):
        std = t.get("standard") or {}
        des = std.get("designation") if std.get("match") == "good" else None
        c = f"meshsolid thread {i}: {t.get('kind')} {t.get('hand')} hand" + (f", {des}" if des else ", no standard match")
        out += [_spec(_pname(prefix + f"thread{i}", "major_d"), std.get("nominal_major_d") if des else t.get("major_d"),
                      "mm", c + " (major diameter)", "thread"),
                _spec(_pname(prefix + f"thread{i}", "pitch"), std.get("nominal_pitch") if des else t.get("pitch"),
                      "mm", c + " (pitch)", "thread"),
                _spec(_pname(prefix + f"thread{i}", "length"), t.get("length"), "mm", c + " (length)", "thread")]
    return out


def _impeller(rep, prefix):
    imp = rep.get("impeller") or {}
    if not imp.get("detected"):
        return []
    bp = imp.get("blade_parameters") or {}
    le = imp.get("le_round_flag") or {}
    c = "meshsolid impeller"
    rows = [("blades", imp.get("blades"), "", "blade count"),
            ("bore_d", (imp.get("bore") or {}).get("d"), "mm", f"bore ({(imp.get('bore') or {}).get('source')})"),
            ("tip_r", bp.get("r_te"), "mm", "trailing-edge (tip) radius"),
            ("exit_height", bp.get("exit_blade_height"), "mm", "blade height at the exit"),
            ("le_plane_z", bp.get("le_plane_z"), "mm", "leading-edge plane height above the back face"),
            ("le_round_r", le.get("rho_mm") if le.get("rounded") else None, "mm",
             f"leading-edge round radius ({le.get('method')})")]
    for key, label in (("root_incl_fillet", "hub"), ("span_50", "mid"), ("tip", "tip")):
        s = (bp.get("spans") or {}).get(key) or {}
        rows += [(f"{label}_wrap", s.get("wrap_deg"), "deg", f"wrap angle, {key}"),
                 (f"{label}_beta_in", s.get("beta_inlet_deg"), "deg", f"inlet blade angle from meridional, {key}"),
                 (f"{label}_beta_out", s.get("beta_exit_deg"), "deg", f"exit blade angle from meridional, {key}")]
    return [_spec(_pname(prefix + "imp", k), v, u, f"{c}: {t}", "impeller") for k, v, u, t in rows]


def parameters_from_report(rep, prefix="ms_"):
    """All parameter specs for a report, de-duplicated by name, in GROUPS order."""
    out, seen = [], set()
    for s in _gear(rep, prefix) + _screw(rep, prefix) + _splines(rep, prefix) + _keyways(rep, prefix) + _bearing_seal(rep, prefix) + _threads(rep, prefix) + _impeller(rep, prefix) + _dimensions(rep, prefix):
        if s is None or s["name"] in seen:
            continue
        seen.add(s["name"]); out.append(s)
    return sorted(out, key=lambda s: GROUPS.index(s["group"]))


def by_group(specs):
    g = {}
    for s in specs:
        g.setdefault(s["group"], []).append(s)
    return g
