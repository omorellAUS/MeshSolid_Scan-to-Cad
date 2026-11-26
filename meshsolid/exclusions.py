"""One exclusion policy for the engine and for Metrology.

Balance cuts have their own depth and total-area caps and are reported on their own line.
They are not counted in the general 5 % exclusion cap. Both sides call ``apply_policy`` /
``engine_summary`` / ``metrology_summary`` so the excluded % and the balance-cut count cannot diverge.
"""
from __future__ import annotations

import math

import numpy as np

GENERAL_CAP_PCT = 5.0
DEFAULT_BALANCE_CAPS = {"depth_mm": 5.0, "area_mm2": 2500.0}


def normalize_caps(caps=None):
    src = dict(DEFAULT_BALANCE_CAPS)
    if caps:
        for key in ("depth_mm", "area_mm2"):
            if caps.get(key) is not None:
                src[key] = float(caps[key])
    src["depth_mm"] = float(src["depth_mm"])
    src["area_mm2"] = float(src["area_mm2"])
    return src


def _n_scan_of(rep, n_scan=None):
    if n_scan:
        return int(n_scan)
    mesh = (rep or {}).get("mesh") or {}
    if mesh.get("vertices"):
        return int(mesh["vertices"])
    stored = (rep or {}).get("exclusions") or {}
    if stored.get("n_scan"):
        return int(stored["n_scan"])
    return 0


def items_from_report(rep):
    """Canonical exclusion items stored on a conversion report.

    Balance cuts come from face-feature exclusions (kind ``balance_cut``). The separate
    ``balance_cuts`` listing is used only when the face classifier did not record any,
    so the two detectors are never added together.
    """
    rep = rep or {}
    items = []
    ff = rep.get("face_features") or {}
    for i, feat in enumerate(ff.get("exclusions") or []):
        if feat.get("flagged_kept_inspected"):
            continue
        if feat.get("exclude") is False:
            continue
        kind = str(feat.get("kind") or "")
        items.append({
            "id": f"ff:{i}",
            "category": "balance_cut" if kind == "balance_cut" else "general",
            "label": feat.get("label") or kind or "end-face feature",
            "n": int(feat.get("n_points") or 0),
            "area_mm2": float(feat.get("area_mm2") or 0.0),
            "depth_mm": float(feat.get("depth_mm") or 0.0),
        })
    if not any(it["category"] == "balance_cut" for it in items):
        bal = rep.get("balance_cuts") or {}
        for plane, block in (bal.get("by_plane") or {}).items():
            for j, cut in enumerate((block or {}).get("cuts") or []):
                items.append({
                    "id": f"bal:{plane}:{j}",
                    "category": "balance_cut",
                    "label": f"balance cut {plane} @ {cut.get('angle_deg')}°",
                    "n": int(cut.get("n_points") or 0),
                    "area_mm2": float(cut.get("area_mm2") or 0.0),
                    "depth_mm": float(cut.get("depth_max_mm") or cut.get("depth_mm") or 0.0),
                })
    ex2 = rep.get("auto_exclusions_2mm") or {}
    for i, region in enumerate(ex2.get("regions") or []):
        items.append({
            "id": f"d2:{i}",
            "category": "general",
            "label": region.get("label") or "auto-exclusion |d| > 2 mm",
            "n": int(region.get("n_points") or 0),
            "area_mm2": float(region.get("area_mm2") or 0.0),
            "depth_mm": float(region.get("depth_max_mm") or 0.0),
        })
    bore = ((rep.get("impeller") or {}).get("bore") or {})
    if bore.get("d") and str(bore.get("source") or "").startswith("measure"):
        items.append({
            "id": "hand:bore",
            "category": "general",
            "label": f"bore Ø{float(bore['d']):g} mm (hand-entered)",
            "n": int(bore.get("n_points") or 0),
            "area_mm2": float(bore.get("area_mm2") or 0.0),
            "depth_mm": 0.0,
        })
    return items


def apply_policy(items, n_scan, caps=None, general_cap_pct=GENERAL_CAP_PCT):
    """Split items into general exclusions and balance cuts.

    Accepted balance cuts are listed on their own line and are omitted from ``pct_excluded``.
    A cut deeper than ``depth_mm``, or a cut that would take the total area over ``area_mm2``,
    is flagged and is not an accepted balance cut. General items (everything else) use the 5 % cap.
    """
    caps = normalize_caps(caps)
    n_scan = int(n_scan or 0)
    denom = max(n_scan, 1)
    general = [it for it in items or [] if it.get("category") != "balance_cut"]
    balance = [it for it in items or [] if it.get("category") == "balance_cut"]
    balance = sorted(balance, key=lambda it: (float(it.get("area_mm2") or 0.0), str(it.get("id"))))
    accepted, flagged = [], []
    area = 0.0
    for it in balance:
        depth = float(it.get("depth_mm") or 0.0)
        a = float(it.get("area_mm2") or 0.0)
        if depth > caps["depth_mm"] + 1e-9 or area + a > caps["area_mm2"] + 1e-9:
            flagged.append(it)
        else:
            accepted.append(it)
            area += a
    n_general = sum(int(it.get("n") or 0) for it in general)
    n_accepted = sum(int(it.get("n") or 0) for it in accepted)
    pct = 100.0 * n_general / denom
    depth_max = max([float(it.get("depth_mm") or 0.0) for it in accepted], default=0.0)
    bal_pct = 100.0 * n_accepted / denom
    over = bool(pct > general_cap_pct)
    line = (f"balance cuts: {len(accepted)} accepted, {n_accepted} points ({bal_pct:.2f} % of the scan), "
            f"{area:.1f} mm², max depth {depth_max:.2f} mm"
            + (f"; {len(flagged)} over the balance-cut cap and kept inspected" if flagged else "")
            + f"; own cap depth <= {caps['depth_mm']:g} mm, total area <= {caps['area_mm2']:g} mm²"
            + "; not counted in the 5 % exclusion cap")
    general_line = (f"exclusions (not balance cuts): {pct:.2f} % of the scan "
                    f"(cap {general_cap_pct:g} %)"
                    + (" — over the cap" if over else ""))
    return {
        "spec_version": "v3",
        "n_scan": n_scan,
        "general_cap_pct": float(general_cap_pct),
        "caps": caps,
        "pct_excluded": round(pct, 4),
        "n_excluded_general": int(n_general),
        "over_general_cap": over,
        "over_general_cap_flag": (f"excluded {pct:.2f} % of the scan, above the {general_cap_pct:g} % limit"
                                  if over else None),
        "accepted_ids": [it.get("id") for it in accepted],
        "flagged_ids": [it.get("id") for it in flagged],
        "balance_cuts": {
            "count": len(accepted),
            "flagged": len(flagged),
            "n_points": int(n_accepted),
            "area_mm2": round(area, 3),
            "depth_max_mm": round(depth_max, 3),
            "pct_of_scan": round(bal_pct, 4),
            "over_cap": bool(flagged),
            "cap_depth_mm": caps["depth_mm"],
            "cap_area_mm2": caps["area_mm2"],
            "counted_against": "own cap (not the 5 % exclusion cap)",
            "line": line,
        },
        "lines": [general_line, line],
    }


def engine_summary(rep, n_scan, caps=None):
    """Engine-side tally. The returned dict is what gets stored on the report."""
    items = items_from_report(rep)
    out = apply_policy(items, n_scan, caps=caps)
    out["items"] = items
    return out


def metrology_summary(rep, n_scan=None, caps=None):
    """Metrology-side tally. Stored engine items are re-measured with the same policy."""
    stored = (rep or {}).get("exclusions") or {}
    items = stored.get("items")
    if items:
        n = _n_scan_of(rep, n_scan or stored.get("n_scan"))
        return apply_policy(items, n, caps=caps or stored.get("caps"))
    n = _n_scan_of(rep, n_scan)
    return apply_policy(items_from_report(rep), n, caps=caps)


def publish(rep, n_scan, caps=None):
    """Write the shared tally onto a conversion report and align the verdict's exclusion %."""
    summary = engine_summary(rep, n_scan, caps=caps)
    rep["exclusions"] = summary
    rep["spec_version"] = "v3"
    _apply_pct_to_verdict(rep.get("verdict"), summary["pct_excluded"], summary["general_cap_pct"])
    bal = rep.get("balance_cuts")
    if isinstance(bal, dict):
        bal["line"] = summary["balance_cuts"]["line"]
        bal["policy"] = summary["balance_cuts"]
    return summary


def _apply_pct_to_verdict(vd, pct, limit):
    if not isinstance(vd, dict):
        return
    pct_r = round(float(pct), 4)
    tgt = vd.get("target")
    if isinstance(tgt, dict):
        tgt["pct_excluded"] = round(pct_r, 3)
        tgt["spec_version"] = "v3"
        checks = tgt.get("checks")
        if isinstance(checks, list):
            for check in checks:
                if check.get("name") == "pct_excluded":
                    check["value"] = round(pct_r, 3)
                    check["ok"] = pct_r <= float(check.get("limit", limit))
            tgt["failed"] = [c["name"] for c in checks if not c.get("ok")]
            tgt["result"] = "PASS" if not tgt["failed"] else "FAIL"
            tgt["label"] = f"{tgt['result']} · {pct_r:.1f} % excluded"
    eng = vd.get("engine_limit")
    if isinstance(eng, dict):
        eng["pct_excluded"] = round(pct_r, 3)
        res = eng.get("result") or ""
        if res:
            eng["label"] = f"{res} · {pct_r:.1f} % excluded"
    metrics = vd.get("metrics")
    if isinstance(metrics, dict):
        metrics["pct_excluded"] = pct_r


def bore_cylinder_volume_mm3(d_mm, length_mm):
    return math.pi * (float(d_mm) / 2.0) ** 2 * abs(float(length_mm))


def hand_entered_volume_mm3(features, length_mm=None):
    """Volume of hand-entered features to subtract from the mesh (a design bore is a cylinder)."""
    total = 0.0
    for feat in features or []:
        if feat.get("volume_mm3") is not None and feat.get("kind") != "bore":
            total += float(feat["volume_mm3"])
            continue
        if feat.get("kind") == "bore" and feat.get("d_mm") is not None:
            length = feat.get("length_mm", length_mm)
            if length:
                total += bore_cylinder_volume_mm3(feat["d_mm"], length)
    return total


def volume_error_pct(solid_mm3, mesh_mm3, hand_entered_mm3=0.0):
    """100 * (V_solid - (V_mesh - V_hand)) / |V_mesh - V_hand|.

    Hand-entered material that the scan still contains (the design bore) is subtracted from the
    mesh volume so the bore is not a volume error.
    """
    if solid_mm3 is None or mesh_mm3 is None:
        return None
    base = float(mesh_mm3) - float(hand_entered_mm3 or 0.0)
    if abs(base) < 1e-9:
        return None
    return round(100.0 * (float(solid_mm3) - base) / abs(base), 4)


def volume_check(solid_mm3, mesh_mm3, hand_entered_mm3=0.0):
    he = float(hand_entered_mm3 or 0.0)
    err = volume_error_pct(solid_mm3, mesh_mm3, he)
    base = None if mesh_mm3 is None else round(float(mesh_mm3) - he, 4)
    return {
        "solid_mm3": None if solid_mm3 is None else round(float(solid_mm3), 4),
        "mesh_mm3": None if mesh_mm3 is None else round(float(mesh_mm3), 4),
        "hand_entered_mm3": round(he, 4),
        "mesh_adjusted_mm3": base,
        "error_pct": err,
        "note": "hand-entered feature volume (for example a design bore) is subtracted from the mesh "
                "volume before the error %",
    }


def mask_on_cylinders(points, zones):
    """True where a sample lies inside a hand-entered cylindrical feature (the design bore)."""
    pts = np.asarray(points, float)
    out = np.zeros(len(pts), bool)
    if len(pts) == 0:
        return out
    for zone in zones or []:
        if (zone or {}).get("type") != "cyl":
            continue
        reason = zone.get("reason") or ""
        kind = zone.get("kind") or ""
        if reason != "hand-entered feature" and kind != "bore":
            continue
        axis = np.asarray(zone.get("dir") or [0, 0, 1], float)
        norm = float(np.linalg.norm(axis))
        if norm < 1e-12:
            continue
        axis = axis / norm
        origin = np.asarray(zone.get("point") or [0, 0, 0], float)
        rel = pts - origin
        along = rel @ axis
        radial = np.linalg.norm(rel - along[:, None] * axis, axis=1)
        on = radial <= float(zone.get("r") or 0.0)
        span = zone.get("t")
        if isinstance(span, (list, tuple)) and len(span) == 2 and span[0] is not None and span[1] is not None:
            on = on & (along >= float(span[0])) & (along <= float(span[1]))
        out |= on
    return out
