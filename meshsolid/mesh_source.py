"""Declared mesh source (``scan`` or ``cad_export``).

The engine classifier is a cross-check only. It can warn when it disagrees with the declared
source. It does not choose the acceptance gate and it does not tighten one.
"""
from __future__ import annotations

import csv
from pathlib import Path

SCAN_CLASSES = {"scan", "decimated"}
CAD_CLASSES = {"cad_tessellation", "lowres_tessellation", "cad_export"}


def normalize_source(value):
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"scan", "cad_export"}:
        return text
    return None


def classifier_family(class_name):
    if class_name in SCAN_CLASSES:
        return "scan"
    if class_name in CAD_CLASSES:
        return "cad_export"
    return None


def lookup_parts_csv(path, mesh_sha):
    """``source`` for a mesh hash in parts.csv. Blank or unknown values are ignored."""
    if not path or not mesh_sha:
        return None
    sha = str(mesh_sha).strip().lower()
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            got = (row.get("input_mesh_sha256") or "").strip().lower()
            if got and got == sha:
                return normalize_source(row.get("source"))
    return None


def reconcile(declared, classifier_class):
    """Declared source wins. A disagreement is a warning, never a gate change."""
    declared = normalize_source(declared)
    family = classifier_family(classifier_class)
    agree = declared is None or family is None or declared == family
    warning = None
    if declared and family and declared != family:
        warning = (f"mesh source cross-check: declared source is {declared}, classifier says "
                   f"{classifier_class} ({family}). The declared source is kept. "
                   "The classifier does not set or tighten the gate.")
    elif declared is None and classifier_class:
        warning = (f"mesh source was not declared (parts.csv / --mesh-source); classifier says "
                   f"{classifier_class}. Gates are unchanged.")
    return {
        "source": declared,
        "classifier": classifier_class,
        "classifier_family": family,
        "agree": bool(agree),
        "warning": warning,
        "gates": "unchanged",
    }


def gates_for_declared_scan(declared, gates, scan_tol, reason="exact mesh"):
    """Acceptance limits for a declared scan.

    The fit tolerance used to build the solid is left alone. What this stops is the exact-mesh
    (or facet-sag) tolerance tightening rms / p95 below the scan-noise tolerance
    ``max(4.5·sigma, 5e-5·diag, 0.005)``. The classifier only produces a warning.
    """
    out = dict(gates)
    if normalize_source(declared) != "scan":
        return out, None
    if out.get("class") == "printed":
        return out, None
    basis = out.get("basis") or ""
    if "overridden accept_rms" in basis or "overridden accept_p95" in basis:
        return out, None
    scan_tol = float(scan_tol)
    scan_rms = 1.5 * scan_tol
    scan_p95 = 3.0 * scan_tol
    rms = float(out.get("rms_limit") or 0.0)
    p95 = float(out.get("p95_limit") or 0.0)
    if rms + 1e-12 >= scan_rms and p95 + 1e-12 >= scan_p95:
        return out, None
    warning = (f"mesh source is scan; classifier ({reason}) would tighten the gate to "
               f"rms<={rms:.5f} mm, p95<={p95:.5f} mm. The classifier only warns. "
               f"Gate uses the scan-noise tolerance {scan_tol:.5f} mm "
               f"(rms<={scan_rms:.5f}, p95<={scan_p95:.5f}).")
    out["rms_limit"] = float(scan_rms)
    out["p95_limit"] = float(scan_p95)
    out["tol"] = scan_tol
    out["basis"] = ("scan source: vertex→solid rms≤1.5·scan-noise tol, p95≤3·scan-noise tol; "
                    f"{reason} was not allowed to tighten the gate")
    out["exact_mesh_gate_ignored"] = True
    return out, warning


def apply_cross_check(classifier_record, gates, declared):
    """Return the same gate limits plus the cross-check record.

    ``gates`` is not replaced and its rms / p95 limits are not reduced.
    """
    record = reconcile(declared, (classifier_record or {}).get("class"))
    out = dict(gates)
    return out, record
