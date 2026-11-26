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


def apply_cross_check(classifier_record, gates, declared):
    """Return the same gate limits plus the cross-check record.

    ``gates`` is not replaced and its rms / p95 limits are not reduced.
    """
    record = reconcile(declared, (classifier_record or {}).get("class"))
    out = dict(gates)
    return out, record
