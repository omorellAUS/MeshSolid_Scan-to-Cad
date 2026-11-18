"""Deviation acceptance accuracy classes.

Two classes (user 2026-11-09):

  machined (default)
      Existing strict gates relative to the fit tolerance:
        vertex→solid rms ≤ 1.5·tol, p95 ≤ 3·tol
        solid→mesh p95 ≤ accept_p95 (+ facet sag when faceted)
      Use for machined / scanned metal parts. Do not auto-apply `printed`.

  printed
      Absolute gates for FDM/SLA / printed drone parts (user: 0.05 mm):
        vertex→solid rms ≤ 0.05 mm, p95 ≤ 0.10 mm  (same 2× ratio as 3/1.5)
      Opt-in via --accuracy printed or Params.accuracy="printed". Never the
      default; machined regressions must stay bit-identical when unset.

Explicit accept_rms / accept_p95 on Params still override the class.
"""
from __future__ import annotations

from typing import Any

# Absolute printed gates (mm). p95 = 2 × rms to mirror machined 3·tol / 1.5·tol.
PRINTED_RMS_MM = 0.05
PRINTED_P95_MM = 0.10

CLASSES = ("machined", "printed")


def normalize_accuracy(accuracy: str | None) -> str:
    if accuracy is None or accuracy == "":
        return "machined"
    a = str(accuracy).strip().lower()
    if a in ("scan", "scanned", "default", "strict"):
        return "machined"
    if a in ("print", "fdm", "sla", "drone"):
        return "printed"
    if a not in CLASSES:
        raise ValueError(f"unknown accuracy class {accuracy!r}; expected one of {CLASSES}")
    return a


def resolve_gates(accuracy: str | None, tol: float,
                  accept_rms: float | None = None,
                  accept_p95: float | None = None) -> dict[str, Any]:
    """Return rms/p95 limits and metadata for the report / self_check."""
    cls = normalize_accuracy(accuracy)
    if cls == "printed":
        rms = PRINTED_RMS_MM
        p95 = PRINTED_P95_MM
        basis = ("printed: absolute vertex→solid rms≤0.05 mm, p95≤0.10 mm "
                 "(user 2026-11-09; FDM/SLA drone parts)")
    else:
        rms = 1.5 * float(tol)
        p95 = 3.0 * float(tol)
        basis = "machined: vertex→solid rms≤1.5·tol, p95≤3·tol (existing default)"
    if accept_rms is not None:
        rms = float(accept_rms)
        basis += f"; overridden accept_rms={accept_rms}"
    if accept_p95 is not None:
        p95 = float(accept_p95)
        basis += f"; overridden accept_p95={accept_p95}"
    return {
        "class": cls,
        "rms_limit": float(rms),
        "p95_limit": float(p95),
        "tol": float(tol),
        "basis": basis,
    }

# --- 1.0.6: additional accuracy classes ---
ORGANIC_RMS_MM = 0.20      # loose gate for organic/impeller non-prismatic parts
ORGANIC_P95_MM = 0.40      # 2× ratio

ROUGH_SCAN_RMS_MM = 0.50   # very loose for rough hand-held scans
ROUGH_SCAN_P95_MM = 1.00   # 2× ratio

CLASSES = ("machined", "printed", "organic", "rough_scan")


def normalize_accuracy(accuracy: str | None) -> str:
    """Return the canonical accuracy class name."""
    if accuracy is None or accuracy == "":
        return "machined"
    a = str(accuracy).strip().lower()
    if a in ("scan", "scanned", "default", "strict", "machined"):
        return "machined"
    if a in ("print", "fdm", "sla", "drone", "printed"):
        return "printed"
    if a in ("organic", "impeller", "freeform"):
        return "organic"
    if a in ("rough", "rough_scan", "handheld", "organic scan"):
        return "rough_scan"
    if a not in CLASSES:
        raise ValueError(f"unknown accuracy class {accuracy!r}; expected one of {CLASSES}")
    return a


# --- 1.0.6 gate resolution ---
def resolve_gates(accuracy: str | None, tol: float | None,
                  accept_rms: float | None = None,
                  accept_p95: float | None = None) -> dict:
    """Return the acceptance gates dict for the given accuracy class.

    The dict always contains: class, rms_limit, p95_limit, basis.
    accuracy classes:
      machined   : rms ≤ 1.5·tol,      p95 ≤ 3·tol      (default, unchanged)
      printed    : rms ≤ 0.05 mm,      p95 ≤ 0.10 mm      (absolute, opt-in)
      organic    : rms ≤ 0.20 mm,      p95 ≤ 0.40 mm      (loose, non-prismatic / impeller)
      rough_scan : rms ≤ 0.50 mm,      p95 ≤ 1.00 mm      (very loose, hand-held)

    Explicit accept_rms / accept_p95 always override the class defaults.
    """
    cls = normalize_accuracy(accuracy)

    # Start from defaults per class
    if cls == "machined":
        rms_def = 1.5 * tol if tol else None
        p95_def = 3.0 * tol if tol else None
        basis = f"machined: rms≤{str(rms_def) if rms_def else '?'} mm, p95≤{str(p95_def) if p95_def else '?'} mm"
    elif cls == "printed":
        rms_def = PRINTED_RMS_MM
        p95_def = PRINTED_P95_MM
        basis = f"printed: rms≤{rss_def:.2f} mm, p95≤{p95_def:.2f} mm"
    elif cls == "organic":
        rms_def = ORGANIC_RMS_MM
        p95_def = ORGANIC_P95_MM
        basis = f"organic: rms≤{rms_def:.2f} mm, p95≤{p95_def:.2f} mm"
    elif cls == "rough_scan":
        rms_def = ROUGH_SCAN_RMS_MM
        p95_def = ROUGH_SCAN_P95_MM
        basis = f"rough_scan: rms≤{rms_def:.2f} mm, p95≤{p95_def:.2f} mm"
    else:
        rms_def = None
        p95_def = None
        basis = "unknown"

    # Now apply the Dict we had before (from user overrides or class defaults)
    out = {"class": cls, "basis": basis, "rms_limit": rms_def, "p95_limit": p95_def}

    # If the caller passed explicit accept_rms/accept_p95, they always win
    if accept_rms is not None:
        out["rms_limit"] = accept_rms
    if accept_p95 is not None:
        out["p95_limit"] = accept_p95

    # Re-compute basis string reflecting the final values
    out["basis"] = (f"{cls}: rms≤{out['rms_limit']:.2f} mm, "
                    f"p95≤{out['p95_limit']:.2f} mm")
    return out
