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
