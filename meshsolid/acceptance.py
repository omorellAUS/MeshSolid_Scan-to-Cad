"""Plain-language gate reporting.

The boolean that decides whether a STEP is written is not changed here. The text
states which rule accepted the solid and, when it missed the class limit, by how much.
"""
from __future__ import annotations


def describe_gate(accepted: bool, rms: float | None, class_rms: float | None, rule: str = "") -> str:
    """One or two sentences for the report and the log.

    A solid kept by the widened blade-scatter gate still says it failed the class
    limit, for example ``Accepted by the 1.5x blade-scatter rule. Failed the class
    limit: 5.6x over 0.009 mm.``
    """
    parts: list[str] = []
    text = rule or ""
    low = text.lower()
    if accepted and "scatter" in low and "1.5" in text:
        parts.append("Accepted by the 1.5x blade-scatter rule.")
    elif accepted and "scatter" in low and ("3 x" in low or "3.0 x" in low or low.strip().startswith("3")):
        parts.append("Accepted by the 3x blade-scatter rule.")
    elif accepted and text:
        parts.append(f"Accepted by the rule: {text}.")
    elif accepted:
        parts.append("Accepted by the class limit.")
    if class_rms is not None and rms is not None and float(class_rms) > 0:
        limit = float(class_rms)
        if float(rms) > limit + 1e-12:
            ratio = float(rms) / limit
            parts.append(f"Failed the class limit: {ratio:.1f}x over {limit:.3f} mm.")
        else:
            parts.append(f"Within the class limit ({limit:.3f} mm).")
    return " ".join(parts)


def annotate_acceptance(acceptance: dict, *, accepted: bool, rms: float, p95: float,
                        class_rms: float, class_p95: float, rule: str) -> dict:
    """Add the plain summary. ``passed`` stays whatever the caller set, so a STEP
    that would have been written is still written."""
    acceptance["summary"] = describe_gate(accepted, rms, class_rms, rule)
    acceptance["class_rms_limit"] = float(class_rms)
    acceptance["class_p95_limit"] = float(class_p95)
    acceptance["class_limit_passed"] = bool(
        float(rms) <= float(class_rms) + 1e-12 and float(p95) <= float(class_p95) + 1e-12)
    acceptance["accepting_rule"] = rule or ""
    return acceptance
