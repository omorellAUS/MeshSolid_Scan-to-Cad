"""Kept for older callers: dimension list -> (name, expression, units, comment).  New code uses
report_params.parameters_from_report, which also maps gear, thread and impeller parameters."""
from .report_params import _dimensions


def parameter_specs(dimensions, prefix="ms_"):
    return [(s["name"], s["expr"], s["units"], s["comment"])
            for s in _dimensions({"dimensions": dimensions}, prefix) if s is not None]
