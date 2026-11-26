"""Strict ``valid_solid``: exactly one closed solid that passes BRepCheck_Analyzer.

The solid count is always returned, including zero.
"""
from __future__ import annotations


def valid_solid_record(n_solids, brepcheck_valid, closed):
    n = 0 if n_solids is None else int(n_solids)
    brep = bool(brepcheck_valid)
    is_closed = bool(closed)
    return {
        "n_solids": n,
        "brepcheck_valid": brep,
        "closed": is_closed,
        "valid_solid": bool(n == 1 and brep and is_closed),
    }


def _shells_closed(shape):
    from OCP.ShapeAnalysis import ShapeAnalysis_Shell
    from OCP.TopAbs import TopAbs_SHELL, TopAbs_SOLID
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    shell_of = getattr(TopoDS, "Shell_s", None) or TopoDS.Shell
    solids = TopExp_Explorer(shape, TopAbs_SOLID)
    n = 0
    while solids.More():
        n += 1
        shells = TopExp_Explorer(solids.Current(), TopAbs_SHELL)
        saw = False
        while shells.More():
            saw = True
            analysis = ShapeAnalysis_Shell()
            analysis.LoadShells(shell_of(shells.Current()))
            if analysis.HasFreeEdges():
                return n, False
            shells.Next()
        if not saw:
            return n, False
        solids.Next()
    return n, n > 0


def assess_shape(shape):
    """Kernel check plus a closed-shell check. ``n_solids`` is always set."""
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.TopAbs import TopAbs_SOLID
    from OCP.TopExp import TopExp_Explorer

    brep = bool(BRepCheck_Analyzer(shape).IsValid())
    try:
        n, closed = _shells_closed(shape)
    except Exception:
        n = 0
        exp = TopExp_Explorer(shape, TopAbs_SOLID)
        while exp.More():
            n += 1
            exp.Next()
        closed = False
    return valid_solid_record(n, brep, closed)
