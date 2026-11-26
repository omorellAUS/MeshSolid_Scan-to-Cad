"""Timing for impeller fuse booleans. Measurement only.

The default solid is the ordinary serial fuse. ``--fuse-bench`` (or
``MESHSOLID_FUSE_BENCH=1``) re-runs the slowest fuse with OpenCASCADE parallel and glue
options and discards those shapes.
"""
from __future__ import annotations

import os
import time

BENCH_VARIANTS = (
    {"name": "serial_glue_off", "parallel": False, "glue": "off"},
    {"name": "parallel_glue_off", "parallel": True, "glue": "off"},
    {"name": "serial_glue_shift", "parallel": False, "glue": "shift"},
    {"name": "parallel_glue_shift", "parallel": True, "glue": "shift"},
)


def bench_enabled(flag=False, env=None):
    if flag:
        return True
    raw = os.environ.get("MESHSOLID_FUSE_BENCH") if env is None else env
    return str(raw or "") == "1"


class FuseProfile:
    def __init__(self):
        self.rows = []

    def call(self, name, fn):
        started = time.perf_counter()
        try:
            return fn()
        finally:
            self.rows.append({"op": name, "seconds": round(time.perf_counter() - started, 4)})

    def table(self):
        return {
            "ops": list(self.rows),
            "total_seconds": round(sum(row["seconds"] for row in self.rows), 4),
            "note": "timing only; boolean results are the default serial fuse",
        }

    def format_table(self):
        lines = ["op  seconds"]
        for row in self.rows:
            lines.append(f"{row['op']}  {row['seconds']:.4f}")
        if not self.rows:
            lines.append("(no boolean ops recorded)")
        return "\n".join(lines)


def _glue_enum(name):
    from OCP.BOPAlgo import BOPAlgo_GlueOff, BOPAlgo_GlueShift
    if name == "shift":
        return BOPAlgo_GlueShift
    return BOPAlgo_GlueOff


def bench_fuse(shape_a, shape_b):
    """Time parallel / glue variants of one fuse. None of the results is the solid."""
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse
    from OCP.collections import List_TopoDS_Shape

    rows = []
    for spec in BENCH_VARIANTS:
        started = time.perf_counter()
        op = BRepAlgoAPI_Fuse()
        if hasattr(op, "SetRunParallel"):
            op.SetRunParallel(bool(spec["parallel"]))
        if hasattr(op, "SetGlue"):
            op.SetGlue(_glue_enum(spec["glue"]))
        args = List_TopoDS_Shape()
        args.Append(shape_a)
        tools = List_TopoDS_Shape()
        tools.Append(shape_b)
        op.SetArguments(args)
        op.SetTools(tools)
        op.Build()
        _discarded = op.Shape()
        del _discarded
        rows.append({
            "variant": spec["name"],
            "parallel": spec["parallel"],
            "glue": spec["glue"],
            "seconds": round(time.perf_counter() - started, 4),
            "used_for_solid": False,
        })
    slowest = max(rows, key=lambda row: row["seconds"]) if rows else None
    return {
        "ran": True,
        "variants": rows,
        "slowest": None if slowest is None else slowest["variant"],
        "note": "benchmark only; the default solid is unchanged",
    }
