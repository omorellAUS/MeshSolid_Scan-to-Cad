"""Command-dialog choices -> meshsolid engine CLI options.  No adsk import (unit-tested outside Fusion).

Every option maps to a flag that exists in `python -m meshsolid` (meshsolid/cli.py); tests check the
flags against the engine's --help.  Nothing here invents engine behaviour: a part type only switches the
engine's existing recognisers on/off and picks the accuracy class."""
from __future__ import annotations

VERSION = "1.0.2a"

# id, label, tooltip, engine options
PART_TYPES = [
    ("auto", "Auto", "Let the engine decide (all recognisers on: gears, impellers, screws).", {}),
    ("mechanical", "Mechanical",
     "Prismatic / turned machined part. Gear, impeller and screw recognition are switched off.",
     {"gears": "off", "impeller": "off", "screws": "off"}),
    ("gear", "Gear", "Involute gear on a turned body (impeller and screw recognition off).",
     {"gears": "auto", "impeller": "off", "screws": "off"}),
    ("impeller", "Impeller", "N-bladed rotor: hub revolve + patterned B-spline blade (gear and screw recognition off). "
     "If the impeller recogniser cannot build it, the run stops with the reason instead of trying the slow general method.",
     {"impeller": "on", "gears": "off", "screws": "off"}),
    ("freeform", "Freeform", "Curved 3D-printed part (wings, ailerons, skins): printed accuracy class, "
     "which lets the engine try its section-loft builder first. Gear/impeller/screw recognition off.",
     {"gears": "off", "impeller": "off", "screws": "off", "accuracy": "printed"}),
]

# id, label, tooltip, engine --accuracy value (None = from part type)
ACCURACY = [
    ("auto", "Auto (from part type)", "Freeform -> Printed; everything else -> Machined.", None),
    ("machined", "Machined (gate 1.5 / 3 x tol)",
     "Deviation gate relative to the fit tolerance, which is estimated from the scan noise unless you set one.",
     "machined"),
    ("printed", "Printed (0.05 / 0.10 mm)", "Absolute gate for FDM/SLA parts: rms 0.05 mm, p95 0.10 mm.", "printed"),
]

# id, label, engine --device value
DEVICE = [
    ("auto", "Auto (GPU if available, else CPU)", "auto"),
    ("cpu", "CPU only", "cpu"),
    ("cuda", "GPU only (stop if no NVIDIA GPU)", "cuda"),
]

TIME_MIN, TIME_MAX = 5, 180            # minutes, time-limit slider


def by_id(table, key):
    for row in table:
        if row[0] == key:
            return row
    raise KeyError(key)


def by_label(table, label):
    for row in table:
        if row[1] == label:
            return row
    raise KeyError(label)


def default_minutes(timeout_s):
    m = int(round((timeout_s or 5400) / 60.0))
    return max(TIME_MIN, min(TIME_MAX, m))


def engine_options(part="auto", accuracy="auto", device="auto", tol_mm=0.0):
    """-> dict for EngineJob.options / BatchJob.options (keys become --kebab-case flags)."""
    opts = dict(by_id(PART_TYPES, part)[3])
    acc = by_id(ACCURACY, accuracy)[3]
    if acc is not None:
        opts["accuracy"] = acc
    opts["device"] = by_id(DEVICE, device)[2]
    if tol_mm and float(tol_mm) > 0:
        opts["tol"] = round(float(tol_mm), 4)
    return opts


def timeout_for(minutes):
    """Slider minutes -> add-in kill timeout in s (the engine's own --time-budget is set 60 s under it by
    EngineJob.command so the watchdog can still write a report)."""
    return int(max(TIME_MIN, min(TIME_MAX, int(minutes)))) * 60


def summary(part, accuracy, device, minutes, tol_mm=0.0):
    """One line for the dialog: what will be sent to the engine."""
    o = engine_options(part, accuracy, device, tol_mm)
    flags = " ".join(f"--{k.replace('_', '-')} {v}" for k, v in o.items())
    return f"engine: {flags} --time-budget {timeout_for(minutes) - 60}"
