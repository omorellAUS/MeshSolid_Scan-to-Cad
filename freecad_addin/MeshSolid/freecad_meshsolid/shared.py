"""Locate the add-in client library (mslib) shared with the Fusion add-in.

Installed: the installer copies mslib next to this package (MeshSolid/mslib).  In the repo it lives
in fusion_addin/MeshSolid/mslib; that copy is used when no installed one exists."""
import os
import sys

WB_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))     # .../Mod/MeshSolid
SETTINGS_FILE = os.path.join(WB_DIR, "settings.json")


def _candidates():
    yield WB_DIR
    yield os.path.join(os.path.dirname(os.path.dirname(WB_DIR)), "fusion_addin", "MeshSolid")


def ensure_mslib():
    for d in _candidates():
        if os.path.isfile(os.path.join(d, "mslib", "engine_client.py")):
            if d not in sys.path:
                sys.path.insert(0, d)
            return d
    raise ImportError("MeshSolid: the shared client library 'mslib' was not found next to the workbench "
                      f"({WB_DIR}); rerun install_windows.ps1 -FreeCAD")


ensure_mslib()
from mslib import engine_client, outcome, report_params, review  # noqa: E402,F401
