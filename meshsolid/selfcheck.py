"""Engine self-check, used by the Windows installer and by the Fusion add-in before its first run.

    python -m meshsolid.selfcheck            # imports + a synthetic box end to end (about 20-60 s)
    python -m meshsolid.selfcheck --quick    # imports only
    ... --json out.json                      # also write the result to a file

Prints one JSON object and exits 0 only if every check passed.  A failed check names the package or
step and the error text; nothing is reported as passed that was not run."""
from __future__ import annotations

import importlib
import json
import os
import platform
import sys
import tempfile
import time

PACKAGES = [("numpy", "numpy"), ("scipy", "scipy"), ("trimesh", "trimesh"), ("rtree", "rtree"),
            ("networkx", "networkx"), ("matplotlib", "matplotlib"), ("OCP", "cadquery-ocp"),
            ("pymeshlab", "pymeshlab"), ("igl", "libigl")]
OPTIONAL = {"igl"}          # deviation falls back to trimesh's closest-point search (slower)


def check(full=True):
    out = {"python": sys.version.split()[0], "platform": platform.platform(), "executable": sys.executable,
           "packages": {}, "errors": [], "warnings": []}
    for mod, pip in PACKAGES:
        try:
            m = importlib.import_module(mod)
            out["packages"][pip] = getattr(m, "__version__", "ok")
        except Exception as e:  # noqa: BLE001
            out["packages"][pip] = None
            (out["warnings"] if mod in OPTIONAL else out["errors"]).append(f"import {mod} ({pip}): {e}")
    try:
        import meshsolid  # noqa: F401
        out["meshsolid"] = os.path.dirname(os.path.abspath(meshsolid.__file__))
    except Exception as e:  # noqa: BLE001
        out["errors"].append(f"import meshsolid: {e}")
    if full and not out["errors"]:
        t0 = time.time()
        try:
            from . import synthetic as S
            from .pipeline import Params, run
            d = tempfile.mkdtemp(prefix="meshsolid_selfcheck_")
            shape, _ = S.box_with_hole()
            mp = os.path.join(d, "box.stl"); S.to_scan(shape, noise=0.02, seed=1).export(mp)
            step = os.path.join(d, "box.step")
            rep, *_ = run(mp, step, os.path.join(d, "box.report.json"), Params(), None, None, None)
            b = rep.get("stages", {}).get("build", {})
            out["roundtrip"] = {"status": rep.get("status"), "step_written": os.path.exists(step),
                                "n_solids": b.get("n_solids"), "seconds": round(time.time() - t0, 1), "dir": d}
            if rep.get("status") != "ok" or not os.path.exists(step):
                out["errors"].append(f"synthetic box round trip: status {rep.get('status')}, "
                                     f"failure {rep.get('failure')}")
        except Exception as e:  # noqa: BLE001
            out["errors"].append(f"synthetic box round trip raised: {type(e).__name__}: {e}")
    out["ok"] = not out["errors"]
    return out


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    res = check(full="--quick" not in argv)
    txt = json.dumps(res, indent=1)
    if "--json" in argv:                       # OCC writes progress to stdout; a file is unambiguous
        with open(argv[argv.index("--json") + 1], "w", encoding="utf-8") as f:
            f.write(txt)
    print(txt)
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
