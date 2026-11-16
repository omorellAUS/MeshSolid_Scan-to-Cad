"""Run a hang-prone OCC fillet with a time limit in a separate process.

POSIX: fork (fast, the shape is inherited).  Windows has no fork: the shape goes to a BRep file and a
fresh `python -m meshsolid.occ_child` process re-finds the edges by their sampled midpoints, fillets,
and writes the result back.  Either way a hang is killed at the timeout and reported as TimeoutError."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np


def edge_midpoint(e):
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    c = BRepAdaptor_Curve(e); t = 0.5 * (c.FirstParameter() + c.LastParameter()); p = c.Value(t)
    return [p.X(), p.Y(), p.Z()]


def _edges(shape):
    """Unique edges in explorer order (deterministic for a shape read back from the same BRep)."""
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_EDGE
    from OCP.TopoDS import TopoDS
    out = []; ex = TopExp_Explorer(shape, TopAbs_EDGE)
    while ex.More():
        e = TopoDS.Edge(ex.Current()); ex.Next()
        if not any(e.IsSame(s) for s in out):
            out.append(e)
    return out


def fillet(shape, edges_r):
    """edges_r: [(TopoDS_Edge, radius)].  The valid filleted shape, or None."""
    from OCP.BRepFilletAPI import BRepFilletAPI_MakeFillet
    from OCP.BRepCheck import BRepCheck_Analyzer
    mk = BRepFilletAPI_MakeFillet(shape)
    for e, r in edges_r:
        mk.Add(float(r), e)
    mk.Build()
    if mk.IsDone() and BRepCheck_Analyzer(mk.Shape()).IsValid():
        return mk.Shape()
    return None


def _use_fork():
    return hasattr(os, "fork") and os.environ.get("MESHSOLID_NO_FORK", "") != "1"


def fillet_in_child(shape, edges_r, timeout):
    """fillet() in a child process with a time limit; raises TimeoutError on a hang."""
    from OCP.BRepTools import BRepTools
    from OCP.BRep import BRep_Builder
    from OCP.TopoDS import TopoDS_Shape
    if _use_fork():
        from .impeller import _in_child
        return _in_child(lambda: fillet(shape, edges_r), timeout)
    d = tempfile.mkdtemp(prefix="msfillet_")
    pin, pout, pjob = (os.path.join(d, n) for n in ("in.brep", "out.brep", "job.json"))
    BRepTools.Write_s(shape, pin)
    with open(pjob, "w") as f:
        json.dump({"edges": [{"mid": edge_midpoint(e), "r": float(r)} for e, r in edges_r]}, f)
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    env = dict(os.environ)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    try:
        p = subprocess.Popen([sys.executable, "-m", "meshsolid.occ_child", pin, pjob, pout],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env, creationflags=flags)
        try:
            rc = p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            p.kill(); p.wait()
            raise TimeoutError
        if rc != 0 or not os.path.exists(pout):
            return None
        sh = TopoDS_Shape(); BRepTools.Read_s(sh, pout, BRep_Builder())
        return sh
    finally:
        for n in (pin, pout, pjob):
            if os.path.exists(n):
                os.unlink(n)
        os.rmdir(d)


def _main(pin, pjob, pout):
    from OCP.BRepTools import BRepTools
    from OCP.BRep import BRep_Builder
    from OCP.TopoDS import TopoDS_Shape
    sh = TopoDS_Shape(); BRepTools.Read_s(sh, pin, BRep_Builder())
    with open(pjob) as f:
        job = json.load(f)
    E = _edges(sh); M = np.array([edge_midpoint(e) for e in E])
    sel = []
    for it in job["edges"]:
        d = np.linalg.norm(M - np.asarray(it["mid"]), axis=1); i = int(np.argmin(d))
        if d[i] > 1e-6 * max(1.0, float(np.abs(M).max())):
            return 3                                   # edge not found again: refuse rather than guess
        sel.append((E[i], it["r"]))
    res = fillet(sh, sel)
    if res is None:
        return 2
    BRepTools.Write_s(res, pout)
    return 0


if __name__ == "__main__":
    sys.exit(_main(*sys.argv[1:4]))
