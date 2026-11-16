"""List analytic faces of a STEP (cylinders/cones/planes/tori) with radius and axial extent along x.
Usage: python scripts/step_faces.py a.step [b.step ...] > out.json"""
import json, sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from meshsolid import occ_util as U  # noqa: E402
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.TopAbs import TopAbs_FACE
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS
from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib
from OCP.GProp import GProp_GProps
from OCP.BRepGProp import BRepGProp
out = {}
for p in sys.argv[1:]:
    s = U.read_step(p)
    ex = TopExp_Explorer(s, TopAbs_FACE)
    agg = defaultdict(lambda: {"n": 0, "area": 0.0, "x0": 1e9, "x1": -1e9})
    bs = {"n": 0, "area": 0.0, "x0": 1e9, "x1": -1e9}
    while ex.More():
        f = TopoDS.Face(ex.Current()); a = BRepAdaptor_Surface(f); t = int(a.GetType())
        b = Bnd_Box(); BRepBndLib.Add_s(f, b); lo, hi = b.CornerMin(), b.CornerMax()
        g = GProp_GProps(); BRepGProp.SurfaceProperties_s(f, g); ar = g.Mass()
        if t == 1:
            c = a.Cylinder(); k = ("cylinder", round(2 * c.Radius(), 3), round(c.Axis().Direction().X(), 3))
        elif t == 0:
            pl = a.Plane(); n = pl.Axis().Direction()
            k = ("plane", round(pl.Location().X(), 2) if abs(n.X()) > 0.99 else "non-axial", None)
        elif t == 2:
            c = a.Cone(); k = ("cone", round(2 * c.RefRadius(), 3), round(c.SemiAngle() * 57.29578, 2))
        elif t == 4:
            tt = a.Torus(); k = ("torus", round(tt.MajorRadius(), 3), round(tt.MinorRadius(), 3))
        else:
            k = None
        d = bs if k is None else agg[k]
        d["n"] += 1; d["area"] += ar; d["x0"] = min(d["x0"], lo.X()); d["x1"] = max(d["x1"], hi.X())
        ex.Next()
    rows = [{"kind": k[0], "p1": k[1], "p2": k[2], "n": v["n"], "area": round(v["area"], 1), "x": [round(v["x0"], 2), round(v["x1"], 2)]}
            for k, v in agg.items() if v["area"] > 5]
    rows.sort(key=lambda r: r["x"][0])
    out[Path(p).name] = {"analytic": rows, "bspline_or_other": {k: round(v, 2) if isinstance(v, float) else v for k, v in bs.items()}}
print(json.dumps(out, indent=1))
