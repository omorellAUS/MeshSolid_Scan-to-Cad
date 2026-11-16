"""Info for one (large) STEP: read time, peak RSS, validity, solids/shells, volume, area, bbox, face types,
then a tessellation saved as STL for later comparison. Usage: python scripts/step_info.py in.step out.json out.stl [lin_defl]"""
import json, resource, sys, time
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from OCP.STEPControl import STEPControl_Reader
from OCP.IFSelect import IFSelect_RetDone
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.TopAbs import TopAbs_FACE, TopAbs_SOLID, TopAbs_SHELL
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS
from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.GProp import GProp_GProps
from OCP.BRepGProp import BRepGProp
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.Interface import Interface_Static
from meshsolid import occ_util as U

KIND = {0: "plane", 1: "cylinder", 2: "cone", 3: "sphere", 4: "torus", 5: "bezier", 6: "bspline",
        7: "revolution", 8: "extrusion", 9: "offset", 10: "other"}
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2)
inp, oj, ostl = sys.argv[1:4]
lin = float(sys.argv[4]) if len(sys.argv) > 4 else 0.05
rec = {"file": Path(inp).name, "bytes": Path(inp).stat().st_size}
t = time.time()
r = STEPControl_Reader()
st = r.ReadFile(inp)
rec["read_status_ok"] = st == IFSelect_RetDone
rec["length_unit_note"] = "OCC converts to mm"
r.TransferRoots()
s = r.OneShape()
rec["read_seconds"] = round(time.time() - t, 1); rec["rss_gb_after_read"] = rss()
def cnt(kind):
    ex = TopExp_Explorer(s, kind); n = 0
    while ex.More(): n += 1; ex.Next()
    return n
rec["n_solids"], rec["n_shells"], rec["n_faces"] = cnt(TopAbs_SOLID), cnt(TopAbs_SHELL), cnt(TopAbs_FACE)
c = Counter(); ex = TopExp_Explorer(s, TopAbs_FACE)
while ex.More():
    c[KIND.get(int(BRepAdaptor_Surface(TopoDS.Face(ex.Current())).GetType()), "other")] += 1; ex.Next()
rec["face_types"] = dict(c)
b = Bnd_Box(); BRepBndLib.Add_s(s, b); lo, hi = b.CornerMin(), b.CornerMax()
rec["bbox_min"] = [round(lo.X(), 3), round(lo.Y(), 3), round(lo.Z(), 3)]
rec["bbox_size"] = [round(hi.X() - lo.X(), 3), round(hi.Y() - lo.Y(), 3), round(hi.Z() - lo.Z(), 3)]
g = GProp_GProps(); BRepGProp.VolumeProperties_s(s, g); rec["volume"] = round(g.Mass(), 1)
g2 = GProp_GProps(); BRepGProp.SurfaceProperties_s(s, g2); rec["area"] = round(g2.Mass(), 1)
# per-solid volumes
sv = []; ex = TopExp_Explorer(s, TopAbs_SOLID)
while ex.More():
    gg = GProp_GProps(); BRepGProp.VolumeProperties_s(ex.Current(), gg); sv.append(round(gg.Mass(), 1)); ex.Next()
rec["solid_volumes_top10"] = sorted(sv, reverse=True)[:10]
print(json.dumps(rec), flush=True)
t = time.time()
rec["brepcheck_valid"] = bool(BRepCheck_Analyzer(s).IsValid()); rec["check_seconds"] = round(time.time() - t, 1)
print("valid", rec["brepcheck_valid"], flush=True)
t = time.time()
m = U.shape_to_trimesh(s, lin=lin, ang=0.2)
m.export(ostl); rec["tess_triangles"] = int(len(m.faces)); rec["tess_seconds"] = round(time.time() - t, 1)
rec["rss_gb_peak"] = rss()
json.dump(rec, open(oj, "w"), indent=1)
print(json.dumps(rec))
