"""Shared helpers for the user test-set harness (STEP truth inventory, scoring).
Never writes geometry into git-tracked folders: inputs live in tests/user_scans/<set>/, outputs in out/user/<set>/."""
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[2]


def out_dir(set_name):
    """Outputs: out/user/<set> by default. MESHSOLID_TESTSET_OUT=<dir> puts them outside the repo (out/ can be
    wiped by other jobs); out/user/<set> is then a symlink to it."""
    import os
    base = os.environ.get("MESHSOLID_TESTSET_OUT")
    link = ROOT / "out" / "user" / set_name
    if not base:
        return link
    real = Path(base) / set_name
    real.mkdir(parents=True, exist_ok=True)
    link.parent.mkdir(parents=True, exist_ok=True)
    if not link.exists() and not link.is_symlink():
        link.symlink_to(real)
    return real
sys.path.insert(0, str(ROOT))
from meshsolid import occ_util as U  # noqa: E402
from OCP.BRepAdaptor import BRepAdaptor_Surface  # noqa: E402
from OCP.BRepBndLib import BRepBndLib  # noqa: E402
from OCP.Bnd import Bnd_Box  # noqa: E402
from OCP.TopAbs import TopAbs_FACE, TopAbs_SHELL, TopAbs_SOLID  # noqa: E402
from OCP.TopExp import TopExp_Explorer  # noqa: E402
from OCP.TopoDS import TopoDS  # noqa: E402

KIND = {0: "plane", 1: "cylinder", 2: "cone", 3: "sphere", 4: "torus", 5: "bezier", 6: "bspline",
        7: "revolution", 8: "extrusion", 9: "offset", 10: "other"}
ZONE = {"plane": "planar", "cylinder": "cylindrical", "cone": "conical_toroidal", "torus": "conical_toroidal",
        "sphere": "conical_toroidal"}   # everything else: freeform


def zone_of(kind):
    return ZONE.get(kind, "freeform")


def product_name(step_path):
    txt = Path(step_path).read_text(errors="ignore")
    m = re.search(r"PRODUCT\s*\(\s*'([^']*)'", txt)
    return m.group(1).strip() if m else Path(step_path).stem


def slug(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def units(step_path):
    txt = Path(step_path).read_text(errors="ignore")
    if "SI_UNIT ( .MILLI., .METRE. )" in txt or "SI_UNIT(.MILLI.,.METRE.)" in txt:
        return "mm"
    if "INCH" in txt.upper():
        return "inch"
    return "unknown"


def explore(shape, kind):
    ex = TopExp_Explorer(shape, kind)
    while ex.More():
        yield ex.Current()
        ex.Next()


def solids(shape):
    return [TopoDS.Solid(s) for s in explore(shape, TopAbs_SOLID)]


def face_kind(face):
    return KIND.get(int(BRepAdaptor_Surface(TopoDS.Face(face)).GetType()), "other")


def face_types(shape):
    return dict(Counter(face_kind(f) for f in explore(shape, TopAbs_FACE)))


def bbox(shape):
    b = Bnd_Box(); BRepBndLib.Add_s(shape, b)
    lo, hi = b.CornerMin(), b.CornerMax()
    return [lo.X(), lo.Y(), lo.Z()], [hi.X(), hi.Y(), hi.Z()]


def tessellate_labeled(shape, lin, ang=0.1):
    """Tessellate a shape; returns (trimesh, per-triangle zone label array of strings, per-triangle face kind)."""
    from OCP.BRep import BRep_Tool
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.TopAbs import TopAbs_REVERSED
    from OCP.TopLoc import TopLoc_Location
    BRepMesh_IncrementalMesh(shape, lin, False, ang, True)
    V, F, K, off = [], [], [], 0
    for fc in explore(shape, TopAbs_FACE):
        face = TopoDS.Face(fc)
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(face, loc)
        if tri is None:
            continue
        tr = loc.Transformation()
        n = tri.NbNodes()
        V.extend((lambda p: (p.X(), p.Y(), p.Z()))(tri.Node(i).Transformed(tr)) for i in range(1, n + 1))
        rev = face.Orientation() == TopAbs_REVERSED
        k = face_kind(face)
        for i in range(1, tri.NbTriangles() + 1):
            a, b, c = tri.Triangle(i).Get()
            F.append((off + a - 1, off + c - 1, off + b - 1) if rev else (off + a - 1, off + b - 1, off + c - 1))
            K.append(k)
        off += n
    m = trimesh.Trimesh(np.array(V), np.array(F, dtype=np.int64), process=False)
    return m, np.array(K)


def p2s(P, m, n=2000000, seed=0):
    """Signed-ish point-to-surface distance from points P to mesh m (point-to-plane vs a dense oriented sample,
    clamped by the point-to-point distance). Fast and accurate when the sample spacing is << curvature radius."""
    from scipy.spatial import cKDTree
    Q, fi = trimesh.sample.sample_surface(m, n, seed=seed)
    N = m.face_normals[fi]
    dd, j = cKDTree(Q).query(P, workers=-1)
    ds = np.einsum("ij,ij->i", P - Q[j], N[j])
    return np.where(np.abs(ds) <= dd, ds, np.sign(ds) * dd)


def stats(d):
    a = np.abs(np.asarray(d))
    if len(a) == 0:
        return None
    return {"rms": round(float(np.sqrt(np.mean(a ** 2))), 4), "p95": round(float(np.percentile(a, 95)), 4),
            "max": round(float(a.max()), 4), "n": int(len(a))}
