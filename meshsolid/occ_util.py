"""Small Open CASCADE (OCP) helpers."""
from __future__ import annotations

import numpy as np
from OCP.BRep import BRep_Tool
from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepGProp import BRepGProp
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.GProp import GProp_GProps
from OCP.IFSelect import IFSelect_RetDone
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Reader, STEPControl_Writer
from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED, TopAbs_SOLID, TopAbs_SHELL
from OCP.TopExp import TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS
from OCP.gp import gp_Ax1, gp_Dir, gp_Pnt, gp_Trsf, gp_Vec
import trimesh


def tessellate(shape, lin=0.01, ang=0.1):
    BRepMesh_IncrementalMesh(shape, lin, False, ang, True)
    verts, faces, off = [], [], 0
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = TopoDS.Face(exp.Current())
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(face, loc)
        if tri is not None:
            tr = loc.Transformation()
            for i in range(1, tri.NbNodes() + 1):
                p = tri.Node(i).Transformed(tr)
                verts.append((p.X(), p.Y(), p.Z()))
            rev = face.Orientation() == TopAbs_REVERSED
            for i in range(1, tri.NbTriangles() + 1):
                a, b, c = tri.Triangle(i).Get()
                faces.append((off + a - 1, off + c - 1, off + b - 1) if rev else (off + a - 1, off + b - 1, off + c - 1))
            off += tri.NbNodes()
        exp.Next()
    return np.array(verts), np.array(faces, dtype=np.int64)


_SURF_NAMES = {0: "plane", 1: "cylinder", 2: "cone", 3: "sphere", 4: "torus", 5: "bezier", 6: "bspline",
               7: "revolution", 8: "extrusion", 9: "offset", 10: "other"}


def brep_faces(shape):
    """[(TopoDS_Face, surface type name)] in TopExp order (the order tessellate() walks)."""
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    out = []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        f = TopoDS.Face(exp.Current())
        try:
            t = int(BRepAdaptor_Surface(f).GetType())
        except Exception:  # noqa: BLE001
            t = 10
        out.append((f, _SURF_NAMES.get(t, "other")))
        exp.Next()
    return out


def triangle_face_ids(shape, tm, lin=0.01, ang=0.1):
    """BRep face index (TopExp order) of every triangle of tm = shape_to_trimesh(shape, lin, ang), matched by
    triangle centroid (repair_mesh reorders and drops triangles). Returns (ids, [type name per face])."""
    from scipy.spatial import cKDTree
    BRepMesh_IncrementalMesh(shape, lin, False, ang, True)
    cents, ids = [], []
    faces = brep_faces(shape)
    for k, (face, _t) in enumerate(faces):
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(face, loc)
        if tri is None:
            continue
        tr = loc.Transformation()
        P = np.array([[q.X(), q.Y(), q.Z()] for q in (tri.Node(i).Transformed(tr) for i in range(1, tri.NbNodes() + 1))])
        T = np.array([tri.Triangle(i).Get() for i in range(1, tri.NbTriangles() + 1)], dtype=np.int64) - 1
        cents.append(P[T].mean(1))
        ids.append(np.full(len(T), k))
    if not cents:
        return np.full(len(tm.faces), -1), [t for _f, t in faces]
    C, I = np.vstack(cents), np.concatenate(ids)
    _d, j = cKDTree(C).query(np.asarray(tm.triangles_center, float))
    return I[j], [t for _f, t in faces]


def shape_to_trimesh(shape, lin=0.01, ang=0.1):
    from .meshio import repair_mesh
    v, f = tessellate(shape, lin, ang)
    m, _ = repair_mesh(v, f, merge_tol=1e-6)
    return m


def volume(shape):
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, props)
    return props.Mass()


def volume_precise(shape, eps=1e-6):
    """Adaptive-precision volume.  The default Gauss rule is off by up to ~1% on long multi-turn
    B-spline sweep faces (helical rotors); this one converges to eps relative error."""
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, props, eps, True)
    return props.Mass()


def area(shape):
    props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(shape, props)
    return props.Mass()


def check(shape):
    return bool(BRepCheck_Analyzer(shape).IsValid())


def count(shape, kind):
    exp, n = TopExp_Explorer(shape, kind), 0
    while exp.More():
        n += 1
        exp.Next()
    return n


def write_step(shape, path):
    w = STEPControl_Writer()
    w.Transfer(shape, STEPControl_AsIs)
    if w.Write(str(path)) != IFSelect_RetDone:
        raise IOError(f"STEP write failed: {path}")


def read_step(path):
    r = STEPControl_Reader()
    if r.ReadFile(str(path)) != IFSelect_RetDone:
        raise IOError(f"STEP read failed: {path}")
    r.TransferRoots()
    return r.OneShape()
