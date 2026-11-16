"""Synthetic CAD parts with known dimensions -> noisy 'scan' meshes."""
from __future__ import annotations

import numpy as np
import trimesh
from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
from OCP.BRepFilletAPI import BRepFilletAPI_MakeFillet
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
from OCP.TopAbs import TopAbs_EDGE
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS
from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Curve
from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt

from .occ_util import shape_to_trimesh


def box_with_hole(L=40.0, W=30.0, H=20.0, D=10.0, cx=None, cy=None):
    cx = L / 2 if cx is None else cx
    cy = W / 2 if cy is None else cy
    box = BRepPrimAPI_MakeBox(L, W, H).Shape()
    cyl = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(cx, cy, -1), gp_Dir(0, 0, 1)), D / 2, H + 2).Shape()
    s = BRepAlgoAPI_Cut(box, cyl).Shape()
    truth = {"L": L, "W": W, "H": H, "hole_d": D, "hole_x": cx, "hole_y": cy,
             "volume": L * W * H - np.pi * D * D / 4 * H}
    return s, truth


def stepped_shaft(steps=((20.0, 30.0), (12.0, 25.0), (8.0, 15.0))):
    z, s = 0.0, None
    for d, l in steps:
        c = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(0, 0, z), gp_Dir(0, 0, 1)), d / 2, l).Shape()
        s = c if s is None else BRepAlgoAPI_Fuse(s, c).Shape()
        z += l
    truth = {"diameters": [d for d, _ in steps], "lengths": [l for _, l in steps],
             "volume": float(sum(np.pi * d * d / 4 * l for d, l in steps))}
    return s, truth


def plate_with_fillets(L=60.0, W=40.0, T=8.0, R=5.0, D=8.0):
    box = BRepPrimAPI_MakeBox(L, W, T).Shape()
    mk = BRepFilletAPI_MakeFillet(box)
    exp = TopExp_Explorer(box, TopAbs_EDGE)
    seen = []
    while exp.More():
        e = TopoDS.Edge(exp.Current())
        c = BRepAdaptor_Curve(e)
        p0, p1 = c.Value(c.FirstParameter()), c.Value(c.LastParameter())
        if abs(p0.X() - p1.X()) < 1e-9 and abs(p0.Y() - p1.Y()) < 1e-9:  # vertical edges
            key = (round(p0.X(), 6), round(p0.Y(), 6))
            if key not in seen:
                seen.append(key)
                mk.Add(R, e)
        exp.Next()
    s = mk.Shape()
    cyl = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(L / 2, W / 2, -1), gp_Dir(0, 0, 1)), D / 2, T + 2).Shape()
    s = BRepAlgoAPI_Cut(s, cyl).Shape()
    vol = (L * W - (4 - np.pi) * R * R) * T - np.pi * D * D / 4 * T
    truth = {"L": L, "W": W, "T": T, "fillet_r": R, "hole_d": D, "volume": vol}
    return s, truth


def to_scan(shape, edge=1.0, noise=0.02, seed=0, rotate=None):
    """Tessellate exactly, subdivide to ~edge length, displace vertices along normals with N(0, noise)."""
    import pymeshlab
    m = shape_to_trimesh(shape, lin=0.002, ang=0.05)
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(m.vertices, m.faces))
    ms.meshing_isotropic_explicit_remeshing(iterations=8, targetlen=pymeshlab.PureValue(edge),
                                            featuredeg=30.0)
    mm = ms.current_mesh()
    m = trimesh.Trimesh(mm.vertex_matrix(), mm.face_matrix(), process=True)
    rng = np.random.default_rng(seed)
    if rotate is not None:  # arbitrary pose, as a real scan would be
        m.apply_transform(rotate)
    if noise > 0:
        m.vertices = m.vertices + m.vertex_normals * rng.normal(0, noise, (len(m.vertices), 1))
    return m


PARTS = {"box_with_hole": box_with_hole, "stepped_shaft": stepped_shaft,
         "plate_with_fillets": plate_with_fillets}


def filleted_flange(D1=40.0, H1=10.0, D2=20.0, H2=15.0, R_top=2.0, R_root=1.5, D_bore=8.0):
    """Revolved part with toroidal blends: flange, boss, edge round on the flange, root fillet, bore."""
    base = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), D1 / 2, H1).Shape()
    boss = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(0, 0, H1), gp_Dir(0, 0, 1)), D2 / 2, H2).Shape()
    s = BRepAlgoAPI_Fuse(base, boss).Shape()
    mk = BRepFilletAPI_MakeFillet(s)
    exp = TopExp_Explorer(s, TopAbs_EDGE)
    while exp.More():
        e = TopoDS.Edge(exp.Current())
        c = BRepAdaptor_Curve(e)
        p = c.Value(c.FirstParameter())
        rr = np.hypot(p.X(), p.Y())
        if abs(p.Z() - H1) < 1e-6 and abs(rr - D1 / 2) < 1e-6:
            mk.Add(R_top, e)
        elif abs(p.Z() - H1) < 1e-6 and abs(rr - D2 / 2) < 1e-6:
            mk.Add(R_root, e)
        exp.Next()
    s = mk.Shape()
    bore = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(0, 0, -1), gp_Dir(0, 0, 1)), D_bore / 2, H1 + H2 + 2).Shape()
    s = BRepAlgoAPI_Cut(s, bore).Shape()
    from .occ_util import volume
    truth = {"D1": D1, "H1": H1, "D2": D2, "H2": H2, "R_top": R_top, "R_root": R_root, "D_bore": D_bore,
             "volume": float(volume(s))}
    return s, truth


PARTS["filleted_flange"] = filleted_flange


# ---------------------------------------------------------------- M3: harder prismatic ladder
def _vol(s):
    from .occ_util import volume
    return float(volume(s))


def _edges(s):
    exp = TopExp_Explorer(s, TopAbs_EDGE)
    while exp.More():
        yield TopoDS.Edge(exp.Current())
        exp.Next()


def _cyl(x, y, z, r, h):
    return BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(x, y, z), gp_Dir(0, 0, 1)), r, h).Shape()


def chamfered_block(L=50.0, W=30.0, H=15.0, C=2.0, D=8.0):
    """Box with equal-distance chamfers on the four top edges and a through hole."""
    from OCP.BRepFilletAPI import BRepFilletAPI_MakeChamfer
    box = BRepPrimAPI_MakeBox(L, W, H).Shape()
    mk = BRepFilletAPI_MakeChamfer(box)
    for e in _edges(box):
        c = BRepAdaptor_Curve(e)
        p0, p1 = c.Value(c.FirstParameter()), c.Value(c.LastParameter())
        if abs(p0.Z() - H) < 1e-9 and abs(p1.Z() - H) < 1e-9:
            mk.Add(C, e)
    s = BRepAlgoAPI_Cut(mk.Shape(), _cyl(L / 2, W / 2, -1, D / 2, H + 2)).Shape()
    return s, {"L": L, "W": W, "H": H, "chamfer": C, "hole_d": D, "volume": _vol(s)}


def slotted_plate(L=60.0, W=40.0, T=8.0, SL=20.0, SW=8.0):
    """Plate with a through obround slot (two half-cylinders + flats) along X."""
    cx, cy = L / 2, W / 2
    slot = BRepPrimAPI_MakeBox(gp_Pnt(cx - SL / 2 + SW / 2, cy - SW / 2, -1), SL - SW, SW, T + 2).Shape()
    slot = BRepAlgoAPI_Fuse(slot, _cyl(cx - SL / 2 + SW / 2, cy, -1, SW / 2, T + 2)).Shape()
    slot = BRepAlgoAPI_Fuse(slot, _cyl(cx + SL / 2 - SW / 2, cy, -1, SW / 2, T + 2)).Shape()
    s = BRepAlgoAPI_Cut(BRepPrimAPI_MakeBox(L, W, T).Shape(), slot).Shape()
    return s, {"L": L, "W": W, "T": T, "slot_len": SL, "slot_w": SW, "volume": _vol(s)}


def counterbore_plate(L=60.0, W=40.0, T=10.0, D=6.6, CB=11.0, CBD=6.4, inset=10.0):
    """Plate with 4 counterbored holes in a rectangular pattern (M6 socket-head style)."""
    s = BRepPrimAPI_MakeBox(L, W, T).Shape()
    centers = [(inset, inset), (L - inset, inset), (inset, W - inset), (L - inset, W - inset)]
    for x, y in centers:
        s = BRepAlgoAPI_Cut(s, _cyl(x, y, -1, D / 2, T + 2)).Shape()
        s = BRepAlgoAPI_Cut(s, _cyl(x, y, T - CBD, CB / 2, CBD + 1)).Shape()
    return s, {"L": L, "W": W, "T": T, "hole_d": D, "cbore_d": CB, "cbore_depth": CBD,
               "centers": centers, "volume": _vol(s)}


def bolt_circle_flange(D=60.0, T=8.0, n=6, d=6.0, PCD=45.0, bore=20.0):
    """Disk with a central bore and n holes on a pitch circle."""
    s = BRepAlgoAPI_Cut(_cyl(0, 0, 0, D / 2, T), _cyl(0, 0, -1, bore / 2, T + 2)).Shape()
    for k in range(n):
        a = 2 * np.pi * k / n
        s = BRepAlgoAPI_Cut(s, _cyl(PCD / 2 * np.cos(a), PCD / 2 * np.sin(a), -1, d / 2, T + 2)).Shape()
    return s, {"D": D, "T": T, "n_holes": n, "hole_d": d, "PCD": PCD, "bore": bore, "volume": _vol(s)}


def countersunk_plate(L=50.0, W=30.0, T=8.0, D=5.5, CS=10.4, angle=90.0):
    """Plate with one through hole and a 90-degree countersink (cone) from the top."""
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeCone
    s = BRepAlgoAPI_Cut(BRepPrimAPI_MakeBox(L, W, T).Shape(), _cyl(L / 2, W / 2, -1, D / 2, T + 2)).Shape()
    h = (CS - D) / 2 / np.tan(np.radians(angle / 2))
    cone = BRepPrimAPI_MakeCone(gp_Ax2(gp_Pnt(L / 2, W / 2, T - h), gp_Dir(0, 0, 1)), D / 2, CS / 2 + 1 * np.tan(np.radians(angle / 2)), h + 1).Shape()
    s = BRepAlgoAPI_Cut(s, cone).Shape()
    return s, {"L": L, "W": W, "T": T, "hole_d": D, "csink_d": CS, "csink_angle": angle, "volume": _vol(s)}


def drafted_block(L=40.0, W=30.0, H=20.0, draft=5.0):
    """Block whose four side faces carry a draft angle (taper inward toward the top)."""
    from OCP.BRepOffsetAPI import BRepOffsetAPI_DraftAngle
    from OCP.gp import gp_Pln
    from OCP.TopAbs import TopAbs_FACE
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.GeomAbs import GeomAbs_Plane
    box = BRepPrimAPI_MakeBox(L, W, H).Shape()
    dr = BRepOffsetAPI_DraftAngle(box)
    exp = TopExp_Explorer(box, TopAbs_FACE)
    neutral = gp_Pln(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1))
    while exp.More():
        f = TopoDS.Face(exp.Current())
        a = BRepAdaptor_Surface(f)
        if a.GetType() == GeomAbs_Plane and abs(a.Plane().Axis().Direction().Z()) < 1e-9:
            dr.Add(f, gp_Dir(0, 0, 1), np.radians(draft), neutral)
        exp.Next()
    dr.Build()
    s = dr.Shape()
    return s, {"L": L, "W": W, "H": H, "draft_deg": draft, "volume": _vol(s)}


M3_PARTS = {"chamfered_block": chamfered_block, "slotted_plate": slotted_plate,
            "counterbore_plate": counterbore_plate, "bolt_circle_flange": bolt_circle_flange,
            "countersunk_plate": countersunk_plate, "drafted_block": drafted_block}


def pocket_plate(L=60.0, W=40.0, T=12.0, PL=36.0, PW=22.0, PD=6.0, Rc=4.0):
    """Plate with a blind rectangular pocket from the top; vertical pocket corners rounded Rc."""
    pocket = BRepPrimAPI_MakeBox(gp_Pnt((L - PL) / 2, (W - PW) / 2, T - PD), PL, PW, PD + 1).Shape()
    mk = BRepFilletAPI_MakeFillet(pocket)
    for e in _edges(pocket):
        c = BRepAdaptor_Curve(e)
        p0, p1 = c.Value(c.FirstParameter()), c.Value(c.LastParameter())
        if abs(p0.X() - p1.X()) < 1e-9 and abs(p0.Y() - p1.Y()) < 1e-9:
            mk.Add(Rc, e)
    s = BRepAlgoAPI_Cut(BRepPrimAPI_MakeBox(L, W, T).Shape(), mk.Shape()).Shape()
    return s, {"L": L, "W": W, "T": T, "pocket_L": PL, "pocket_W": PW, "pocket_depth": PD,
               "corner_r": Rc, "volume": _vol(s)}


def chamfer_fillet_block(L=50.0, W=30.0, H=16.0, C=1.5, R=3.0):
    """Box with R fillets on the 4 vertical edges, then a C chamfer around the whole top outline
    (planes, cylinders, chamfer cones at the corners)."""
    from OCP.BRepFilletAPI import BRepFilletAPI_MakeChamfer
    box = BRepPrimAPI_MakeBox(L, W, H).Shape()
    mk = BRepFilletAPI_MakeFillet(box)
    for e in _edges(box):
        c = BRepAdaptor_Curve(e)
        p0, p1 = c.Value(c.FirstParameter()), c.Value(c.LastParameter())
        if abs(p0.X() - p1.X()) < 1e-9 and abs(p0.Y() - p1.Y()) < 1e-9:
            mk.Add(R, e)
    s = mk.Shape()
    ch = BRepFilletAPI_MakeChamfer(s)
    for e in _edges(s):
        c = BRepAdaptor_Curve(e)
        p0, p1 = c.Value(c.FirstParameter()), c.Value(c.LastParameter())
        pm = c.Value(0.5 * (c.FirstParameter() + c.LastParameter()))
        if all(abs(p.Z() - H) < 1e-9 for p in (p0, p1, pm)):
            ch.Add(C, e)
    s = ch.Shape()
    return s, {"L": L, "W": W, "H": H, "chamfer": C, "fillet_r": R, "volume": _vol(s)}


M3_PARTS["pocket_plate"] = pocket_plate
M3_PARTS["chamfer_fillet_block"] = chamfer_fillet_block


# ---------------------------------------------------------------- threaded parts (mesh-native)
def _iso_thread_r(z, th, r_major, pitch, hand=1):
    """ISO basic 60-degree profile as a radial height field: root flat p/4 at the minor radius,
    crest flat p/8 at the major radius, straight 60-degree flanks. hand=+1 right-handed."""
    depth = 0.54127 * pitch                     # 5H/8, H = 0.866 p
    phase = np.mod((z - hand * pitch * th / (2 * np.pi)) / pitch, 1.0)
    f = np.interp(phase, [0.0, 0.125, 0.4375, 0.5625, 0.875, 1.0], [0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
    # phase 0-0.125 root, 0.125-0.4375 flank, 0.4375-0.5625 crest (p/8), 0.5625-0.875 flank,
    # 0.875-1.0 root -> root flat = 0.25 p
    return r_major - depth + depth * f


def threaded_stud(d=12.0, pitch=1.75, shank_len=15.0, thread_len=22.0, chamfer=1.0, hand=1,
                  n_theta=144, dz=0.15, noise=0.02, seed=1):
    """Stud: plain shank at the major diameter, then an external ISO thread (cut stops with a step at
    the run-out), 45-degree end chamfer. Built directly as a noisy scan mesh on a (theta, z) grid.
    Returns (mesh, truth)."""
    R = d / 2
    L = shank_len + thread_len
    th = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    zs_shank = np.linspace(0.0, shank_len, int(np.ceil(shank_len / 0.5)) + 1)
    zs_thr = np.linspace(shank_len, L, int(np.ceil(thread_len / dz)) + 1)

    def radius(z, t):
        r = _iso_thread_r(z, t, R, pitch, hand) if z >= shank_len else np.full_like(t, R)
        if z > L - chamfer:
            r = np.minimum(r, R - (z - (L - chamfer)))
        return r

    rows = [(z, np.full_like(th, R)) for z in zs_shank]               # shank, top row at z=shank_len
    rows += [(z, radius(z, th)) for z in zs_thr]                      # thread starts at the same z (step)
    V, F = [], []
    for z, r in rows:
        V.append(np.c_[r * np.cos(th), r * np.sin(th), np.full_like(th, z)])
    V = np.vstack(V)
    n = n_theta
    for i in range(len(rows) - 1):
        a, b = i * n, (i + 1) * n
        for j in range(n):
            j2 = (j + 1) % n
            F.append((a + j, a + j2, b + j2))
            F.append((a + j, b + j2, b + j))
    c0 = len(V); V = np.vstack([V, [[0, 0, 0]]])
    c1 = len(V); V = np.vstack([V, [[0, 0, L]]])
    top = (len(rows) - 1) * n
    for j in range(n):
        j2 = (j + 1) % n
        F.append((c0, j2, j))
        F.append((c1, top + j, top + j2))
    m = trimesh.Trimesh(V, np.array(F), process=True)
    m.update_faces(m.nondegenerate_faces())
    m.remove_unreferenced_vertices()
    trimesh.repair.fix_normals(m)
    rng = np.random.default_rng(seed)
    if noise > 0:
        m.vertices = m.vertices + m.vertex_normals * rng.normal(0, noise, (len(m.vertices), 1))
    # truth volume by integrating r^2/2 over theta and z on a fine grid
    tt = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    zz = np.linspace(shank_len, L, 8001)
    vol_thr = np.trapezoid([np.mean(radius(z, tt) ** 2) * np.pi for z in zz], zz)
    vol = np.pi * R * R * shank_len + vol_thr
    truth = {"thread": {"type": "external", "major_d": d, "pitch": pitch, "hand": "right" if hand > 0 else "left",
                        "z_start": shank_len, "z_end": L, "designation": f"M{d:g}x{pitch:g}"},
             "shank_d": d, "volume": float(vol),
             # what the model should be: thread zone replaced by the major-diameter cylinder, chamfer kept
             "model_volume": float(np.pi * R * R * (L - chamfer)
                                   + np.pi / 3 * chamfer * (R * R + R * (R - chamfer) + (R - chamfer) ** 2))}
    return m, truth


MESH_PARTS = {"threaded_stud": threaded_stud}


def _oval_section(c, t_dir, y_dir, a_inplane, b_out):
    """Closed ellipse wire centred at c in the plane normal to t_dir; b_out along y_dir."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire
    from OCP.gp import gp_Elips
    major, minor = max(a_inplane, b_out), min(a_inplane, b_out)
    # major axis along y (out of plane) when b_out >= a_inplane; equal radii -> circle
    xd = y_dir if b_out >= a_inplane else np.cross(t_dir, y_dir)
    ax = gp_Ax2(gp_Pnt(*map(float, c)), gp_Dir(*map(float, t_dir)), gp_Dir(*map(float, xd)))
    e = BRepBuilderAPI_MakeEdge(gp_Elips(ax, float(major), float(minor))).Edge()
    return BRepBuilderAPI_MakeWire(e).Wire()


def pipe_elbow(r_out=10.0, r_in=7.0, Rb=40.0, oval=0.06, flange_r=21.0, flange_t=6.0, socket_r=8.5,
               n_bolts=4, bolt_d=4.0, pcd=32.0, n_sec=13):
    """90-degree pipe elbow with two bolted flanges. The bend is ovalised (cross-section flattened in the
    bend plane by `oval`, zero at both ends), so no torus fits it: a freeform (B-spline) test part.
    Bend in the XZ plane: starts at (Rb,0,0) heading +z, ends at (0,0,Rb) heading -x."""
    from OCP.BRepOffsetAPI import BRepOffsetAPI_ThruSections

    def loft(r):
        ts = BRepOffsetAPI_ThruSections(True, False, 1e-6)
        for t in np.linspace(0, np.pi / 2, n_sec):
            c = np.array([Rb * np.cos(t), 0.0, Rb * np.sin(t)])
            td = np.array([-np.sin(t), 0.0, np.cos(t)])
            e = oval * np.sin(2 * t)
            ts.AddWire(_oval_section(c, td, np.array([0.0, 1.0, 0.0]), r * (1 - e), r * (1 + e)))
        ts.Build()
        return ts.Shape()

    def flange(c, ax):
        ax = np.asarray(ax, float)
        base = np.asarray(c, float) - ax * flange_t
        f = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(*base), gp_Dir(*ax)), flange_r, flange_t).Shape()
        f = BRepAlgoAPI_Cut(f, BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(*(base - ax)), gp_Dir(*ax)),
                                                        socket_r, flange_t + 2).Shape()).Shape()
        u = np.array([0.0, 1.0, 0.0])
        w = np.cross(ax, u)
        for k in range(n_bolts):
            a = 2 * np.pi * (k + 0.5) / n_bolts
            p = np.asarray(c, float) + pcd / 2 * (np.cos(a) * u + np.sin(a) * w) - ax * (flange_t + 1)
            f = BRepAlgoAPI_Cut(f, BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(*p), gp_Dir(*ax)), bolt_d / 2,
                                                            flange_t + 2).Shape()).Shape()
        return f

    s = BRepAlgoAPI_Cut(loft(r_out), loft(r_in)).Shape()
    s = BRepAlgoAPI_Fuse(s, flange((Rb, 0, 0), (0, 0, 1))).Shape()     # flange below z=0, axis +z
    s = BRepAlgoAPI_Fuse(s, flange((0, 0, Rb), (-1, 0, 0))).Shape()    # flange beyond x=0, axis -x
    from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
    u = ShapeUpgrade_UnifySameDomain(s, True, True, True)
    u.Build()
    s = u.Shape()
    truth = {"r_out": r_out, "r_in": r_in, "bend_radius": Rb, "ovalisation": oval, "flange_r": flange_r,
             "flange_t": flange_t, "n_bolts": n_bolts, "bolt_d": bolt_d, "pcd": pcd, "volume": _vol(s)}
    return s, truth


FREEFORM_PARTS = {"pipe_elbow": pipe_elbow}
