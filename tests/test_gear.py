"""Synthetic involute gears: an exact gear solid is tessellated, noised, and recognised; the recovered
parameters must match the generating ones and the rebuilt gear must be one valid solid."""
import math

import numpy as np
import pytest
import trimesh

from meshsolid import gear as G
from meshsolid import occ_util as U


def _gear_solid(N, mn, pa_deg, beta_deg, hand, face=20.0, shaft_r=None, x=0.0):
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder
    from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt
    beta = math.radians(beta_deg)
    rp = N * mn / (2 * math.cos(beta))
    at = math.atan(math.tan(math.radians(pa_deg)) / math.cos(beta))
    rb = rp * math.cos(at)
    s_n = mn * (math.pi / 2 + 2 * x * math.tan(math.radians(pa_deg)))
    psi_b = s_n / math.cos(beta) / (2 * rp) + G.inv(at)
    r_tip, r_root = rp + (1 + x) * mn, rp - (1.25 - x) * mn
    L = 2 * math.pi * rp / math.tan(beta) if beta_deg else float("inf")
    w = 0.0 if not beta_deg else (2 * math.pi / L) * (1 if hand == "right" else -1)
    a0, u = np.zeros(3), np.array([0.0, 0.0, 1.0])
    blank = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), r_tip, face).Shape()
    sr = shaft_r or 0.6 * r_root
    for z0 in (-12.0, face):
        st = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(0, 0, z0), gp_Dir(0, 0, 1)), sr, 12.0).Shape()
        f = BRepAlgoAPI_Fuse(blank, st); f.Build(); blank = f.Shape()
    tools = G.build_gear_cut(N, rb, psi_b, r_root, 0.3 * mn, r_tip, w, 0.1, 0.0, face, a0, u)
    c = BRepAlgoAPI_Cut(blank, tools); c.Build()
    return c.Shape(), dict(rb=rb, rp=rp, r_tip=r_tip, r_root=r_root, L=L)


def _scan(shape, sigma, seed=0):
    m = U.shape_to_trimesh(shape, lin=0.004, ang=0.05)
    m = trimesh.Trimesh(m.vertices, m.faces, process=True)
    rng = np.random.default_rng(seed)
    m.vertices = m.vertices + rng.normal(0, sigma, m.vertices.shape)
    return m


@pytest.mark.slow
def test_helical_33T_24NDP_20PA_20LH():
    s, ref = _gear_solid(33, 25.4 / 24, 20.0, 20.0, "left", face=20.0)
    m = _scan(s, 0.01)
    g, info = G.analyse(m, tol=0.02, n_samples=400000)
    assert g is not None, info
    sn = g.snapped
    assert g.teeth == 33 and g.hand == "left"
    assert sn["pitch_system"] == "DP" and sn["normal_DP"] == 24.0
    assert sn["normal_pressure_angle_deg"] == 20.0
    assert abs(sn["helix_angle_pitch_deg"] - 20.0) < 0.1
    assert abs(sn["pressure_angle_fit_deg"] - 20.0) < 0.5
    assert abs(sn["profile_shift_x"]) < 0.05 and abs(sn["profile_shift_x_from_thickness"]) < 0.05
    assert sn["tip_snapped"] and abs(sn["tip_diameter_standard"] - 2 * ref["r_tip"]) < 0.01
    assert abs(g.fit["rb"] - ref["rb"]) < 0.03
    assert max(g.fit["flank_plus"]["rms"], g.fit["flank_minus"]["rms"]) < 0.02
    b = sn["build"]
    cut = G.build_gear_cut(g.teeth, b["rb"], b["psi_b"], b["r_root"], b["rho"], b["r_tip"], b["w"],
                           g.fit["clocking_rad"], g.chamfers["z_end0"], g.chamfers["z_end1"],
                           np.asarray(g.axis_point), np.asarray(g.axis_dir))
    assert U.check(cut)


@pytest.mark.slow
def test_spur_20T_module2():
    s, ref = _gear_solid(20, 2.0, 20.0, 0.0, "spur", face=16.0)
    m = _scan(s, 0.01, seed=1)
    g, info = G.analyse(m, tol=0.02, n_samples=400000)
    assert g is not None, info
    sn = g.snapped
    assert g.teeth == 20 and g.hand == "spur"
    assert sn["pitch_system"] == "module" and sn["normal_module"] == 2.0
    assert sn["normal_pressure_angle_deg"] == 20.0
    assert abs(sn["profile_shift_x"]) < 0.05
    assert abs(2 * g.r_root - 2 * ref["r_root"]) < 0.1


def test_involute_space_profile_is_exact():
    """The swept flank is the involute helicoid to < 1e-4 mm."""
    N, mn, beta = 33, 25.4 / 24, math.radians(20.0)
    rp = N * mn / (2 * math.cos(beta)); at = math.atan(math.tan(math.radians(20)) / math.cos(beta)); rb = rp * math.cos(at)
    psi_b = math.pi / 2 * mn / math.cos(beta) / (2 * rp) + G.inv(at)
    L = 2 * math.pi * rp / math.tan(beta); w = -2 * math.pi / L
    comp = G.build_gear_cut(N, rb, psi_b, rp - 1.25 * mn, 0.3 * mn, rp + mn, w, 0.0, 0.0, 10.0, np.zeros(3), np.array([0, 0, 1.0]))
    from OCP.TopoDS import TopoDS_Iterator
    sp = TopoDS_Iterator(comp).Value()
    assert U.check(sp) and U.volume(sp) > 0
    tm = U.shape_to_trimesh(sp, lin=0.001, ang=0.02)
    r, th, z = G.cyl(tm.vertices, np.zeros(3), np.array([0, 0, 1.0]))      # the gear's own frame
    rT = G._fillet_geom(rb, psi_b, rp - 1.25 * mn, 0.3 * mn, N)["rT"]   # above the root fillet
    k = (r > max(rb, rT) + 0.02) & (r < rp + mn) & (z > 0.5) & (z < 9.5)
    thp = th[k] - w * z[k]
    a = np.arccos(np.clip(rb / r[k], -1, 1)); psi = psi_b - (np.tan(a) - a)
    pitch = 2 * np.pi / N
    d = np.minimum(np.abs(np.angle(np.exp(1j * (thp - psi)))), np.abs(np.angle(np.exp(1j * (thp - (pitch - psi)))))) * r[k]
    assert k.sum() > 1000 and np.percentile(d, 99) < 1e-4
