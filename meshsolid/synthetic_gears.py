"""Synthetic straight bevel gears and cylindrical worms with known parameters, as scan-like meshes (M12).

Bevel gear (straight teeth, tip and root converging at the pitch-cone apex, Tredgold back-cone profile):
  pitch cone angle delta, outer cone distance Re = m N / (2 sin delta), addendum m, dedendum 1.25 m,
  20 deg involute profile of the virtual gear N_v = N / cos delta developed on the back cone.  Every tooth
  surface line passes through the apex, so the tooth section in any plane normal to the axis is a scaled
  copy of one section; the body is bounded by the back cone and the front cone (face width b), a back
  face, a hub face and a bore.  Built analytically per meridian, then remeshed + noise + pose (scanify),
  independent of the analyser under test.
Worm (ZA, straight-sided axial profile): axial module m, starts z1, diameter quotient q (d1 = q m),
  addendum m, dedendum 1.2 m, axial flank angle 20 deg, tooth thickness p_x / 2 on d1; swept with lead
  z1 pi m (synthetic_screw.rotor_mesh), plain shafts both sides.
"""
from __future__ import annotations

import math

import numpy as np
import trimesh

from .synthetic_screw import rotor_mesh, scanify, trapezoid_profile

ALPHA = math.radians(20.0)


def _inv(a):
    return np.tan(a) - a


def developed_tooth(Nv, m, alpha=ALPHA, ha=1.0, hf=1.25, n=4000):
    """Polar profile rho(phi) of an involute gear with Nv (may be non-integer) teeth, module m: tip land,
    involute flanks, a radial flank below the base circle, root land.  Returns a periodic callable."""
    Rv = m * Nv / 2; rb = Rv * math.cos(alpha); ra = Rv + ha * m; rf = Rv - hf * m
    rlo = max(rb, rf)
    rr = np.linspace(rlo, ra, n)
    psi = math.pi / (2 * Nv) + _inv(alpha) - _inv(np.arccos(np.clip(rb / rr, -1, 1)))     # half tooth angle
    half = math.pi / Nv
    ramp = 0.02 * half                                                         # radial flank -> root land
    if rf < rb:
        ph = np.r_[0.0, psi[::-1], psi[0] + ramp, half]
        rho = np.r_[ra, rr[::-1], rf, rf]
    else:
        ph = np.r_[0.0, psi[::-1], half]
        rho = np.r_[ra, rr[::-1], rf]
    pitch = 2 * half

    def R(phi):
        t = np.mod(np.asarray(phi, float) + half, pitch) - half
        return np.interp(np.abs(t), ph, rho)
    return R, dict(Rv=Rv, rb=rb, ra=ra, rf=rf)


def bevel_mesh(N, m, delta_deg, b, bore=None, n_per_tooth=96, k_tooth=40):
    d = math.radians(delta_deg)
    R = m * N / 2; Re = R / math.sin(d); H = Re * math.cos(d)
    Nv = N / math.cos(d)
    Rd, inf = developed_tooth(Nv, m)
    n_th = N * n_per_tooth
    th = np.linspace(0, 2 * math.pi, n_th, endpoint=False)
    rv = Rd(th * math.cos(d))                                    # developed radius at the matching angle
    r_bc = rv * math.cos(d); z_bc = -R * math.tan(d) + rv * math.sin(d)
    rho0 = r_bc * H / (H - z_bc)                                 # section in the plane z = 0
    cd, sd = math.cos(d), math.sin(d)
    D = rho0 * R + H * H                                         # (Q - A) . (R, -H) * ... (see docstring)
    lam_out = Re * Re / D; lam_in = Re * (Re - b) / D
    A = np.array([0.0, H])

    def ray(lam):                                                # (r, z) on the tooth surface
        return np.c_[lam * rho0, H + lam * (0 - H)]
    Xo, Xi = ray(lam_out), ray(lam_in)
    bore = bore if bore is not None else 0.25 * R
    # back cone below the teeth down to the back face, front cone down to the hub face
    r_bo = max(0.7 * R, bore + 0.2 * R); t_b = (r_bo - R) / cd; zb = t_b * sd
    Rp = R * (1 - b / Re); zp = H * b / Re
    r_fi = max(0.75 * Rp, bore + 0.1 * R); zf = zp + (r_fi - Rp) / cd * sd
    if not zf > zb + 1.0:
        raise ValueError("face width too large for this blank")
    segs = []

    def lin(p, q, k):                                           # k points from p (incl.) to q (excl.)
        t = np.linspace(0, 1, k, endpoint=False)[:, None, None]
        return p[None] * (1 - t) + q[None] * t
    one = np.ones((n_th, 1))
    P1 = np.c_[np.full(n_th, bore), np.full(n_th, zb)]
    P2 = np.c_[np.full(n_th, r_bo), np.full(n_th, zb)]
    P5 = np.c_[np.full(n_th, r_fi), np.full(n_th, zf)]
    P6 = np.c_[np.full(n_th, bore), np.full(n_th, zf)]
    segs = [lin(P1, P2, 10), lin(P2, Xo, 12), lin(Xo, Xi, k_tooth), lin(Xi, P5, 12), lin(P5, P6, 10),
            lin(P6, P1, 16)]
    M = np.concatenate(segs, 0)                                  # (k, n_th, 2) meridian points
    k = M.shape[0]
    X = M[..., 0] * np.cos(th)[None]; Y = M[..., 0] * np.sin(th)[None]; Z = M[..., 1]
    V = np.stack([X, Y, Z], -1).reshape(-1, 3)
    i = np.arange(k)[:, None]; j = np.arange(n_th)[None]
    a = i * n_th + j; bb = i * n_th + (j + 1) % n_th
    c = ((i + 1) % k) * n_th + (j + 1) % n_th; e = ((i + 1) % k) * n_th + j
    F = np.vstack([np.c_[a.ravel(), bb.ravel(), c.ravel()], np.c_[a.ravel(), c.ravel(), e.ravel()]])
    msh = trimesh.Trimesh(V, F, process=True)
    msh.update_faces(msh.nondegenerate_faces()); msh.remove_unreferenced_vertices()
    trimesh.repair.fix_normals(msh)
    th_a = math.atan(m / Re); th_f = math.atan(1.25 * m / Re)
    tr = dict(kind="bevel_straight", teeth=N, module=m, pitch_angle_deg=delta_deg, face_width=b,
              outer_cone_distance=Re, outer_pitch_dia=m * N, outer_tip_dia=m * N + 2 * m * cd,
              tip_angle_deg=math.degrees(d + th_a), root_angle_deg=math.degrees(d - th_f),
              pressure_angle_deg=20.0, apex_height=H, bore_dia=2 * bore, virtual_teeth=Nv)
    return msh, tr


def worm_mesh(m, z1, q, hand=1, L=None, shaft_len=(20.0, 25.0)):
    px = math.pi * m; lead = z1 * px; d1 = q * m
    r_tip, r_root = d1 / 2 + m, d1 / 2 - 1.2 * m
    crest = px / 2 - 2 * m * math.tan(ALPHA)                     # thickness px/2 at d1 (1 m below the tip)
    Rp = trapezoid_profile(z1, lead, r_root, r_tip, flank_deg=20.0, land_frac=crest / px)
    # trapezoid_profile puts the crest at axial phase 0..crest; Rp(phi) maps transverse angle -> radius
    L = L or round(6 * px, 1)
    w = hand * 2 * math.pi / lead
    msh = rotor_mesh(Rp, z1, w, 0.0, L, shaft_r=r_root - 0.6 * m, shaft_len=shaft_len, dz=0.3)
    tr = dict(kind="worm_ZA", starts=z1, module=m, axial_pitch=px, lead=lead, hand="right" if hand > 0 else "left",
              q=q, ref_dia=d1, tip_dia=2 * r_tip, root_dia=2 * r_root, axial_flank_deg=20.0,
              lead_angle_deg=math.degrees(math.atan(z1 / q)), length=L)
    return msh, tr


PARTS = {
    "bevel_mitre_20t_m3": lambda: bevel_mesh(20, 3.0, 45.0, 12.0),
    "bevel_gear_30t_m2": lambda: bevel_mesh(30, 2.0, math.degrees(math.atan(30 / 15)), 10.0),
    "bevel_pinion_15t_m2": lambda: bevel_mesh(15, 2.0, math.degrees(math.atan(15 / 30)), 10.0),
    "worm_m2_z1_q10": lambda: worm_mesh(2.0, 1, 10, +1),
    "worm_m3_z2_q11_left": lambda: worm_mesh(3.0, 2, 11, -1),
    "worm_m1_25_z4_q16": lambda: worm_mesh(1.25, 4, 16, +1),
}


def part(name, edge=0.6, noise=0.03, seed=0):
    clean, tr = PARTS[name]()
    m, T = scanify(clean, edge=edge, noise=noise, seed=seed)
    tr.update(clean_mesh=clean, pose=T)
    return m, tr
