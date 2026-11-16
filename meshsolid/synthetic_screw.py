"""Synthetic helical screw rotors with known parameters, as scan-like meshes (M8 test data).

A rotor is a transverse polar profile r(phi) (period 2 pi / N) swept helically: the surface point at
height z and angle theta has r = R(theta - Theta(z)), Theta(z) = cumulative twist (rad); constant twist
rate w = dTheta/dz (rad/mm), w > 0 right hand, lead = 2 pi / |w|.  Rotor ends are flat, with a turned
journal shaft on each side.  The mesh is built from the analytic surface (independent of the OCC builder
under test), then remeshed irregularly, displaced along the normals with Gaussian noise and placed in an
arbitrary pose (Artec-like).

Profiles (star-shaped about the axis, as transverse helicoid sections of these rotors are):
  male_profile       asymmetric lobes (SRM-like: steep leading flank, long trailing flank)
  female_profile     generated as the envelope of the male (ratio N_m / N_f), as a hob would cut it
  trapezoid_profile  screw-pump thread: trapezoidal axial thread form mapped to the transverse section
  cycloid_profile    triple-screw (Imo-type) style lobes with epi/hypocycloidal flanks
"""
from __future__ import annotations

import math

import numpy as np
import trimesh

TWO_PI = 2 * math.pi


def _periodic_interp(grid, vals):
    g = np.r_[grid - TWO_PI, grid, grid + TWO_PI]; v = np.r_[vals, vals, vals]

    def R(phi):
        return np.interp(np.mod(np.asarray(phi, float), TWO_PI), g, v)
    return R


# ----------------------------------------------------------------------------- profiles r(phi)
def male_profile(N=4, r_root=24.0, r_tip=40.0, skew=0.65, sharp=1.6):
    """Asymmetric lobes; skew != 0.5 moves the crest off the lobe centre."""
    def R(phi):
        t = np.mod(np.asarray(phi, float) * N / TWO_PI, 1.0)
        s = np.where(t < skew, 0.5 * t / skew, 0.5 + 0.5 * (t - skew) / (1 - skew))
        return r_root + (r_tip - r_root) * np.sin(math.pi * s) ** sharp
    return R


def female_profile(male, N_m=4, N_f=6, r_root_m=24.0, r_tip_m=40.0, n_psi=2000, n_bins=6144):
    """Female transverse profile = envelope of the male in the female's frame.  Centre distance from the
    pitch circles (male pitch circle at mid-lobe height, radii in ratio N_m : N_f).  Returns (R, info)."""
    r_pm = 0.5 * (r_tip_m + r_root_m)
    A = r_pm * (1 + N_f / N_m)
    r_tip_f = A - r_root_m                                    # female blank touches the male root circle
    phis = np.linspace(0, TWO_PI, 1440 * N_m, endpoint=False)
    rm = male(phis); pm = np.c_[rm * np.cos(phis), rm * np.sin(phis)]
    env = np.full(n_bins, r_tip_f)
    for psi in np.linspace(0, TWO_PI / N_m, n_psi, endpoint=False):
        c, s = math.cos(psi), math.sin(psi)
        q = pm @ np.array([[c, s], [-s, c]]) + np.array([A, 0.0])      # male turned by psi, centre (A, 0)
        a = psi * N_m / N_f                                              # female turns -a: body frame = Rot(+a)
        ca, sa = math.cos(a), math.sin(a)
        q = q @ np.array([[ca, sa], [-sa, ca]])
        rq = np.hypot(q[:, 0], q[:, 1]); k = rq < r_tip_f
        b = (np.mod(np.arctan2(q[k, 1], q[k, 0]), TWO_PI) / TWO_PI * n_bins).astype(int) % n_bins
        np.minimum.at(env, b, rq[k])
    per = n_bins // N_f
    lobe = env[: per * N_f].reshape(N_f, per).min(0)          # all female lobes equal
    full = np.tile(lobe, N_f)
    grid = (np.arange(len(full)) + 0.5) * TWO_PI / len(full)
    return _periodic_interp(grid, full), {"A": A, "r_pitch_m": r_pm, "r_pitch_f": A - r_pm,
                                          "r_root_f": float(full.min()), "r_tip_f": float(full.max())}


def trapezoid_profile(starts=2, lead=40.0, r_root=14.0, r_tip=22.0, flank_deg=15.0, land_frac=0.35):
    """Screw-pump screw with a trapezoidal axial thread form (crest land, flanks at flank_deg from the
    radial, root land); axial pitch = lead / starts.  Transverse section r(phi) = r_ax(phi lead / 2 pi)."""
    p = lead / starts
    h = r_tip - r_root
    fl = h * math.tan(math.radians(flank_deg))
    crest = land_frac * p
    root = p - crest - 2 * fl
    if root <= 0:
        raise ValueError("flanks too wide for the pitch")

    def r_ax(z):
        t = np.mod(z, p)
        return np.select([t < crest, t < crest + fl, t < crest + fl + root],
                         [np.full_like(t, r_tip), r_tip - (t - crest) / fl * h, np.full_like(t, r_root)],
                         r_root + (t - crest - fl - root) / fl * h)

    def R(phi):
        return r_ax(np.asarray(phi, float) / TWO_PI * lead)
    return R


def cycloid_profile(N=2, r_root=10.0, r_tip=17.0, n=200000, n_bins=8192):
    """N lobes: epicycloidal crests (rolling circle outside the pitch circle) and hypocycloidal troughs
    (inside), angularly scaled so crest and trough each span pi / N.  Polar form via the outer envelope."""
    rp = 0.5 * (r_root + r_tip); rho = 0.25 * (r_tip - r_root)        # cusps reach rp +- 2 rho
    span = TWO_PI * rho / rp                                   # pitch-circle angle of one rolling cusp
    t = np.linspace(0, TWO_PI, n)
    ex = (rp + rho) * np.cos(t * rho / rp) - rho * np.cos((rp + rho) / rp * t)
    ey = (rp + rho) * np.sin(t * rho / rp) - rho * np.sin((rp + rho) / rp * t)
    hx = (rp - rho) * np.cos(t * rho / rp) + rho * np.cos((rp - rho) / rp * t)
    hy = (rp - rho) * np.sin(t * rho / rp) - rho * np.sin((rp - rho) / rp * t)
    ae = np.arctan2(ey, ex) / span * (math.pi / N); re = np.hypot(ex, ey)
    ah = np.arctan2(hy, hx) / span * (math.pi / N) + math.pi / N; rh = np.hypot(hx, hy)
    A = np.r_[ae, ah]; Rr = np.r_[re, rh]
    env = np.zeros(n_bins)
    for k in range(N):
        b = (np.mod(A + TWO_PI * k / N, TWO_PI) / TWO_PI * n_bins).astype(int) % n_bins
        np.maximum.at(env, b, Rr)
    grid = (np.arange(n_bins) + 0.5) * TWO_PI / n_bins
    ok = env > 0
    return _periodic_interp(grid[ok], env[ok])


# ----------------------------------------------------------------------------- mesh
def rotor_mesh(R, N, twist, z0, z1, shaft_r, shaft_len=(20.0, 30.0), n_phi=None, dz=0.4):
    """Watertight helical rotor mesh.  twist: constant rate w (rad/mm) or a callable Theta(z)."""
    Theta = twist if callable(twist) else (lambda z, w=float(twist): w * np.asarray(z, float))
    n_phi = n_phi or max(720, 240 * N)
    ph = np.linspace(0, TWO_PI, n_phi, endpoint=False)
    zs = np.linspace(z0, z1, max(3, int(math.ceil((z1 - z0) / dz)) + 1))

    def ring(rr, z):
        return np.c_[rr * np.cos(ph), rr * np.sin(ph), np.full(n_phi, z)]
    V = [ring(R(ph - Theta(z)), z) for z in zs]
    nz = len(zs)
    i = np.arange(n_phi); i1 = (i + 1) % n_phi
    F = []
    for j in range(nz - 1):
        a, b, c, d = j * n_phi + i, j * n_phi + i1, (j + 1) * n_phi + i1, (j + 1) * n_phi + i
        F += [np.c_[a, b, c], np.c_[a, c, d]]
    nv = nz * n_phi

    def cap(jring, z, z_end, sgn):
        nonlocal nv
        V.extend([ring(np.full(n_phi, shaft_r), z), ring(np.full(n_phi, shaft_r), z_end),
                  np.array([[0.0, 0.0, z_end]])])
        c0, c1, ctr = nv, nv + n_phi, nv + 2 * n_phi
        nv += 2 * n_phi + 1
        p0, p1 = jring * n_phi + i, jring * n_phi + i1
        tri = np.vstack([np.c_[p0, c0 + i1, p1], np.c_[p0, c0 + i, c0 + i1], np.c_[c0 + i, c1 + i1, c0 + i1],
                         np.c_[c0 + i, c1 + i, c1 + i1], np.c_[c1 + i, np.full(n_phi, ctr), c1 + i1]])
        F.append(tri if sgn < 0 else tri[:, [0, 2, 1]])
    cap(0, z0, z0 - shaft_len[0], -1)
    cap(nz - 1, z1, z1 + shaft_len[1], +1)
    m = trimesh.Trimesh(np.vstack(V), np.vstack(F), process=True)
    trimesh.repair.fix_normals(m)
    return m


def scanify(m, edge=0.6, noise=0.03, seed=0, pose=True):
    """Irregular isotropic remesh + Gaussian normal noise + arbitrary pose (Artec-like)."""
    import pymeshlab
    ms = pymeshlab.MeshSet(); ms.add_mesh(pymeshlab.Mesh(m.vertices, m.faces))
    ms.meshing_isotropic_explicit_remeshing(iterations=5, targetlen=pymeshlab.PureValue(edge), featuredeg=170.0)   # scanners keep no crisp edges
    ms.meshing_isotropic_explicit_remeshing(iterations=2, targetlen=pymeshlab.PureValue(edge), featuredeg=170.0)
    mm = ms.current_mesh()
    out = trimesh.Trimesh(mm.vertex_matrix(), mm.face_matrix(), process=True)
    rng = np.random.default_rng(seed)
    T = np.eye(4)
    if pose:
        T = trimesh.transformations.random_rotation_matrix(rng.random(3))
        T[:3, 3] = rng.uniform(-50, 50, 3)
        out.apply_transform(T)
    if noise > 0:
        out.vertices = out.vertices + out.vertex_normals * rng.normal(0, noise, (len(out.vertices), 1))
    return out, T


# ----------------------------------------------------------------------------- test parts with truth
PARTS = ("male_4", "female_6", "pump_twin", "pump_triple", "variable_pitch")


def part(name, seed=0, noise=0.03, edge=0.6):
    """(scan mesh, truth dict) for a named test part.  truth['R'] is the exact transverse profile at z=0
    (rotor frame), truth['pose'] the 4x4 rotor-frame -> scan transform."""
    if name == "male_4":
        R = male_profile(4, 24.0, 40.0); lead, L = 300.0, 120.0; w = TWO_PI / lead
        tr = dict(lobes=4, lead_mm=lead, hand="right", r_tip=40.0, r_root=24.0, shaft_r=15.0,
                  kind="screw_compressor_male")
    elif name == "female_6":
        R, inf = female_profile(male_profile(4, 24.0, 40.0), 4, 6, 24.0, 40.0)
        lead, L = 300.0 * 6 / 4, 120.0; w = -TWO_PI / lead      # female lead = male lead N_f / N_m, other hand
        tr = dict(lobes=6, lead_mm=lead, hand="left", r_tip=inf["r_tip_f"], r_root=inf["r_root_f"], shaft_r=20.0,
                  kind="screw_compressor_female", centre_distance=inf["A"])
    elif name == "pump_twin":
        lead, L = 40.0, 100.0; R = trapezoid_profile(2, lead, 14.0, 22.0); w = TWO_PI / lead
        tr = dict(lobes=2, lead_mm=lead, hand="right", r_tip=22.0, r_root=14.0, shaft_r=9.0, kind="screw_pump")
    elif name == "pump_triple":
        lead, L = 36.0, 90.0; R = cycloid_profile(2, 10.0, 17.0); w = -TWO_PI / lead
        tr = dict(lobes=2, lead_mm=lead, hand="left", r_tip=17.0, r_root=10.0, shaft_r=7.0, kind="screw_pump")
    elif name == "variable_pitch":
        R = trapezoid_profile(1, 30.0, 12.0, 18.0, flank_deg=10.0, land_frac=0.3)
        L = 120.0; lead0, lead1 = 30.0, 50.0
        w0, w1 = TWO_PI / lead0, TWO_PI / lead1
        w = lambda z: w0 * np.asarray(z, float) + 0.5 * (w1 - w0) / L * np.asarray(z, float) ** 2
        tr = dict(lobes=1, lead_mm=[lead0, lead1], hand="right", r_tip=18.0, r_root=12.0, shaft_r=8.0,
                  kind="screw_pump", variable_pitch=True, Theta=w)
    else:
        raise KeyError(name)
    tr["length"] = L
    tw = tr["Theta"](L) if callable(w) else w * L
    tr["wrap_deg"] = abs(math.degrees(tw))
    tr["R"] = R
    m = rotor_mesh(R, tr["lobes"], w, 0.0, L, shaft_r=tr["shaft_r"])
    s, T = scanify(m, noise=noise, seed=seed, edge=edge)
    tr["pose"] = T; tr["clean_mesh"] = m
    return s, tr
