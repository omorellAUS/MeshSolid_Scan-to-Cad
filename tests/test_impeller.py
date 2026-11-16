import numpy as np
import trimesh

from meshsolid import impeller as I


def _rotor(n_blades, r_in=10.0, r_out=40.0, teeth_only=False):
    parts = [trimesh.creation.cylinder(radius=r_out, height=4.0, sections=256)]
    parts[0].apply_translation([0, 0, 2.0])
    for k in range(n_blades):
        a = 2 * np.pi * k / n_blades
        if teeth_only:      # gear-like: short teeth in a thin ring at the OD
            b = trimesh.creation.box([3.0, 1.5, 4.0]); b.apply_translation([r_out + 1.0, 0, 2.0])
        else:               # blade: thin radial sheet from the hub to the rim, standing on the disc
            b = trimesh.creation.box([r_out - r_in, 1.2, 8.0]); b.apply_translation([(r_in + r_out) / 2, 0, 8.0])
        b = b.subdivide().subdivide().subdivide()
        b.apply_transform(trimesh.transformations.rotation_matrix(a, [0, 0, 1]))
        parts.append(b)
    m = trimesh.util.concatenate(parts)
    return m.subdivide_to_size(1.0) if hasattr(m, "subdivide_to_size") else m


def test_probe_finds_blades():
    n, info = I.probe(_rotor(9))
    assert n == 9, info


def test_probe_rejects_turned_body_and_gear():
    n, _ = I.probe(_rotor(0))
    assert n is None
    n, _ = I.probe(_rotor(24, teeth_only=True))
    assert n is None


def test_edge_round_recovers_radius():
    rho, t2 = 0.6, 1.0
    h = np.linspace(0.02, 1.5, 15)
    w = np.where(h < rho, t2 - rho + np.sqrt(np.clip(rho ** 2 - (rho - h) ** 2, 0, None)), t2)
    out = I.edge_round(h, w)
    assert abs(out["rho"] - rho) < 0.05 and abs(out["half_t"] - t2) < 0.05
