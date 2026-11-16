"""Scan assembled from two captures that were never merged: the faces of the two shells that look at
each other across the gap are left out of the revolve profile (pinion X2.B1872, z -95.5 / -93)."""
import numpy as np
import trimesh

from meshsolid.revolve import _joint_caps


class _Res:
    def __init__(self):
        self.profile_refits = []


def _cyl(z0, z1, r=10.0):
    c = trimesh.creation.cylinder(radius=r, height=z1 - z0, sections=96)
    c.apply_translation([0, 0, 0.5 * (z0 + z1)])
    return c


def test_two_shells_facing_caps_flagged_only():
    m = trimesh.util.concatenate([_cyl(-30, -1.2), _cyl(1.0, 30, r=14.0)])
    res = _Res()
    keep = _joint_caps(m, [0, 0, 0], [0, 0, 1.0], res)
    assert keep is not None
    C = m.triangles_center; n = m.face_normals
    lower_top = (np.abs(C[:, 2] + 1.2) < 1e-6) & (n[:, 2] > 0.9)
    upper_bot = (np.abs(C[:, 2] - 1.0) < 1e-6) & (n[:, 2] < -0.9) & (np.hypot(C[:, 0], C[:, 1]) < 9.5)
    assert lower_top.any() and upper_bot.any()
    assert keep[lower_top].mean() > 0.9 and keep[upper_bot].mean() > 0.9
    side = np.abs(n[:, 2]) < 0.1
    assert not keep[side].any()                      # outer cylinders are real surfaces
    far = (np.abs(C[:, 2]) > 20)
    assert not keep[far].any()
    assert res.profile_refits[-1]["shells"] == 2


def test_single_shell_with_groove_is_untouched():
    m = _cyl(-30, 30)
    assert _joint_caps(m, [0, 0, 0], [0, 0, 1.0], _Res()) is None


def test_shells_too_far_apart_not_flagged():
    m = trimesh.util.concatenate([_cyl(-30, -5.0), _cyl(5.0, 30)])
    assert _joint_caps(m, [0, 0, 0], [0, 0, 1.0], _Res()) is None
