import numpy as np
from meshsolid.balance import BalanceConfig, detect


def _face(z, r0, r1, n, rng):
    r = np.sqrt(rng.uniform(r0 ** 2, r1 ** 2, n)); t = rng.uniform(-np.pi, np.pi, n)
    return np.c_[r * np.cos(t), r * np.sin(t), np.full(n, z)]


def test_balance_cuts_tagged_by_plane():
    rng = np.random.default_rng(0)
    back = _face(10.0, 50, 80, 60000, rng); front = _face(0.0, 8, 35, 20000, rng)
    P = np.r_[back, front]
    Nn = np.r_[np.tile([0, 0, 1.0], (len(back), 1)), np.tile([0, 0, -1.0], (len(front), 1))]
    ds = rng.normal(0, 0.02, len(P))
    ds += 0.05 * P[:, 0] / 80 * (P[:, 2] > 5)                 # face tilt (runout) must not count
    cb = np.array([76 * np.cos(np.radians(40)), 76 * np.sin(np.radians(40)), 10.0])
    cf = np.array([20 * np.cos(np.radians(-100)), 20 * np.sin(np.radians(-100)), 0.0])
    ds[np.linalg.norm(P - cb, axis=1) < 3.0] -= 1.0
    ds[np.linalg.norm(P - cf, axis=1) < 2.0] -= 0.4
    faces = dict(z_back=10.0, z_front=0.0, back_sign=1.0, r_front_out=35.0, r_back_in=50.0)
    cuts, mask = detect(P, ds, Nn, faces, 7.0, 80.0, BalanceConfig())
    by = {c.plane: c for c in cuts}
    assert len(cuts) == 2 and set(by) == {"back", "front"}
    assert abs(by["back"].angle_deg - 40) < 3 and abs(by["back"].depth_mm - 1.0) < 0.15
    assert abs(by["front"].angle_deg + 100) < 5 and abs(by["front"].depth_mm - 0.4) < 0.1
    assert mask.sum() > 0


def test_no_cuts_on_clean_faces():
    rng = np.random.default_rng(1)
    P = np.r_[_face(10.0, 50, 80, 60000, rng), _face(0.0, 8, 35, 20000, rng)]
    Nn = np.where((P[:, 2] > 5)[:, None], [0, 0, 1.0], [0, 0, -1.0])
    ds = rng.normal(0, 0.03, len(P))
    faces = dict(z_back=10.0, z_front=0.0, back_sign=1.0, r_front_out=35.0, r_back_in=50.0)
    cuts, mask = detect(P, ds, Nn, faces, 7.0, 80.0, BalanceConfig())
    assert cuts == [] and not mask.any()
