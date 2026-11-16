"""Meridional-curve fallbacks that replaced 'ValueError: no crossing' crashes (KK&K stage 1, 165 mm 15-blade)."""
import numpy as np
import pytest
from scipy.interpolate import UnivariateSpline

from meshsolid import impeller as I


def _curve(r, z):
    """meridional curve as the analyser stores it: s(t) with t = r - z, s = r + z, linearly extended."""
    t, s = r - z, r + z
    o = np.argsort(t)
    sp = UnivariateSpline(t[o], s[o], k=3, s=0)
    return I._Ext1(sp, t[o][0], t[o][-1], end_window=0.5)


def test_z_or_end_returns_front_end_when_hub_never_reaches_le_plane():
    r = np.linspace(20, 100, 60); z = 75 - 0.004 * (r - 20) ** 2          # hub nose at z 75
    sp = _curve(r, z)
    t, reached = I.t_at_z_or_end(sp, 70.0, sp.t0 - 20, sp.t1, sp.t0)
    assert reached and abs((sp(t) - t) / 2 - 70.0) < 1e-6
    t, reached = I.t_at_z_or_end(sp, 77.98, sp.t0 - 20, sp.t1, sp.t0)    # KK&K: LE plane above hub nose
    assert not reached and t == sp.t0


def test_z_closest_accepts_small_shortfall_and_rejects_large():
    r = np.linspace(14, 80, 60); z = 52.11 - 0.008 * (r - 20) ** 2         # tip curve peaks 0.07 under z_le, then rounds away
    sp = _curve(r, z)
    with pytest.raises(ValueError):
        I.t_at_z(sp, 52.183, sp.t0 - 20, sp.t1)
    t, gap = I.t_at_z_closest(sp, 52.183, sp.t0 - 20, sp.t1)
    assert 0 < gap < 0.2
    with pytest.raises(ValueError):
        I.t_at_z_closest(sp, 60.0, sp.t0 - 20, sp.t1)


def test_r_wide_reaches_rim_when_end_slope_bends_away():
    r = np.r_[np.linspace(30, 108, 50), 108 + 2.7 * np.sin(np.linspace(0, np.pi / 2, 12))[1:]]
    z = np.r_[np.linspace(60, 10, 50), 10 - 6 * (1 - np.cos(np.linspace(0, np.pi / 2, 12)))[1:]]  # TE round, r ends 110.7
    sp = _curve(r, z)
    with pytest.raises(ValueError):
        I.t_at_r(sp, 119.0, sp.t0, sp.t1 + 10)
    t, c = I.t_at_r_wide(sp, 119.0, sp.t0, sp.t1 + 10)
    assert c is not sp and abs((c(t) + t) / 2 - 119.0) < 1e-6      # the caller must build on the widened curve
    assert np.allclose(c(np.linspace(sp.t0, sp.t1, 20)), sp(np.linspace(sp.t0, sp.t1, 20)))  # same inside the data
