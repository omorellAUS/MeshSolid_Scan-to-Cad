import numpy as np
import pytest

from meshsolid import fit as F

rng = np.random.default_rng(0)


def _u(v):
    return v / np.linalg.norm(v)


def test_plane():
    n = _u(np.array([0.2, -0.3, 0.9]))
    u = _u(np.cross(n, [1, 0, 0])); v = np.cross(n, u)
    P = np.array([5, 1, 2]) + rng.uniform(-10, 10, (500, 1)) * u + rng.uniform(-10, 10, (500, 1)) * v
    P += rng.normal(0, 0.01, (500, 1)) * n
    p = F.fit_plane(P, np.tile(n, (500, 1)))
    assert abs(p.params["normal"] @ n) > 1 - 1e-5
    assert abs((p.params["point"] - [5, 1, 2]) @ n) < 0.003
    assert p.rms == pytest.approx(0.01, rel=0.2)


def test_cylinder():
    a = _u(np.array([0.1, 0.2, 1.0]))
    u, v = F._perp_basis(a)
    t = rng.uniform(0, 1.5 * np.pi, 800); h = rng.uniform(-10, 10, 800)
    R = 7.3
    N = np.cos(t)[:, None] * u + np.sin(t)[:, None] * v
    P = np.array([1, 2, 3]) + h[:, None] * a + R * N + rng.normal(0, 0.01, (800, 1)) * N
    p = F.fit_cylinder(P, N)
    assert p.params["radius"] == pytest.approx(R, abs=0.005)
    assert abs(p.params["axis"] @ a) > 1 - 1e-5


def test_sphere():
    N = rng.normal(size=(600, 3)); N /= np.linalg.norm(N, axis=1, keepdims=True)
    N = N[N[:, 2] > 0]  # hemisphere only
    P = np.array([3, -1, 4]) + 12.5 * N + rng.normal(0, 0.01, (len(N), 1)) * N
    p = F.fit_sphere(P, N)
    assert p.params["radius"] == pytest.approx(12.5, abs=0.005)
    assert np.allclose(p.params["center"], [3, -1, 4], atol=0.01)


def test_cone():
    a = _u(np.array([0.0, 0.3, 1.0]))
    apex = np.array([1.0, 2.0, -3.0]); ha = np.radians(25)
    u, v = F._perp_basis(a)
    t = rng.uniform(0, 2 * np.pi, 1000); h = rng.uniform(3, 15, 1000)
    rdir = np.cos(t)[:, None] * u + np.sin(t)[:, None] * v
    P = apex + h[:, None] * a + (h * np.tan(ha))[:, None] * rdir
    N = rdir * np.cos(ha) - a * np.sin(ha)
    p = F.fit_cone(P + rng.normal(0, 0.005, (1000, 1)) * N, N)
    assert np.degrees(p.params["half_angle"]) == pytest.approx(25, abs=0.1)
    assert np.allclose(p.params["apex"], apex, atol=0.05)


def test_fit_best_prefers_simplest():
    P = np.c_[rng.uniform(0, 10, (300, 2)), rng.normal(0, 0.005, 300)]
    N = np.tile([0, 0, 1.0], (300, 1))
    best, _ = F.fit_best(P, N, tol=0.05)
    assert best.kind == "plane"


@pytest.mark.parametrize("sign", [1, -1])
def test_torus_partial_arc(sign):
    a = _u(np.array([0.2, 1.0, -0.1]))
    u, v = F._perp_basis(a)
    c = np.array([3.0, -2.0, 5.0]); R, r = 21.0, 2.5
    t = rng.uniform(0, 2 * np.pi, 1500)       # around the axis
    s = rng.uniform(0, 0.5 * np.pi, 1500)     # quarter of the tube (a fillet)
    rad = np.cos(t)[:, None] * u + np.sin(t)[:, None] * v
    n = np.cos(s)[:, None] * rad + np.sin(s)[:, None] * a
    P = c + R * rad + r * n
    p = F.fit_torus(P + rng.normal(0, 0.005, (1500, 1)) * n, sign * n)
    assert p is not None
    assert p.params["major_radius"] == pytest.approx(R, abs=0.02)
    assert p.params["minor_radius"] == pytest.approx(r, abs=0.02)
    assert abs(p.params["axis"] @ a) > 1 - 1e-4


def test_draft_dimensions_from_planes():
    from types import SimpleNamespace as NS
    import numpy as np
    from meshsolid.pipeline import draft_dimensions

    def plane(n):
        n = np.asarray(n, float)
        return NS(kind="plane", params={"normal": n / np.linalg.norm(n), "point": np.zeros(3)})
    a = np.radians(5.0)
    prims = {0: plane([0, 0, 1]), 1: plane([0, 0, -1]), 2: plane([np.cos(a), 0, np.sin(a)]),
             3: plane([0, -np.cos(a), np.sin(a)]), 4: plane([1, 0, 0])}
    regs = [NS(id=i, faces=list(range(100 if i < 2 else 10)), area=1000.0 if i < 2 else 50.0) for i in prims]
    d = draft_dimensions(regs, prims)
    assert sorted(x["regions"][0] for x in d) == [2, 3]   # the vertical plane (4) has no draft
    assert all(abs(x["value"] - 5.0) < 1e-9 for x in d)
