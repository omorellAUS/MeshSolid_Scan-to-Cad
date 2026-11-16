"""Splitter-blade impellers (full + splitter alternating): recognised from the ring vote pattern instead of
being rejected as 'not a bladed rotor'.  Synthetic wheel: 8 full blades over 0.3..1.0 R, 8 splitters
half-way between them over 0.5..1.0 R."""
import numpy as np

from meshsolid import impeller as I


def _wheel(n_full=8, r_split=0.5, R=70.0, seed=0):
    rng = np.random.default_rng(seed)
    rr, th, nt = [], [], []
    for k in range(2 * n_full):
        splitter = k % 2 == 1
        r0 = r_split * R if splitter else 0.3 * R
        r = rng.uniform(r0, R, 4000)
        t = 2 * np.pi * k / (2 * n_full) + 0.004 * (r - r0) / R          # slight wrap
        t = (t + rng.normal(0, 0.003, r.size) + np.pi) % (2 * np.pi) - np.pi
        rr.append(r); th.append(t); nt.append(np.where(rng.random(r.size) < 0.5, 1.0, -1.0))
    # hub/back faces: axial-facing area at all angles (no tangential component)
    rh = rng.uniform(0.1 * R, R, 20000); thh = rng.uniform(-np.pi, np.pi, rh.size)
    r = np.r_[np.concatenate(rr), rh]; t = np.r_[np.concatenate(th), thh]
    n = np.r_[np.concatenate(nt), np.zeros(rh.size)]
    return r, t, n, np.ones(r.size), R


def test_ring_votes_show_full_then_double_count():
    r, th, nt, A, R = _wheel()
    N, votes = I.blade_count(r, th, nt, A, R)
    assert votes == [8, 16, 16, 16]


def test_splitter_pattern_recognised():
    r, th, nt, A, R = _wheel()
    _, votes = I.blade_count(r, th, nt, A, R)
    sp = I.splitter_pattern(votes, tang=0.4)
    assert sp == {"n_full": 8, "n_splitter": 8, "splitter_le_ring_index": 1, "votes": [8, 16, 16, 16]}


def test_plain_and_odd_wheels_are_not_splitters():
    assert I.splitter_pattern([15, 15, 15, 15]) is None          # 165 mm 15-blade wheel: all full blades
    assert I.splitter_pattern([16, 16, 16, 16]) is None          # ring test blind here; per-sector LE test covers it
    assert I.splitter_pattern([8, 16, 8, 16]) is None            # not monotone inner -> outer
    assert I.splitter_pattern([8, 16, 16, 16], tang=0.1) is None  # no blade area


def test_classify_families_alternating_le():
    rng = np.random.default_rng(1)
    le = np.where(np.arange(16) % 2 == 0, 0.0, 0.42) + rng.normal(0, 0.01, 16)
    f = I.classify_families(le)
    assert f["splitter"] and f["splitter_parity"] == 1 and f["n_full"] == 8 and abs(f["p_split"] - 0.42) < 0.02


def test_classify_families_single_family():
    rng = np.random.default_rng(2)
    assert not I.classify_families(rng.normal(0.01, 0.01, 16))["splitter"]
    assert not I.classify_families(rng.normal(0.01, 0.01, 15))["splitter"]          # odd N


def test_splitter_assembly_one_valid_solid_matching_truth():
    """8 full + 8 splitter blades on a hub disc: full blade r 10..40, splitter = the same blade beyond the
    meridional line r = 25 (cut with _region_beyond_line), patterned with _pattern_indexed and fused."""
    import math
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.TopAbs import TopAbs_SOLID
    from OCP.gp import gp_Pnt, gp_Ax1, gp_Dir
    from meshsolid import occ_util as U
    F = I.Frame(np.zeros(3), np.array([0, 0, 1.0]), np.array([1.0, 0, 0]), np.array([0, 1.0, 0]))
    hub = BRepPrimAPI_MakeCylinder(45.0, 4.0).Shape()
    t, h = 1.2, 8.0
    blade = BRepPrimAPI_MakeBox(gp_Pnt(10.0, -t / 2, 3.0), gp_Pnt(40.0, t / 2, 3.0 + h)).Shape()   # 1 mm into the hub
    region = I._region_beyond_line(F, np.c_[np.full(5, 25.0), np.linspace(-2, 15, 5)], 60.0, -5.0)
    sblade = I._common(blade, region)
    # the revolved cut face is the cylinder r = 25 (not a plane): + int (25 - sqrt(625 - y^2)) dy dz
    y = np.linspace(-t / 2, t / 2, 2001); sag = float(np.trapezoid(25.0 - np.sqrt(625.0 - y ** 2), y)) * h
    assert abs(U.volume(sblade) - (15.0 * t * h + sag)) < 1e-3
    ax = gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1))
    blades = I._pattern_indexed(ax, 16, lambda k: sblade if k % 2 == 1 else blade)
    shape = BRepAlgoAPI_Fuse(hub, blades).Shape()
    assert BRepCheck_Analyzer(shape).IsValid() and U.count(shape, TopAbs_SOLID) == 1
    # truth: hub + blade parts above the hub top (z 4 .. 11)
    truth = math.pi * 45.0 ** 2 * 4.0 + 8 * (30.0 * t * (h - 1.0)) + 8 * (15.0 * t * (h - 1.0) + sag * (h - 1.0) / h)
    assert abs(U.volume(shape) - truth) < 0.05


def test_classify_families_small_cutback_cummins140():
    """Real per-blade LE p of the Cummins 140 mm wheel (8+8): small cutback, clearly separated groups."""
    le = [0.0749, 0.0191, 0.0729, 0.0192, 0.0858, 0.0195, 0.0887, 0.0117,
          0.0569, 0.021, 0.0772, 0.0179, 0.0646, 0.0196, 0.0886, 0.0146]
    f = I.classify_families(le)
    assert f["splitter"] and f["splitter_parity"] == 0 and abs(f["p_split"] - 0.076) < 0.002


def test_classify_families_noisy_splitter_le_cummins140_reanalysed():
    """Same wheel after a different axis iteration: the splitters' LE spread is 0.10 (0.046..0.148) but the
    full blades are tight (0.012..0.021) and every splitter is clear of them."""
    le = [0.0871, 0.0185, 0.0852, 0.0185, 0.1059, 0.0189, 0.1478, 0.0123,
          0.0461, 0.0208, 0.0834, 0.0176, 0.0637, 0.019, 0.1261, 0.0147]
    f = I.classify_families(le)
    assert f["splitter"] and f["splitter_parity"] == 0


def test_classify_families_overlapping_groups_rejected():
    le = [0.02, 0.05, 0.06, 0.02, 0.07, 0.03, 0.02, 0.06, 0.05, 0.02, 0.06, 0.03, 0.02, 0.05, 0.07, 0.02]
    assert not I.classify_families(le)["splitter"]


def test_slab_orientation_positive_volume_either_grid_order():
    """_slab_from_grids must give a positive-volume solid whichever side grid comes first (an inverted
    slab - Cummins 140, -4004 mm3 - trims to an empty blade)."""
    import numpy as np
    from meshsolid import impeller as I, occ_util as U
    u, v = np.meshgrid(np.linspace(0, 20, 9), np.linspace(0, 10, 7), indexing="ij")
    a = np.stack([u, v, 0.3 * np.sin(u / 7.0)], -1)
    b = a.copy(); b[..., 2] += 2.0
    vols = [U.volume(I._slab_from_grids(g1, g2)) for g1, g2 in ((a, b), (b, a), (a[::-1], b[::-1]), (b[:, ::-1], a[:, ::-1]))]
    for vol in vols:
        assert abs(vol - 400.0) < 5.0, vols
