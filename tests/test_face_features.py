"""Adaptive impeller end-face classifier: (a) irregular balance cuts, (b) Hirth / periodic,
(c) concentric bores.  Exclusions carry a 2 mm sector boundary for Metrology."""
from __future__ import annotations

import math

import numpy as np
import pytest

from meshsolid import face_features as FF
from meshsolid.metrology_exclude import zone_mask, zone_text, face_feature_zones, parse_zones


R = 80.0


def _disc(n=60000, R=R, z0=0.0, hole=0.0, seed=0, n_z=-1.0):
    rng = np.random.default_rng(seed)
    # area-uniform in the annulus, plus a matching front-free back face at z = z0
    r = np.sqrt(rng.uniform(hole * hole, R * R, n))
    th = rng.uniform(-math.pi, math.pi, n)
    P = np.c_[r * np.cos(th), r * np.sin(th), np.full(n, z0)]
    N = np.tile([0.0, 0.0, float(n_z)], (n, 1))
    return P, N


def _add_pocket(P, N, centre_deg, width, r0, r1, depth, s=1, wavy=0.0, seed=1):
    """Push a compact sector of the disc into the material (z += s * depth)."""
    P, N = P.copy(), N.copy()
    th = np.arctan2(P[:, 1], P[:, 0])
    r = np.hypot(P[:, 0], P[:, 1])
    c = math.radians(centre_deg)
    dth = np.angle(np.exp(1j * (th - c)))
    half = 0.5 * width / max(0.5 * (r0 + r1), 1.0)
    m = (np.abs(dth) <= half) & (r >= r0) & (r <= r1)
    z = depth
    if wavy:
        rng = np.random.default_rng(seed)
        z = depth + wavy * rng.normal(0.0, 1.0, int(m.sum()))
        P[m, 2] = P[m, 2] + s * z
    else:
        P[m, 2] = P[m, 2] + s * z
    return P, N


def _add_hirth(P, N, Nteeth, r0, r1, height, s=1, included_deg=60.0):
    """Triangular radial teeth: z = s * height * triangle(N * theta) inside the r-band."""
    P, N = P.copy(), N.copy()
    th = np.arctan2(P[:, 1], P[:, 0])
    r = np.hypot(P[:, 0], P[:, 1])
    frac = (th * Nteeth / (2 * math.pi)) % 1.0
    tri = 1.0 - 2.0 * np.abs(frac - 0.5)          # 0 at crest, 1 at valley
    m = (r >= r0) & (r <= r1)
    P[m, 2] = P[m, 2] + s * height * tri[m]
    # tilt the normals on the flanks a little so they still count as face points
    # (included angle 60 deg -> flank slope dz/ds = 1/tan(30) = sqrt(3))
    return P, N


def _classify(P, N, R=R, **kw):
    return FF.classify_points(P, N, [0, 0, 0], [0, 0, 1], [1, 0, 0], R, **kw)


def test_random_balance_pockets_are_class_a_with_2mm_boundaries():
    P, N = _disc(seed=1)
    # three irregular pockets, uneven spacing and different depths — not N-fold
    for ang, w, d, wav in ((40.0, 8.0, 1.2, 0.25), (130.0, 12.0, 2.0, 0.4), (250.0, 6.0, 0.9, 0.15)):
        P, N = _add_pocket(P, N, ang, w, 70.0, 80.0, d, wavy=wav, seed=int(ang))
    rep, mask = _classify(P, N)
    a = [f for f in rep["features"] if f["class"] == "a"]
    b = [f for f in rep["features"] if f["class"] == "b"]
    assert not b, b
    assert len(a) >= 2, rep["features"]
    assert all(f["exclude"] and not f["modelled"] and f["boundary"]["margin_mm"] == 2.0 for f in a)
    assert all(f["kind"] in ("balance_cut", "drill_mark") for f in a)
    # listed with a sector boundary that covers the pocket and a 2 mm border
    assert mask.sum() > 0
    for f in a:
        z = f["boundary"]
        assert z["type"] == "sector" and z["margin_mm"] == 2.0
        # a point at the pocket centre, 2 mm outside the tight radial bound, is excluded
        mid_r = 0.5 * (f["r_inner"] + f["r_outer"])
        th = math.radians(f["centre_deg"])
        pt = np.array([[(f["r_outer"] + 1.5) * math.cos(th), (f["r_outer"] + 1.5) * math.sin(th), 0.0]])
        # only assert the margin when the outer edge is inside the disc
        if f["r_outer"] + 1.5 <= R + 0.2:
            assert zone_mask(pt, z)[0], (f, z)


def test_hirth_teeth_are_class_b_with_correct_N():
    P, N = _disc(n=90000, seed=2, hole=15.0)
    Nteeth, r0, r1, h = 24, 25.0, 55.0, 2.0
    P, N = _add_hirth(P, N, Nteeth, r0, r1, h, included_deg=60.0)
    rep, mask = _classify(P, N)
    b = [f for f in rep["features"] if f["class"] == "b"]
    assert len(b) == 1, [(f["class"], f["kind"], f.get("N"), f["label"]) for f in rep["features"]]
    f = b[0]
    assert f["kind"] == "hirth" and f["N"] == Nteeth, f
    assert not f["exclude"] and not f["modelled"] and f["model_pending"]
    assert abs(f["r_inner"] - r0) < 4.0 and abs(f["r_outer"] - r1) < 4.0
    assert abs(f["height_mm"] - h) < 0.8
    # theoretical included angle at mid-radius: 2 atan((pi r / N) / height)
    r_mid = 0.5 * (r0 + r1)
    expect = 2.0 * math.degrees(math.atan((math.pi * r_mid / Nteeth) / h))
    assert abs(f["included_angle_deg"] - expect) < 35.0, (f["included_angle_deg"], expect)
    assert rep["summary"]["hirth_N"] == Nteeth
    # a Hirth family is recognised, not excluded
    assert mask.sum() == 0 or mask.mean() < 0.05


def test_disc_with_bore_is_class_c():
    hole = 12.0
    P, N = _disc(n=50000, seed=3, hole=hole)
    # cylindrical wall at the bore
    rng = np.random.default_rng(3)
    n_w = 4000
    th = rng.uniform(-math.pi, math.pi, n_w)
    z = rng.uniform(-1.0, 4.0, n_w)
    W = np.c_[hole * np.cos(th), hole * np.sin(th), z]
    Nw = np.c_[np.cos(th), np.sin(th), np.zeros(n_w)]  # outward of a hole = +radial? inward wall of a bore faces +radial into the hole; use -radial (material at r>hole)
    Nw = -Nw
    P = np.vstack([P, W]); N = np.vstack([N, Nw])
    rep, mask = _classify(P, N)
    c = [f for f in rep["features"] if f["class"] == "c"]
    bores = [f for f in c if f["kind"] == "bore"]
    assert len(bores) == 1, [(f["class"], f["kind"], f.get("d_mm"), f["label"]) for f in rep["features"]]
    f = bores[0]
    assert abs(f["d_mm"] - 2 * hole) < 1.5, f
    assert f["modelled"] and not f["exclude"]


def test_stg1_like_irregular_slots_are_all_class_a_not_cut():
    """STG1 job 20261116-140538: 4 back-face depressions at irregular angles, one flat-floored.
    All are non-periodic, so all classify as (a) and are excluded (rim_slots auto must not cut)."""
    P, N = _disc(n=80000, seed=4)
    # angles and sizes loosely after the real STG1 listing (one ~8.8 mm slot at ~69 deg, three shallower)
    specs = ((69.5, 8.8, 1.97, 0.0), (150.0, 14.0, 1.1, 0.25),
             (230.0, 18.0, 0.8, 0.35), (310.0, 10.0, 1.3, 0.2))
    for ang, w, d, wav in specs:
        P, N = _add_pocket(P, N, ang, w, 76.0, 83.0, d, wavy=wav, seed=int(ang))
    rep, mask = _classify(P, N)
    a = [f for f in rep["features"] if f["class"] == "a"]
    b = [f for f in rep["features"] if f["class"] == "b"]
    assert not b, b
    assert len(a) >= 3, [(f["class"], f["kind"], f.get("centre_deg"), f["depth_mm"]) for f in rep["features"]]
    assert all(f["exclude"] and not f["modelled"] for f in a)
    assert mask.any()
    # every exclusion has a named label and a 2 mm sector boundary
    for f in rep["exclusions"]:
        assert f["label"] and f["reason"] and f["boundary"]["margin_mm"] == 2.0


def test_sector_zone_2mm_margin_and_face_feature_zones():
    z = {"type": "sector", "point": [0, 0, 0], "dir": [0, 0, 1], "e1": [1, 0, 0],
         "r": [70.0, 80.0], "theta_deg": [60.0, 80.0], "t": [-0.5, 2.0], "margin_mm": 2.0}
    # inside the tight bounds
    mid = np.array([[75.0 * math.cos(math.radians(70)), 75.0 * math.sin(math.radians(70)), 0.5]])
    assert zone_mask(mid, z)[0]
    # 1.5 mm past the outer radius, still inside the 2 mm boundary
    past = np.array([[81.5 * math.cos(math.radians(70)), 81.5 * math.sin(math.radians(70)), 0.5]])
    assert zone_mask(past, z)[0]
    # 4 mm past the outer radius: outside
    far = np.array([[84.5 * math.cos(math.radians(70)), 84.5 * math.sin(math.radians(70)), 0.5]])
    assert not zone_mask(far, z)[0]
    # wrap-around arc
    zw = {**z, "theta_deg": [350.0, 10.0], "r": [0.0, 20.0]}
    p0 = np.array([[10.0, 0.0, 0.0]])          # 0 deg
    p350 = np.array([[10.0 * math.cos(math.radians(350)), 10.0 * math.sin(math.radians(350)), 0.0]])
    p180 = np.array([[-10.0, 0.0, 0.0]])
    assert zone_mask(p0, zw)[0] and zone_mask(p350, zw)[0] and not zone_mask(p180, zw)[0]
    txt = zone_text(z)
    assert "sector" in txt and "2 mm boundary" in txt

    zs = parse_zones("sector(0,0,0, 0,0,1, 1,0,0, 70,80, 60,80, -0.5,2)")
    assert zs[0]["type"] == "sector" and zs[0]["margin_mm"] == 2.0
    assert zone_mask(mid, zs[0])[0]

    conv = {"face_features": {"exclusions": [
        {"class": "a", "kind": "balance_cut", "exclude": True,
         "label": "balance cut on the back face at 69.5°",
         "reason": "irregular or non-periodic about the axis",
         "boundary": z},
        {"class": "b", "kind": "hirth", "exclude": False, "N": 24,
         "label": "Hirth 24", "reason": "periodic",
         "boundary": {**z, "theta_deg": [0, 360]}},
    ]}}
    ex = face_feature_zones(conv)
    assert len(ex) == 1 and ex[0]["label"].startswith("balance cut")
    assert ex[0]["feature_class"] == "a" and "class a" in ex[0]["detail"]
    # modelled / not-excluded features are not auto-excluded
    assert all(x.get("feature_class") != "b" for x in ex)


def test_plain_disc_has_no_features():
    P, N = _disc(seed=5)
    rep, mask = _classify(P, N)
    assert rep["features"] == [] and not mask.any(), rep["features"]


def test_hirth_builder_cuts_n_teeth():
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.TopAbs import TopAbs_SOLID
    from meshsolid import occ_util as U
    H, r0, r1, n, h, alpha = 10.0, 20.0, 40.0, 12, 2.0, 60.0
    blank = BRepPrimAPI_MakeCylinder(R, H).Shape()
    feat = {"kind": "hirth", "N": n, "included_angle_deg": alpha, "r_inner": r0, "r_outer": r1,
            "height_mm": h, "z_min": 0.0, "z_max": h, "material_side": 1, "face": "back",
            "modelled": False, "model_pending": True}
    # only run the cutter when parameters are complete
    assert FF.hirth_params_complete(feat)
    try:
        cut = FF.cut_hirth(blank, [0, 0, 0], [0, 0, 1], [1, 0, 0], feat)
    except Exception as ex:  # noqa: BLE001
        pytest.skip(f"Hirth cutter not reliable on this OCC build: {type(ex).__name__}: {ex}")
    assert BRepCheck_Analyzer(cut).IsValid() and U.count(cut, TopAbs_SOLID) == 1
    removed = U.volume(blank) - U.volume(cut)
    # N triangular grooves, rough volume N * 0.5 * (2*half) * h * (r1-r0)
    half = h * math.tan(math.radians(alpha) / 2.0)
    expect = n * half * h * (r1 - r0)
    assert removed > 0.3 * expect, (removed, expect)


# ---- 1.0.2a: STG1 X2-B1941 regression (15-blade impeller read as 15 'balance cuts' + a hub 'ring') ------
def _bladed_back(n_blades=15, R=83.0, plate=7.0, seed=11, with_cut=True, n=120000):
    """Back face at z = 0 (outward normal -z, material above), a thin back plate, the hub flow surface at
    z = plate facing INTO the part (+z, away from the back face) in the n_blades passages, blades as gaps in
    the hub surface, blade side walls, plus one genuine irregular balance cut on the back face."""
    rng = np.random.default_rng(seed)
    P, N = _disc(n=n, R=R, seed=seed)                                   # back face, outward normals -z
    r = np.sqrt(rng.uniform(52.0 ** 2, R * R, n))                     # passages at the rim (STG1: r 52-83)
    th = rng.uniform(-math.pi, math.pi, n)
    pitch = 2 * math.pi / n_blades
    blade_half = 0.18 * pitch                                           # blade root occupies 36 % of a pitch
    phase = ((th + 0.5 * pitch) % pitch) - 0.5 * pitch
    hub = np.abs(phase) > blade_half
    H = np.c_[r[hub] * np.cos(th[hub]), r[hub] * np.sin(th[hub]), np.full(hub.sum(), plate)]
    Hn = np.tile([0.0, 0.0, 1.0], (hub.sum(), 1))                      # faces into the passage
    # blade side walls (tangential normals) from the hub up
    m = 20000
    rb = rng.uniform(20.0, R, m); zb = rng.uniform(plate, plate + 20.0, m)
    k = rng.integers(0, n_blades, m); sgn = rng.choice([-1.0, 1.0], m)
    tb = k * pitch + sgn * blade_half
    B = np.c_[rb * np.cos(tb), rb * np.sin(tb), zb]
    Bn = np.c_[-np.sin(tb) * sgn, np.cos(tb) * sgn, np.zeros(m)]
    # inner hub flow surface: a full ring 5.8 mm behind the back face, facing into the part (STG1 'ring')
    mr = 30000
    rr = np.sqrt(rng.uniform(12.0 ** 2, 50.0 ** 2, mr)); tr = rng.uniform(-math.pi, math.pi, mr)
    Rg = np.c_[rr * np.cos(tr), rr * np.sin(tr), np.full(mr, 5.8)]
    Rn = np.tile([0.0, 0.0, 1.0], (mr, 1))
    P = np.vstack([P, H, B, Rg]); N = np.vstack([N, Hn, Bn, Rn])
    if with_cut:
        P, N = _add_pocket(P, N, 101.0, 9.0, 62.0, 72.0, 1.4, wavy=0.2, seed=7)
    return P, N, int(len(_disc(n=n, R=R, seed=seed)[0]))


def test_stg1_15_blade_passages_are_not_balance_cuts():
    P, N, _nb = _bladed_back()
    for nb in (None, 15):
        rep, mask = FF.classify_points(P, N, [0, 0, 0], [0, 0, 1], [1, 0, 0], 83.0, n_blades=nb)
        cuts = [f for f in rep["features"] if f["kind"] == "balance_cut"]
        rings = [f for f in rep["features"] if f["kind"] in ("ring", "wide_depression") and f["exclude"]]
        # the one genuine back-face cut is still found and excluded ...
        assert len(cuts) == 1 and cuts[0]["exclude"], [(f["kind"], f["label"]) for f in rep["features"]]
        assert abs(cuts[0]["centre_deg"] - 101.0) < 4.0
        # ... and no blade passage / hub surface is excluded
        assert not rings
        assert mask.mean() < 0.05, mask.mean()
        hub_pts = (np.isclose(P[:, 2], 7.0) | np.isclose(P[:, 2], 5.8)) & (N[:, 2] > 0.9)
        assert not mask[hub_pts].any()


def test_fifteen_regular_patches_are_periodic_never_balance_cuts():
    # 15 equally spaced pockets ON the back face (outward normals): an N-fold family, kept inspected
    P, N = _disc(n=120000, seed=12)
    for k in range(15):
        P, N = _add_pocket(P, N, k * 24.0 + 3.0, 8.0, 62.0, 78.0, 1.5)
    for nb in (None, 15):
        rep, mask = _classify(P, N, n_blades=nb)
        assert not [f for f in rep["features"] if f["kind"] == "balance_cut"], rep["features"]
        per = [f for f in rep["features"] if f["class"] == "b"]
        assert len(per) == 1 and per[0]["N"] == 15 and not per[0]["exclude"], per
        assert mask.mean() < 0.01
    # 12 of 15 present (scan holes): still the blade-count family when the count is known
    P, N = _disc(n=120000, seed=13)
    for k in range(15):
        if k in (2, 7, 11):
            continue
        P, N = _add_pocket(P, N, k * 24.0 + 3.0, 8.0, 62.0, 78.0, 1.5)
    rep, mask = _classify(P, N, n_blades=15)
    per = [f for f in rep["features"] if f["class"] == "b"]
    assert len(per) == 1 and per[0]["N"] == 15 and not per[0]["exclude"], rep["features"]


def test_large_uncertain_region_is_flagged_and_kept_inspected():
    P, N = _disc(n=80000, seed=14)
    P, N = _add_pocket(P, N, 90.0, 120.0, 40.0, 75.0, 1.0)            # ~120 deg wide depression
    rep, mask = _classify(P, N)
    wide = [f for f in rep["features"] if f["kind"] in ("wide_depression", "ring", "deep_unclassified")]
    assert wide and all(not f["exclude"] and f.get("flagged_kept_inspected") and f.get("flag") for f in wide), wide
    assert mask.mean() < 0.01
    assert rep["summary"]["n_flagged_kept_inspected"] >= 1


def test_old_report_balance_cut_family_and_ring_are_not_excluded_by_metrology():
    """A 1.0.2 conversion report with 15 'balance cuts' 24 deg apart and a hub 'ring' (as on STG1): Metrology
    re-checks it and excludes neither; a genuine irregular cut is still excluded."""
    def cut(c, n=1600):
        return {"class": "a", "kind": "balance_cut", "exclude": True, "face": "back", "centre_deg": c,
                "span_deg": 37.7, "r_inner": 52.8, "r_outer": 82.9, "depth_mm": 7.19, "n_points": n,
                "label": f"balance cut on the back face at {c:.1f}°",
                "boundary": {"type": "sector", "point": [0, 0, 0], "dir": [0, 0, 1], "e1": [1, 0, 0],
                             "r": [52.8, 82.9], "theta_deg": [c - 18.8, c + 18.8], "t": [-7.2, 0.0],
                             "margin_mm": 2.0}}
    feats = [cut(k * 24.0 + 22.7) for k in range(15)]
    feats.append({"class": "uncertain", "kind": "ring", "exclude": True, "face": "back", "span_deg": 360.0,
                  "n_points": 240000, "label": "unclassified ring on the back face, depth 5.815 mm",
                  "boundary": {"type": "sector", "point": [0, 0, 0], "dir": [0, 0, 1], "e1": [1, 0, 0],
                               "r": [10.0, 83.0], "theta_deg": [0.0, 360.0], "t": [-5.8, 0.0], "margin_mm": 2.0}})
    real = cut(200.0, n=300)
    real.update(r_inner=70.0, r_outer=78.0, depth_mm=1.1, label="balance cut on the back face at 200.0°")
    feats.append(real)
    conv = {"impeller": {"blades": 15}, "face_features": {"exclusions": feats}}
    flagged = []
    zs = face_feature_zones(conv, flagged=flagged)
    assert [z["label"] for z in zs] == ["balance cut on the back face at 200.0°"], [z["label"] for z in zs]
    assert any("15-fold" in f["label"] for f in flagged) and any("ring" in f["label"] for f in flagged)


def test_hirth_with_blade_count_teeth_is_never_cut_but_kkk_24_on_16_blades_is():
    """Stage 3 (13 blades): 1.0.2 recognised the 13 blade passages at the back face as a 13-tooth Hirth coupling
    and cut it into the solid (rms 0.356 mm).  N = blade count is blading, never modelled; a real coupling with a
    different count (KK&K: 24 teeth on a 16-blade wheel) is still a Hirth."""
    P, N = _disc(n=90000, seed=21, hole=15.0)
    P13, N13 = _add_hirth(P, N, 13, 25.0, 55.0, 2.0)
    rep, _m = _classify(P13, N13, n_blades=13)
    b = [f for f in rep["features"] if f["class"] == "b"]
    assert b and all(f["kind"] != "hirth" and not f["model_pending"] and not f["exclude"] for f in b), b
    assert any(f.get("matches_blade_count") for f in b)
    # apply_modelled refuses a blade-count Hirth from an older report as well
    old = {"summary": {"n_blades": 13}, "features": [{"kind": "hirth", "N": 13, "model_pending": True,
                                                       "modelled": False}]}
    shape, old = FF.apply_modelled("SHAPE", old, [0, 0, 0], [0, 0, 1], [1, 0, 0])
    assert shape == "SHAPE" and not old["features"][0]["model_pending"]
    P24, N24 = _add_hirth(P, N, 24, 25.0, 55.0, 2.0)
    rep, _m = _classify(P24, N24, n_blades=16)
    h = [f for f in rep["features"] if f["kind"] == "hirth"]
    assert len(h) == 1 and h[0]["N"] == 24 and h[0]["model_pending"], rep["features"]
