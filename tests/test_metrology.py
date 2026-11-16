"""CLI / library tests for meshsolid.metrology on synthetic solids with a known offset."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import trimesh

ROOT = Path(__file__).resolve().parents[1]


def _box_step(path, extents=(20.0, 20.0, 20.0)):
    from meshsolid import occ_util as U
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.gp import gp_Pnt
    lx, ly, lz = extents
    shape = BRepPrimAPI_MakeBox(gp_Pnt(-lx / 2, -ly / 2, -lz / 2), lx, ly, lz).Shape()
    U.write_step(shape, str(path))
    return path


def test_default_bands_pass_is_green_and_sides_follow_palette():
    from meshsolid.metrology import default_bands, GREEN
    bands = default_bands(-0.2, 0.2, 0.05, 7)
    assert bands[0]["lo"] == pytest.approx(-0.2) and bands[-1]["hi"] == pytest.approx(0.2)
    in_tol = [b for b in bands if b["in_tol"]]
    assert in_tol and all(b["lo"] >= -0.05 - 1e-9 and b["hi"] <= 0.05 + 1e-9 for b in in_tol)
    assert any(b["rgb"] == list(GREEN) for b in in_tol)
    # below-pass trends toward blue, above toward red
    below = [b for b in bands if b["hi"] <= 0.05 + 1e-9 and not b["in_tol"]]
    above = [b for b in bands if b["lo"] >= 0.05 - 1e-9 and not b["in_tol"]]
    assert below and above
    assert below[0]["rgb"][2] >= below[0]["rgb"][0]   # bluer than red
    assert above[-1]["rgb"][0] >= above[-1]["rgb"][2]  # redder than blue


def test_identical_scan_and_solid_are_on_size(tmp_path):
    from meshsolid.metrology import run
    step = _box_step(tmp_path / "box.step")
    from meshsolid.occ_util import read_step, shape_to_trimesh
    tm = shape_to_trimesh(read_step(str(step)), lin=0.4, ang=0.3)
    scan = tmp_path / "box.stl"
    tm.export(scan)
    out = tmp_path / "m0"
    rep = run(scan, step, out, pass_tol=0.05, n_bands=7, n_probes=5, do_zones=False,
              device="cpu", lin=0.4)
    st = rep["stats"]
    assert st["rms"] < 0.02 and st["p95"] < 0.05
    assert st["within_pass_pct"] >= 99.0
    assert (out / "report.json").is_file() and (out / "report.png").is_file()
    assert (out / "legend.png").is_file() and (out / "deviation.csv").is_file()
    assert (out / "heatmap.png").is_file()
    assert rep["files"]["bands"]


def test_known_outward_offset_gives_expected_stats(tmp_path):
    """Inflate every vertex along its normal by 0.10 mm: signed deviation should be about +0.10."""
    from meshsolid.metrology import run
    step = _box_step(tmp_path / "box.step")
    from meshsolid.occ_util import read_step, shape_to_trimesh
    tm = shape_to_trimesh(read_step(str(step)), lin=0.3, ang=0.25)
    offset = 0.10
    mesh = tm.copy()
    mesh.vertices = mesh.vertices + mesh.vertex_normals * offset
    scan = tmp_path / "offset.stl"
    mesh.export(scan)
    out = tmp_path / "m1"
    rep = run(scan, step, out, range_mm=(-0.2, 0.2), pass_tol=0.05, n_bands=7,
              n_probes=8, probes=[[0.0, 0.0, 10.1]], device="cpu", lin=0.3,
              do_zones=False, section=([0, 0, 0], [0, 0, 1]))
    st = rep["stats"]
    # faces are exact; edges/corners are a bit short of a pure normal offset
    assert st["mean"] == pytest.approx(offset, abs=0.03)
    assert st["rms"] == pytest.approx(offset, abs=0.03)
    assert st["p95"] == pytest.approx(offset, abs=0.04)
    assert st["max_plus"] > 0.07
    assert st["within_pass_pct"] < 20.0          # 0.10 is outside ±0.05
    # most vertices fall in an over-tolerance (yellow/red) band
    over = [b for b in rep["bands"] if b["lo"] >= 0.05 - 1e-9]
    assert over
    typed = [p for p in rep["probes"] if p["kind"] == "typed"]
    assert typed and typed[0]["deviation_mm"] == pytest.approx(offset, abs=0.05)
    worst = [p for p in rep["probes"] if p["kind"] == "worst"]
    assert worst and all("xyz_mm" in p and "deviation_mm" in p for p in worst)
    assert rep["section"] and rep["section"]["n"] > 0
    assert "Interactive point picking" in " ".join(rep["notes"])
    data = json.loads((out / "report.json").read_text())
    assert data["stats"]["rms"] == st["rms"]


def test_cli_help_and_job_folder(tmp_path):
    import subprocess, sys
    help_txt = subprocess.run([sys.executable, "-m", "meshsolid.metrology", "-h"],
                              cwd=ROOT, capture_output=True, text=True, timeout=60).stdout
    for flag in ("--range", "--pass", "--bands", "--align", "--device", "--exclude",
                 "--max-dist", "--probes", "--probe", "--section", "--band-json"):
        assert flag in help_txt, flag
    from meshsolid.metrology import find_job_files, run
    step = _box_step(tmp_path / "job" / "mesh_body.step") if False else None
    job = tmp_path / "job"; job.mkdir()
    step = _box_step(job / "mesh_body.step")
    from meshsolid.occ_util import read_step, shape_to_trimesh
    tm = shape_to_trimesh(read_step(str(step)), lin=0.5, ang=0.4)
    tm.export(job / "mesh_body.stl")
    (job / "mesh_body.report.json").write_text(json.dumps({"status": "ok", "step": str(step)}))
    found = find_job_files(job)
    assert Path(found["scan"]).name == "mesh_body.stl" and Path(found["solid"]).name == "mesh_body.step"
    rc = subprocess.run([sys.executable, "-m", "meshsolid.metrology", str(job), "--out", str(tmp_path / "out"),
                         "--device", "cpu", "--lin", "0.5", "--no-zones"],
                        cwd=ROOT, capture_output=True, text=True, timeout=180)
    assert rc.returncode == 0, rc.stdout + rc.stderr
    assert "status ok" in rc.stdout
    assert (tmp_path / "out" / "report.json").is_file()


# ---- exclusions, hand-entered features, PDF -----------------------------------------------------------
def _pdf_pages(path):
    import re
    return len(re.findall(rb"/Type\s*/Page[^s]", Path(path).read_bytes()))


def test_parse_zones_box_and_cylinder():
    from meshsolid.metrology_exclude import parse_zones, zone_mask
    zs = parse_zones("box(0,0,0, 1,2,3); cyl(0,0,0, 0,0,2, 1.5, -1, 1)")
    assert [z["type"] for z in zs] == ["box", "cyl"]
    pts = np.array([[0.5, 1, 1], [5, 5, 5], [-1.0, 0, 0.5], [-1.0, 0, 3.0]])
    assert zone_mask(pts, zs[0]).tolist() == [True, False, False, False]
    assert zone_mask(pts, zs[1]).tolist() == [True, False, True, False]   # t range -1..1 along unit axis
    for bad in ("box(1,2,3)", "cyl(0,0,0, 0,0,0, 1)", "sphere(0,0,0,1)", "cyl(0,0,0, 0,0,1, -2)"):
        with pytest.raises(ValueError):
            parse_zones(bad)
    # the add-in's dialog validator agrees with the engine
    import sys
    sys.path.insert(0, str(ROOT / "fusion_addin" / "MeshSolid"))
    from mslib.metrology_ui_defaults import parse_zones as ui_parse
    assert len(ui_parse("box(0,0,0, 1,2,3); cyl(0,0,0, 0,0,2, 1.5, -1, 1)")) == 2
    for bad in ("box(1,2,3)", "cyl(0,0,0, 0,0,0, 1)", "sphere(0,0,0,1)"):
        with pytest.raises(ValueError):
            ui_parse(bad)


def test_typed_zone_and_face_exclusion_grey_band_and_pdf(tmp_path):
    from meshsolid.metrology import run
    step = _box_step(tmp_path / "box.step")
    scan = trimesh.creation.box(extents=(20.0, 20.0, 20.0)).subdivide().subdivide()
    scan.vertices[scan.vertices[:, 2] > 9.99, 2] += 0.5        # top face 0.5 mm high
    scan.export(tmp_path / "scan.stl")
    rep = run(tmp_path / "scan.stl", step, tmp_path / "o1", range_mm=(-0.2, 0.2), device="cpu", n_probes=3)
    assert rep["stats"]["max_plus"] > 0.45 and rep["stats"]["result"] == "FAIL"
    # typed zone over the top face
    rep2 = run(tmp_path / "scan.stl", step, tmp_path / "o2", range_mm=(-0.2, 0.2), device="cpu", n_probes=3,
               zones_text="box(-11,-11,9.5, 11,11,11)", meta={"part_name": "box", "operator": "QA"})
    st2 = rep2["stats"]
    # all inspected points are in band, but the zone removed ~1/6 of the scan: the honest verdict refuses PASS
    assert st2["max_abs"] < 0.05 and st2["pct_excluded"] > 5.0
    assert st2["result"] == "FAIL" and st2["verdict"]["target"]["failed"] == ["pct_excluded"]
    assert st2["result_engine_limit"] == "REFUSED" and "excluded" in st2["result_engine_limit_label"]
    ni = rep2["not_inspected"]
    assert ni and ni[0]["reason"] == "typed zone" and ni[0]["n"] > 0
    assert rep2["stats"]["n_excluded"] == ni[0]["n"]
    ex = [b for b in rep2["files"]["bands"] if b.get("excluded")]
    assert ex and Path(ex[0]["path"]).is_file() and ex[0]["rgb"] == [150, 150, 154]
    pdf = Path(rep2["files"]["report_pdf"])
    assert pdf.is_file() and pdf.read_bytes()[:4] == b"%PDF" and _pdf_pages(pdf) >= 6
    assert sum(b["pct"] for b in rep2["bands"]) == pytest.approx(100.0, abs=0.01)
    # face selection: the solid's top face tessellation
    top = trimesh.Trimesh([[-10, -10, 10], [10, -10, 10], [10, 10, 10], [-10, 10, 10]], [[0, 1, 2], [0, 2, 3]])
    top.export(tmp_path / "faces.stl")
    rep3 = run(tmp_path / "scan.stl", step, tmp_path / "o3", range_mm=(-0.2, 0.2), device="cpu", n_probes=3,
               exclude_faces=[tmp_path / "faces.stl"])
    assert rep3["not_inspected"][0]["reason"] == "user face selection"
    assert rep3["stats"]["max_abs"] < 0.05


def test_hand_entered_bore_is_auto_excluded_and_listed(tmp_path):
    """A solid with a typed bore (no bore in the scan): the scan's closed ends over the bore measure
    about R to the bore wall. By default that region is not inspected and is listed with its reason."""
    from meshsolid import synthetic as S, occ_util as OU
    from meshsolid.metrology import run
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    shape, _ = S.box_with_hole(D=10.0)
    OU.write_step(shape, str(tmp_path / "part.step"))
    (tmp_path / "part.report.json").write_text(json.dumps({"impeller": {"bore": {"d": 10.0, "source": "measure bore_d"}}}))
    S.to_scan(BRepPrimAPI_MakeBox(40.0, 30.0, 20.0).Shape(), edge=1.0, noise=0.0).export(tmp_path / "scan.stl")
    off = run(tmp_path / "scan.stl", tmp_path / "part.step", tmp_path / "off", device="cpu", n_probes=3,
              auto_exclude=False, auto_exclude_2mm=False, pdf=False)
    assert off["stats"]["max_plus"] > 3.0 and off["hand_entered"]
    on = run(tmp_path / "scan.stl", tmp_path / "part.step", tmp_path / "on", device="cpu", n_probes=3)
    assert on["stats"]["max_abs"] < 0.05
    ni = [x for x in on["not_inspected"] if x["reason"] == "hand-entered feature"]
    assert ni and ni[0]["n"] > 0 and "bore" in ni[0]["detail"] and "hand-entered" in ni[0]["detail"]
    assert Path(on["files"]["report_pdf"]).is_file()


# ---- distinct band colours (red reserved for beyond-cap) -----------------------------------------------
def test_default_bands_are_all_distinct_and_never_red():
    from meshsolid.metrology import default_bands, color_dist, CAP_RGB, GRAY
    for n in range(3, 12):
        for rng in ((-0.2, 0.2), (-0.1, 0.3), (-0.3, 0.05)):
            r = [tuple(b["rgb"]) for b in default_bands(rng[0], rng[1], 0.05, n)]
            assert len(set(r)) == len(r), (n, rng, r)
            assert all(color_dist(a, b) >= 45 for a, b in zip(r, r[1:])), (n, rng, r)
            assert all(color_dist(a, CAP_RGB) >= 55 for a in r), (n, rng)
            assert all(color_dist(a, GRAY) >= 45 for a in r), (n, rng)
    b7 = [x["color"] for x in default_bands(-0.2, 0.2, 0.05, 7)]
    # the two orange (+0.05..0.10, +0.10..0.15) and two blue (-0.20..-0.15, -0.15..-0.10) bands now differ
    assert b7[4] != b7[5] and b7[0] != b7[1]


def test_user_band_colours_are_kept_and_clashes_reported(tmp_path):
    from meshsolid.metrology import load_bands, colour_clashes, CAP_RGB
    same = [{"lo": -0.1, "hi": -0.05, "color": "#2040c0"}, {"lo": -0.05, "hi": 0.05, "color": "#24aa41"},
            {"lo": 0.05, "hi": 0.1, "color": "#f08c1e"}, {"lo": 0.1, "hi": 0.15, "color": "#f08c1e"},
            {"lo": 0.15, "hi": 0.2, "color": "#dc1e1e"}]                    # duplicate orange + a red like the cap
    b = load_bands(same, -0.1, 0.2, 0.05, 5)
    assert [x["color"] for x in b] == [x["color"] for x in same]           # never recoloured silently
    assert not any(x.get("adjusted") for x in b)
    w = colour_clashes(b, CAP_RGB)
    assert any(c["band"] == 3 and c["with"] == "band 3" for c in w)        # the two oranges
    assert any(c["band"] == 4 and "beyond-cap" in c["with"] for c in w)    # red band vs the cap colour


def test_default_bands_never_need_adjusting_against_default_cap():
    from meshsolid.metrology import default_bands, colour_clashes, CAP_RGB, GRAY
    for n in range(3, 12):
        b = default_bands(-0.2, 0.2, 0.05, n)
        assert not colour_clashes(b, CAP_RGB, GRAY), n
        assert all(sum(x["rgb"]) > 90 for x in b), n                        # no near-black band


# ---- map views follow the rotation axis (Fusion designs are often Y-up) --------------------------------
def _silhouette_aspect(img):
    a = np.asarray(img)[:, :, :3].astype(int)
    fg = np.abs(a - np.array([18, 20, 24])).sum(2) > 30
    ys, xs = np.nonzero(fg)
    return (xs.max() - xs.min() + 1) / (ys.max() - ys.min() + 1)


@pytest.mark.parametrize("axis", [(0, 1, 0), (0, 0, 1), (1, 0, 0)])
def test_view_frames_follow_rotation_axis(axis):
    from meshsolid.metrology import view_frames, render_shaded
    disc = trimesh.creation.cylinder(radius=20.0, height=4.0, sections=64)
    boss = trimesh.creation.cylinder(radius=6.0, height=10.0, sections=48)
    boss.apply_translation([0, 0, 6.0])                     # a hub on one side (the 'front')
    part = trimesh.util.concatenate([disc, boss])
    R = trimesh.geometry.align_vectors([0, 0, 1], axis)
    part.apply_transform(R)
    fr = {n: (Rm, lab) for n, Rm, lab in view_frames(part)}
    name = ("+" if sum(axis) > 0 else "-") + "XYZ"[int(np.argmax(np.abs(axis)))]
    assert name in fr["end"][1] and "rotation axis" in fr["end"][1] and "across the axis" in fr["side"][1]
    assert np.allclose(fr["end"][0][2], axis, atol=1e-6)     # looking along the axis, from the hub side
    assert abs(fr["side"][0][2] @ np.asarray(axis, float)) < 1e-6 and np.allclose(fr["side"][0][1], axis, atol=1e-6)
    for Rm, _ in fr.values():                                # right-handed, not mirrored
        assert np.allclose(Rm @ Rm.T, np.eye(3), atol=1e-9) and np.linalg.det(Rm) == pytest.approx(1.0)
    rgb = np.full((len(part.vertices), 3), 200)
    end = render_shaded(part, rgb, size=300, R=fr["end"][0])
    side = render_shaded(part, rgb, size=300, R=fr["side"][0])
    assert _silhouette_aspect(end) == pytest.approx(1.0, abs=0.08)      # round disc seen end-on
    assert _silhouette_aspect(side) > 2.0                                # flat profile seen from the side


# ---- outlier cap: explicit, true values reported, own colour ------------------------------------------
def _box_with_raised_top(tmp_path, raise_mm=0.5):
    step = _box_step(tmp_path / "box.step")
    scan = trimesh.creation.box(extents=(20.0, 20.0, 20.0)).subdivide().subdivide()
    top = scan.vertices[:, 2] > 9.99
    scan.vertices[top, 2] += raise_mm
    scan.export(tmp_path / "scan.stl")
    return step, tmp_path / "scan.stl", int(top.sum())


def test_outlier_cap_reports_true_max_counts_and_own_colour(tmp_path):
    from meshsolid.metrology import run
    step, scan, n_top = _box_with_raised_top(tmp_path)
    rep = run(scan, step, tmp_path / "o", range_mm=(-0.2, 0.2), device="cpu", n_probes=5, max_dist=0.1,
              cap_rgb=(255, 0, 255), section=((0, 0, 0), (1, 0, 0)))
    st = rep["stats"]
    assert st["cap_mm"] == 0.1 and st["max_abs"] <= 0.1                  # stats are within the cap ...
    assert st["true_max_plus"] == pytest.approx(0.5, abs=0.02)          # ... but the true max is reported
    assert st["n_beyond_cap"] == st["n_beyond_plus"] >= n_top and st["n_beyond_minus"] == 0
    assert st["pct_beyond_cap"] == pytest.approx(100.0 * st["n_beyond_cap"] / (st["n"] + st["n_beyond_cap"]), abs=0.01)
    cap_rows = [x for x in rep["not_inspected"] if x["reason"] == "beyond outlier cap"]
    assert cap_rows and cap_rows[0]["n"] == st["n_beyond_cap"] and "+0.5" in cap_rows[0]["detail"]
    assert st["n_not_inspected"] == 0                                     # nothing else excluded
    p0 = rep["probes"][0]
    assert p0["beyond_cap"] and p0["deviation_mm"] == pytest.approx(0.5, abs=0.02) and "beyond cap" in p0["kind"]
    capb = [b for b in rep["files"]["bands"] if b.get("beyond_cap")]
    assert capb and capb[0]["rgb"] == [255, 0, 255] and capb[0]["cap_mm"] == 0.1 and Path(capb[0]["path"]).is_file()
    assert not any(b.get("excluded") for b in rep["files"]["bands"])       # cap points are not 'grey excluded'
    ply = trimesh.load(rep["files"]["deviation_ply"], process=False)
    top = ply.vertices[:, 2] > 10.4
    assert (np.asarray(ply.visual.vertex_colors)[top, :3] == [255, 0, 255]).all()
    import csv
    rows = list(csv.DictReader(open(rep["files"]["csv"])))
    assert sum(int(r["beyond_cap"]) for r in rows) == st["n_beyond_cap"]
    assert rep["settings"]["cap_color"] == "#ff00ff"
    # without a cap: every point counted, true max == max
    rep2 = run(scan, step, tmp_path / "o2", range_mm=(-0.2, 0.2), device="cpu", n_probes=3, pdf=False)
    assert rep2["stats"]["max_plus"] == rep2["stats"]["true_max_plus"] == pytest.approx(0.5, abs=0.02)
    assert "cap_mm" not in rep2["stats"] and not rep2["not_inspected"]


def test_cap_legend_and_section_draw_cap_colour():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from meshsolid.metrology import draw_legend, plot_section, default_bands
    fig, ax = plt.subplots()
    draw_legend(ax, default_bands(-0.2, 0.2, 0.05, 7), 0.05, n_excluded=12, cap=0.1, n_cap=4577)
    texts = [t.get_text() for t in ax.texts]
    assert any("beyond cap |d| > 0.1" in t and "4,577" in t for t in texts)
    assert any("not inspected" in t and "12" in t for t in texts)
    fig2, ax2 = plt.subplots()
    sec = {"u_mm": np.array([0.0, 1, 2, 3]), "v_mm": np.array([0.0, 1, 0, 1]),
           "deviation_mm": np.array([0.01, 0.3, np.nan, -0.02])}
    plot_section(fig2, ax2, sec, -0.2, 0.2, cap=0.1, cap_rgb=(220, 30, 30))
    labels = [t.get_text() for t in ax2.get_legend().get_texts()]
    assert "not inspected" in labels and any("beyond cap" in x for x in labels)
    plt.close("all")


# ---- exclusions breakdown: per reason and per face / zone ----------------------------------------------
def test_exclusion_breakdown_per_face_zone_and_cap(tmp_path):
    from meshsolid.metrology import run
    step, scan, _n = _box_with_raised_top(tmp_path)
    side = trimesh.Trimesh([[10, -10, -10], [10, 10, -10], [10, 10, 10], [10, -10, 10]], [[0, 1, 2], [0, 2, 3]])
    side.export(tmp_path / "exclude_face_01_plane.stl")
    rep = run(scan, step, tmp_path / "o", range_mm=(-0.2, 0.2), device="cpu", n_probes=3, max_dist=0.1,
              exclude_faces=[tmp_path / "exclude_face_01_plane.stl"], zones_text="box(-11,-11,-11, -9,11,11)")
    st, nin = rep["stats"], rep["not_inspected"]
    labels = [e["label"] for e in nin]
    assert labels[0] == "face 1 (plane)" and labels[1] == "typed zone 1 (box)" and labels[2].startswith("outlier cap")
    assert all(e["n"] > 0 for e in nin)
    assert sum(e["n_new"] for e in nin) == st["n_excluded"]                  # 'counted' adds up to the total
    assert nin[2]["n_new"] == nin[2]["n"]           # the cap only counts points not already excluded by faces / zones
    assert set(st["excluded_by_reason"]) == {"user face selection", "typed zone", "beyond outlier cap"}
    assert sum(st["excluded_by_reason"].values()) == st["n_excluded"]
    assert [b["label"] for b in st["excluded_breakdown"]] == labels
    import re
    pdf_txt = Path(rep["files"]["report_pdf"]).read_bytes()
    assert pdf_txt[:4] == b"%PDF"


def test_zone_tags_use_their_own_part_rules_and_are_all_listed(tmp_path, monkeypatch):
    """The user's case: ticking 'bore', 'backface' AND 'tip' made the whole run use gear rules, so bore and
    back face silently matched nothing and only the outer-radius 'tip' band was excluded."""
    import meshsolid.metrology as M
    calls = []
    real = M.classify_zones

    def spy(scan, solid_tm, kind="auto"):
        calls.append(kind)
        kind = "impeller" if kind == "auto" else kind
        z = np.full(len(scan.vertices), "hub" if kind == "impeller" else "flank", dtype=object)
        z[: 10] = "bore" if kind == "impeller" else "tip"
        if kind == "impeller":
            z[10:30] = "backface"
        return z, kind, "spy"
    monkeypatch.setattr(M, "classify_zones", spy)
    step, scan, _n = _box_with_raised_top(tmp_path, raise_mm=0.0)
    rep = M.run(scan, step, tmp_path / "o", device="cpu", n_probes=1, pdf=False, do_map=False,
                exclude=["bore", "backface", "tip"])
    ni = {e["label"]: e for e in rep["not_inspected"]}
    TIP = "outer rim + blade tips, r >= 92 % of outer radius (gear 'tip' rule)"       # on an impeller the label says what the gear rule removes
    assert set(ni) == {"'bore' zone (impeller)", "'backface' zone (impeller)", TIP}
    assert ni["'bore' zone (impeller)"]["n"] == 10 and ni["'backface' zone (impeller)"]["n"] == 20
    assert "not the nose cone" in ni[TIP]["detail"]
    assert "a impeller" not in ni[TIP]["detail"] and "recognised as an impeller" in ni[TIP]["detail"]
    assert "gear" in calls and "impeller" not in calls                    # auto result reused for impeller tags
    assert real is not spy


def test_part_type_from_report():
    from meshsolid.metrology import part_type_from_report as pt
    assert pt({})[0] is None
    assert pt({"stages": {"build": {"method": "impeller"}}, "impeller": {"blades": 15}})[0] == "impeller"
    # the pinion: gear detected, built as a revolve, impeller recogniser said no -> gear rules
    assert pt({"stages": {"build": {"method": "revolve_profile"}}, "gear": {"detected": True, "teeth": 33},
               "recognisers": {"name": "impeller", "status": "none"}})[0] == "gear"
    assert pt({"stages": {"build": {"method": "extrude_profile"}}, "gear": {"detected": False}})[0] == "none"
