"""M10: bearing boundary dimensions and labyrinth seal teeth on synthetic parts with known values."""
import pytest
from pathlib import Path

from meshsolid import synthetic_bearings as B
from meshsolid.pipeline import Params, run

pytest.importorskip("pymeshlab")


def _run(name, tmp_path):
    m, tr = B.part(name)
    p = tmp_path / f"{name}.stl"; m.export(p)
    rep, *_ = run(str(p), str(tmp_path / "o.step"), str(tmp_path / "r.json"), Params())
    return rep, tr


@pytest.mark.parametrize("name", ["bearing_6205_2z", "bearing_6308_2z"])
def test_bearing(name, tmp_path):
    rep, tr = _run(name, tmp_path)
    assert rep["status"] == "ok"
    b = rep["bearing"]
    assert tr["designation"] in b["match"]["designations"]
    assert abs(b["bore_d"] - tr["d"]) < 0.05 and abs(b["outside_D"] - tr["D"]) < 0.05 and abs(b["width_B"] - tr["B"]) < 0.05
    assert rep["labyrinth"] == []


@pytest.mark.parametrize("name", ["seal_6t", "seal_9t"])
def test_labyrinth(name, tmp_path):
    rep, tr = _run(name, tmp_path)
    assert rep["status"] == "ok"
    assert "bearing" not in rep
    (s,) = rep["labyrinth"]
    assert s["side"] == tr["side"] and s["teeth"] == tr["teeth"]
    assert abs(s["pitch"] - tr["pitch"]) < 0.05
    assert abs(s["tip_diameter"] - tr["tip_diameter"]) < 0.05
    assert abs(s["groove_diameter"] - tr["groove_diameter"]) < 0.05
    assert abs(s["tip_land"] - tr["tip_land"]) < 0.15        # the scan rounds the tip corners


# The synthetic scan is not bit-reproducible: under parallel CPU load B.part("bearing_6205_open") returns one of
# two different remeshes (pymeshlab isotropic remeshing; identical OCC tessellation, different scan: 3,150,284 vs
# 3,134,084 byte STL). Variant A builds; variant B fails in the engine (local-trim unify fails, max edge tolerance
# 5.3 mm > 0.18 mm limit). That made this test flaky in the full suite. Both variants are frozen in tests/data so
# the outcome no longer depends on load: A must pass, B is a known engine bug (strict xfail flags the fix).
DATA = Path(__file__).parent / "data"


def _frozen(name):
    import gzip
    import io
    import trimesh
    return trimesh.load(io.BytesIO(gzip.decompress((DATA / name).read_bytes())), file_type="stl", process=False)


def _open_bearing_case(scan, tr, tmp_path):
    p = tmp_path / "scan.stl"; scan.export(p)
    rep, *_ = run(str(p), str(tmp_path / "o.step"), str(tmp_path / "r.json"), Params(bearing_build="envelope"))
    assert rep["status"] == "ok", rep.get("failure")
    el = rep["rolling_elements"]
    assert el["elements"] == tr["elements"] and el["count"] == tr["Z"]
    assert abs(el["diameter"] - tr["Dw"]) < 0.05 and abs(el["pitch_diameter"] - tr["pitch_diameter"]) < 0.05
    assert rep["bearing"]["match"]["designations"] == [tr["designation"]]
    env = rep["bearing"]["envelope"]
    assert env["valid"] and Path(env["step"]).exists() and (env["d"], env["D"], env["B"]) == (tr["d"], tr["D"], tr["B"])


@pytest.mark.parametrize("variant", [
    "A",
    pytest.param("B", marks=pytest.mark.xfail(strict=True, reason="engine bug: local-trim unify fails on this "
                                                                     "remesh variant (max edge tolerance 5.3 mm)")),
])
def test_open_bearing_6205_frozen_scans(variant, tmp_path):
    _, tr = B.PARTS["bearing_6205_open"]()
    _open_bearing_case(_frozen(f"bearing_6205_open_scan{variant}.stl.gz"), tr, tmp_path)


def test_open_bearing_nu205(tmp_path):
    m, tr = B.part("bearing_nu205_open")
    _open_bearing_case(m, tr, tmp_path)
