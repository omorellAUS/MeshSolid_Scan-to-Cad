"""Thread recognition: helical phase scoring, standard lookup, and the threaded stud end to end."""
import json

import numpy as np
import pytest

from meshsolid import synthetic as S
from meshsolid import thread as T


def _stud_band():
    m, truth = S.threaded_stud()
    c = np.asarray(m.triangles_center)
    sel = (c[:, 2] > 16) & (c[:, 2] < 35) & (np.abs(np.asarray(m.face_normals)[:, 2]) < 0.95)
    r, z, th = T.cylindrical(c[sel], np.zeros(3), np.array([0.0, 0.0, 1.0]))
    return r, z, th


def test_phase_score_finds_pitch_and_hand():
    r, z, th = _stud_band()
    best = max(((T.phase_score(r, z, th, p, h)[0], p, h) for p in T.CANDIDATE_PITCHES for h in (1, -1)
                if p * 3 < z.max() - z.min()), key=lambda t: t[0])
    assert best[1] == pytest.approx(1.75)
    assert best[2] == 1
    assert best[0] > 0.9


def test_standard_designation():
    s = T.standard_designation(11.98, 1.75, False)
    assert s["designation"] == "M12x1.75" and s["match"] == "good"


@pytest.mark.slow
def test_threaded_stud_end_to_end(tmp_path):
    from meshsolid.pipeline import Params, run
    m, truth = S.threaded_stud()
    stl = tmp_path / "stud.stl"
    m.export(stl)
    rep, *_ = run(stl, tmp_path / "stud.step", tmp_path / "stud.json", Params())
    assert rep["status"] == "ok"
    assert rep["stages"]["build"]["brepcheck_valid"]
    (t,) = rep["threads"]
    assert t["kind"] == "external" and t["hand"] == "right"
    assert t["pitch"] == pytest.approx(1.75, rel=0.01)
    assert t["major_d"] == pytest.approx(12.0, abs=0.1)
    assert t["standard"]["designation"] == "M12x1.75"
    assert (tmp_path / "stud.step").stat().st_size > 0
    json.loads((tmp_path / "stud.json").read_text())
