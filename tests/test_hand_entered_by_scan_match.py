"""A solid with no link to its conversion (Fusion body that lost the attribute, Stage 3 'Body4') still gets the
hand-entered bore excluded: the conversion report is found by a byte-identical scan in the jobs folder."""
import json
import shutil

import trimesh

from meshsolid import metrology as M
from meshsolid import metrology_exclude as X
from meshsolid import occ_util as U


def _solid_with_bore(path, R=20.0, H=10.0, rb=3.0):
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCP.gp import gp_Ax2, gp_Pnt, gp_Dir
    ax = gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1))
    outer = BRepPrimAPI_MakeCylinder(ax, R, H).Shape()
    ax2 = gp_Ax2(gp_Pnt(0, 0, -1), gp_Dir(0, 0, 1))
    hole = BRepPrimAPI_MakeCylinder(ax2, rb, H + 2).Shape()
    U.write_step(BRepAlgoAPI_Cut(outer, hole).Shape(), str(path))


def test_bore_excluded_via_scan_match(tmp_path):
    jobs = tmp_path / "jobs"
    conv_job, met_job = jobs / "20261115-064855", jobs / "20261115-072554"
    conv_job.mkdir(parents=True); met_job.mkdir()
    scan = trimesh.creation.cylinder(radius=20.0, height=10.0, sections=128)   # no bore in the scan
    scan.apply_translation((0, 0, 5.0))
    scan = trimesh.Trimesh(*trimesh.remesh.subdivide_to_size(scan.vertices, scan.faces, 1.0))
    scan.export(conv_job / "mesh_body.stl")
    shutil.copy(conv_job / "mesh_body.stl", met_job / "scan.stl")
    (conv_job / "mesh_body.report.json").write_text(json.dumps({
        "tool": "meshsolid", "input": str(conv_job / "mesh_body.stl"), "status": "ok",
        "impeller": {"blades": 13, "bore": {"d": 6.0, "source": "measure bore_d"}}}))
    _solid_with_bore(met_job / "solid.step")
    assert X.find_convert_report(met_job / "solid.step") is None
    found, job = X.find_convert_report_for_scan(met_job / "scan.stl")
    assert job == conv_job.name
    rep = M.run(str(met_job / "scan.stl"), str(met_job / "solid.step"), str(met_job), pdf=False, do_map=False,
                do_section=False, do_histogram=False)
    labels = [x["label"] for x in rep["not_inspected"]]
    assert any(lb.startswith("bore Ø6") for lb in labels), labels
    assert any("found by scan match" in n for n in rep["notes"])
    # a different scan does not match
    other = trimesh.creation.box((10, 10, 10)); other.export(met_job / "other.stl")
    assert X.find_convert_report_for_scan(met_job / "other.stl") == (None, None)
