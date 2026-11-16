"""freecadcmd check: import an engine STEP as one Part solid and write the report's parameters into the
spreadsheet.  Usage (env): MS_FC_STEP, MS_FC_REPORT, MS_FC_RESULT; run with
freecadcmd -c "exec(open('tests/freecad/import_check.py').read())"."""
import json, os, sys, time
WB = os.path.join(os.environ.get("MS_FC_REPO", os.getcwd()), "freecad_addin", "MeshSolid")
sys.path.insert(0, WB)
import FreeCAD
from freecad_meshsolid import core
from mslib.outcome import explain
from mslib.review import review_model
out = {}
try:
    rep = json.load(open(os.environ["MS_FC_REPORT"]))
    rep["step"] = os.environ["MS_FC_STEP"]
    o = explain(0, rep, "/nonexistent")
    res = {"report": rep, "outcome": o, "step": o["step"], "report_path": os.environ["MS_FC_REPORT"],
           "model": review_model(rep, o)}
    doc = FreeCAD.newDocument("imp")
    t0 = time.time()
    groups = {g["group"] for g in res["model"]["param_groups"]}
    feat, msg = core.apply(doc, res, groups)
    out["seconds"] = round(time.time() - t0, 1)
    out["outcome"] = o["kind"]; out["msg"] = msg
    if feat is not None:
        s = feat.Shape
        out["solid"] = {"n_solids": len(s.Solids), "valid": s.isValid(), "faces": len(s.Faces),
                        "volume": round(s.Volume, 1), "bbox": [round(x, 2) for x in (s.BoundBox.XLength, s.BoundBox.YLength, s.BoundBox.ZLength)]}
    sh = doc.getObject("MeshSolidParams")
    if sh:
        out["aliases"] = {}
        r = 2
        while sh.getContents(f"A{r}"):
            name = sh.getContents(f"A{r}").lstrip("'")
            out["aliases"][name] = str(sh.get(name)); r += 1
    p = os.environ.get("MS_FC_SAVE")
    if p:
        doc.saveAs(p); out["saved"] = p
except Exception:  # noqa: BLE001
    import traceback; out["error"] = traceback.format_exc()
open(os.environ["MS_FC_RESULT"], "w").write(json.dumps(out, default=str))
