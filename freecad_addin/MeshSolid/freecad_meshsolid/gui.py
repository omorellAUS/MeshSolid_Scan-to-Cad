"""FreeCAD GUI for MeshSolid: run task panel (mesh object or file, measurements, notes, progress,
cancel) and review task panel (outcome, deviation vs limits, flags, notes, preview, parameter groups).

The engine runs as a separate process; a QTimer polls it, so FreeCAD stays responsive."""
import os
import time
import traceback

import FreeCAD
import FreeCADGui
from PySide import QtCore, QtGui, QtWidgets

from . import core
from .shared import engine_client as EC, review as RV

from .shared import WB_DIR
ICON = os.path.join(WB_DIR, "Resources", "icons", "MeshSolid.svg")
_state = {"checked": False}
_last_review = None
_last_run = None


def _msg(title, text, icon=QtWidgets.QMessageBox.Information):
    box = QtWidgets.QMessageBox(icon, "MeshSolid: " + title, text)
    box.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
    box.exec_()


def parse_measure(text):
    out = {}
    for part in (text or "").replace(",", ";").split(";"):
        if part.strip():
            if "=" not in part:
                raise ValueError(f"measurement '{part.strip()}' is not name=value")
            k, v = part.split("=", 1)
            out[k.strip()] = float(v)
    return out


def _selected_mesh():
    for o in FreeCADGui.Selection.getSelection():
        if o.isDerivedFrom("Mesh::Feature"):
            return o
    return None


class RunPanel:
    def __init__(self):
        self.run = self.job = self.timer = None
        self.mesh_obj = _selected_mesh()
        w = self.form = QtWidgets.QWidget(); w.setWindowTitle("Mesh to Solid")
        lay = QtWidgets.QFormLayout(w)
        self.src = QtWidgets.QLabel(f"Mesh object: {self.mesh_obj.Label}" if self.mesh_obj else
                                    "No Mesh object selected: give a mesh file")
        lay.addRow(self.src)
        row = QtWidgets.QHBoxLayout(); self.file = QtWidgets.QLineEdit(); b = QtWidgets.QPushButton("...")
        b.clicked.connect(self._browse); row.addWidget(self.file); row.addWidget(b)
        lay.addRow("Mesh file (STL/OBJ)", row)
        self.file.setEnabled(self.mesh_obj is None)
        self.measure = QtWidgets.QLineEdit(); self.measure.setPlaceholderText("bore_d=14.29; D3=40.02")
        lay.addRow("Measurements", self.measure)
        self.notes = QtWidgets.QLineEdit(); lay.addRow("Notes", self.notes)
        self.review = QtWidgets.QCheckBox("Review before import"); self.review.setChecked(True); lay.addRow(self.review)
        self.status = QtWidgets.QLabel(""); self.status.setWordWrap(True); lay.addRow(self.status)

    def _browse(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self.form, "Mesh file", "", "Meshes (*.stl *.obj *.ply)")
        if p:
            self.file.setText(p)

    def getStandardButtons(self):
        return int(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)

    def accept(self):
        if self.run is not None:
            self.status.setText(self.status.text() + "\n(running; Cancel stops the engine)")
            return False
        try:
            s = core.settings()
            if not _state["checked"]:
                ok, lines = EC.selfcheck(s, quick=True)
                if not ok:
                    _msg("engine not ready", "The meshsolid engine is not ready, so nothing was run:\n\n" +
                         "\n".join(lines) + "\n\nRerun install_windows.ps1 -FreeCAD, or fix settings.json in "
                         "the MeshSolid workbench folder.", QtWidgets.QMessageBox.Warning)
                    return False
                _state["checked"] = True
            try:
                meas = parse_measure(self.measure.text())
            except ValueError as e:
                _msg("measurements", f"{e}. Use name=value pairs separated by ';'."); return False
            out = core.jobs_dir()
            if self.mesh_obj is not None:
                path = os.path.join(out, "mesh_object.stl")
                if core.export_mesh(self.mesh_obj, path) == 0:
                    _msg("no mesh", "The selected Mesh object has no facets, so nothing was run."); return False
            else:
                path = self.file.text().strip().strip('"')
                if not os.path.exists(path):
                    _msg("no mesh", "Select a Mesh object or give an existing mesh file."); return False
            notes = [t.strip() for t in self.notes.text().split(";") if t.strip()]
            self.s = s
            self.job, self.run = core.start(path, out, meas, notes, s)
            for wdg in (self.file, self.measure, self.notes, self.review):
                wdg.setEnabled(False)
            self.timer = QtCore.QTimer(); self.timer.timeout.connect(self._tick); self.timer.start(300)
            self.status.setText("engine: starting")
        except Exception:  # noqa: BLE001
            _msg("add-in error", "MeshSolid failed (add-in error, not an engine result):\n" + traceback.format_exc(),
                 QtWidgets.QMessageBox.Critical)
        return False                    # the dialog stays open until the run ends

    def _tick(self):
        rc = self.run.poll()
        if rc is None:
            self.status.setText(f"engine ({self.run.elapsed():.0f} s): {self.run.last_stage()}")
            return
        self.timer.stop()
        self._done(rc)

    def _done(self, rc):
        want_review = self.review.isChecked()       # read before closeDialog deletes the form's widgets
        FreeCADGui.Control.closeDialog()
        try:
            res = core.finish(self.job, rc, self.s)
            if not res["outcome"]["importable"]:
                o = res["outcome"]
                _msg(o["title"], f"{o['message']}\n\n" + (f"Next step: {o['hint']}\n\n" if o.get("hint") else "") +
                     f"Job folder: {res['out_dir']}\nReport: {res['report_path']}\nEngine log: {res['log']}",
                     QtWidgets.QMessageBox.Warning)
                return
            if want_review:
                global _last_review
                _last_review = ReviewPanel(res)          # kept so tests (and the timer) can reach it
                FreeCADGui.Control.showDialog(_last_review)
            else:
                groups = {g["group"] for g in res["model"]["param_groups"] if g["default"]}
                _apply(res, groups)
        except Exception:  # noqa: BLE001
            _msg("add-in error", traceback.format_exc(), QtWidgets.QMessageBox.Critical)

    def reject(self):
        if self.run is not None and self.run.poll() is None:
            self.run.cancel()
            if self.timer:
                self.timer.stop()
            self._done(self.run.poll())
            return True
        FreeCADGui.Control.closeDialog()
        return True


def _apply(res, groups):
    doc = FreeCAD.ActiveDocument or FreeCAD.newDocument("MeshSolid")
    doc.openTransaction("MeshSolid import")
    try:
        feat, text = core.apply(doc, res, groups)
    finally:
        doc.commitTransaction()
    if feat is not None:
        FreeCADGui.ActiveDocument.ActiveView.fitAll()
    _msg("import", text, QtWidgets.QMessageBox.Information if feat is not None else QtWidgets.QMessageBox.Warning)


class ReviewPanel:
    def __init__(self, res):
        self.res = res
        m = res["model"]
        w = self.form = QtWidgets.QWidget(); w.setWindowTitle("MeshSolid review")
        lay = QtWidgets.QVBoxLayout(w)
        tb = QtWidgets.QTextBrowser(); tb.setHtml(RV.review_html(m)); tb.setMinimumHeight(260); lay.addWidget(tb)
        if res.get("preview"):
            pm = QtGui.QPixmap(res["preview"])
            if not pm.isNull():
                lab = QtWidgets.QLabel(); lab.setPixmap(pm.scaledToWidth(440, QtCore.Qt.SmoothTransformation))
                lay.addWidget(lab)
        self.boxes = {}
        if m["param_groups"]:
            lay.addWidget(QtWidgets.QLabel("Write to spreadsheet MeshSolidParams (values are recorded, not driving):"))
        for g in m["param_groups"]:
            cb = QtWidgets.QCheckBox(f"{g['label']} ({g['count']})"); cb.setChecked(bool(g["default"]))
            lay.addWidget(cb); self.boxes[g["group"]] = cb
        lay.addWidget(QtWidgets.QLabel(f"Job folder: {res['out_dir']}"))

    def getStandardButtons(self):
        return int(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)

    def accept(self):
        groups = {g for g, cb in self.boxes.items() if cb.isChecked()}   # before the form is deleted
        FreeCADGui.Control.closeDialog()
        _apply(self.res, groups)
        return True

    def reject(self):
        FreeCADGui.Control.closeDialog()
        FreeCAD.Console.PrintMessage(f"MeshSolid: not imported; job folder {self.res['out_dir']}\n")
        return True


class MeshToSolidCommand:
    def GetResources(self):
        return {"Pixmap": ICON, "MenuText": "Mesh to Solid",
                "ToolTip": "Convert the selected Mesh object (or a mesh file) to one Part solid with the meshsolid engine"}

    def IsActive(self):
        return not FreeCADGui.Control.activeDialog()

    def Activated(self):
        global _last_run
        _last_run = RunPanel()
        FreeCADGui.Control.showDialog(_last_run)


class BatchCommand:
    def GetResources(self):
        return {"Pixmap": ICON, "MenuText": "Batch folder to solids",
                "ToolTip": "Convert every mesh in a folder; meshes with a same-name STEP are scored against it"}

    def IsActive(self):
        return not FreeCADGui.Control.activeDialog()

    def Activated(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(None, "MeshSolid: folder of meshes")
        if not folder:
            return
        out_dir = os.path.join(core.jobs_dir(), "batch_" + time.strftime("%Y%m%d_%H%M%S"))
        dlg = QtWidgets.QProgressDialog("engine: starting", "Cancel", 0, 0)
        dlg.setWindowTitle("MeshSolid batch"); dlg.show()

        def tick(run):
            dlg.setLabelText(f"batch ({run.elapsed():.0f} s): {run.last_stage()}")
            QtWidgets.QApplication.processEvents()
            return dlg.wasCanceled()
        try:
            rc, text = core.run_batch(folder, out_dir, tick=tick, poll=0.2)
        finally:
            dlg.close()
        _msg("batch", text, QtWidgets.QMessageBox.Information if rc in (None, 0) else QtWidgets.QMessageBox.Warning)


def register():
    FreeCADGui.addCommand("MeshSolid_MeshToSolid", MeshToSolidCommand())
    FreeCADGui.addCommand("MeshSolid_Batch", BatchCommand())
