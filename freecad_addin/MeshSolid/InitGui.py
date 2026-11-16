import FreeCADGui


class MeshSolidWorkbench(FreeCADGui.Workbench):
    MenuText = "MeshSolid"
    ToolTip = "Mesh to solid with the meshsolid engine"

    def __init__(self):
        import os
        import freecad_meshsolid as _p
        self.__class__.Icon = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(_p.__file__))),
                                           "Resources", "icons", "MeshSolid.svg")

    def Initialize(self):
        try:
            from freecad_meshsolid import gui
            gui.register()
        except Exception as e:  # noqa: BLE001  (shown when the command is used, not hidden)
            import FreeCAD
            FreeCAD.Console.PrintError(f"MeshSolid: {e}\n")

            class _Broken:
                def GetResources(self):
                    return {"MenuText": "Mesh to Solid (not available)", "ToolTip": str(e)}

                def Activated(self):
                    from PySide import QtWidgets
                    QtWidgets.QMessageBox.warning(None, "MeshSolid", f"MeshSolid cannot start:\n{e}")
            FreeCADGui.addCommand("MeshSolid_MeshToSolid", _Broken())
        cmds = ["MeshSolid_MeshToSolid"] + (["MeshSolid_Batch"] if "MeshSolid_Batch" in FreeCADGui.listCommands() else [])
        self.appendToolbar("MeshSolid", cmds)
        self.appendMenu("MeshSolid", cmds)

    def GetClassName(self):
        return "Gui::PythonWorkbench"


FreeCADGui.addWorkbench(MeshSolidWorkbench())
