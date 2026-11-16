"""Regenerate tests/fusion_api_members.json from Autodesk's Fusion API intellisense stubs.

The stubs are the `adsk` package on PyPI (core.py / fusion.py, "generated for code intellisense only").
They list every real class, method, property and enum value of the Fusion API. The strict adsk stub
in tests/test_addin.py only allows members listed in the JSON, so a call such as
CommandInputs.addColorCommandInput (which does not exist) fails the tests instead of failing in Fusion.

    pip download --no-deps adsk && tar xzf adsk-*.tar.gz
    python scripts/gen_fusion_api_members.py adsk-2.0.12157/adsk tests/fusion_api_members.json
"""
import ast
import json
import sys
from pathlib import Path

# Classes the add-in touches (directly or through stubs in the tests). Members include inherited ones.
CLASSES = {
    "core": ["Application", "UserInterface", "Command", "CommandInputs", "CommandInput", "CommandDefinitions",
             "CommandDefinition", "CommandCreatedEvent", "CommandEvent", "InputChangedEvent", "CustomEvent",
             "CommandCreatedEventArgs", "CommandEventArgs", "InputChangedEventArgs",
             "SelectionCommandInput", "Selection", "BoolValueCommandInput", "StringValueCommandInput",
             "FloatSpinnerCommandInput", "IntegerSpinnerCommandInput", "IntegerSliderCommandInput",
             "DropDownCommandInput", "ButtonRowCommandInput", "ListItems", "ListItem", "GroupCommandInput",
             "TextBoxCommandInput", "ImageCommandInput", "ProgressDialog", "FileDialog", "FolderDialog",
             "Color", "Appearance", "Appearances", "Properties", "ColorProperty", "MaterialLibraries",
             "MaterialLibrary", "ImportManager", "Matrix3D", "ValueInput", "Document", "Documents",
             "Products", "Workspaces", "Workspace", "ToolbarTabs", "ToolbarTab", "ToolbarPanels",
             "ToolbarPanel", "ToolbarPanelList", "ToolbarControls", "CommandControl", "ObjectCollection",
             "DropDownStyles", "DialogResults", "DocumentTypes", "Attributes", "Attribute",
             "MessageBoxButtonTypes", "MessageBoxIconTypes", "Palettes", "Palette", "HTMLEvent",
             "HTMLEventArgs", "PaletteDockingStates", "Surface", "SurfaceTypes"],
    "fusion": ["Design", "Component", "BRepBody", "BRepBodies", "MeshBody", "MeshBodies", "MeshBodyList",
               "PolygonMesh", "TriangleMesh", "MeshManager", "TriangleMeshCalculator", "ExportManager",
               "STEPExportOptions", "BaseFeatures", "BaseFeature", "Features", "Occurrences", "Occurrence",
               "UserParameters", "UserParameter", "TemporaryBRepManager", "MeshUnits", "DesignTypes",
               "TriangleMeshQualityOptions", "BRepFace", "BRepFaces"],
}


def _classes(path):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    out = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            bases = [ast.unparse(b).split(".")[-1] for b in node.bases]
            members = set()
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and not item.name.startswith("_"):
                    members.add(item.name)
                elif isinstance(item, ast.Assign):
                    for t in item.targets:
                        if isinstance(t, ast.Name) and not t.id.startswith("_"):
                            members.add(t.id)
            out[node.name] = (bases, members)
    return out


def main(stub_dir, out_path):
    allc = {}
    for mod in ("core", "fusion"):
        allc.update(_classes(Path(stub_dir) / f"{mod}.py"))

    def members(name, seen=()):
        bases, own = allc.get(name, ([], set()))
        res = set(own)
        for b in bases:
            if b in allc and b not in seen:
                res |= members(b, seen + (name,))
        return res

    data = {"_source": f"adsk intellisense stubs ({Path(stub_dir).resolve().parent.name})"}
    for mod, names in CLASSES.items():
        for n in names:
            if n not in allc:
                raise SystemExit(f"{n} is not a Fusion API class")
            data[f"{mod}.{n}"] = sorted(members(n))
    Path(out_path).write_text(json.dumps(data, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out_path}: {len(data) - 1} classes")


if __name__ == "__main__":
    main(*sys.argv[1:3])
