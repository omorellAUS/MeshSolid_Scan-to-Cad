"""Build a versioned lossless Windows install zip for MeshSolid (Fusion add-in + engine).

No private user meshes, no .venv, no settings.json. CPU is the default path; GPU extras
are optional via requirements-gpu.txt / install -Gpu.
"""
from __future__ import annotations

import json
import zipfile
from datetime import date
from pathlib import Path

INCLUDE_FILES = [
    "INSTALL_MeshSolid.bat",
    "requirements.txt",
    "requirements-gpu.txt",
    "package.xml",
    "CHANGELOG.md",
]
INCLUDE_TREES = [
    "fusion_addin",
    "freecad_addin",
    "meshsolid",
    "scripts/bench_device.py",
]
EXCLUDE_DIR_NAMES = {"__pycache__", ".venv", ".git", ".pytest_cache"}
EXCLUDE_FILE_NAMES = {"settings.json"}
EXCLUDE_SUFFIXES = {".pyc", ".pyo"}
EXCLUDE_NAME_SUBSTR = ("_1d96ac0", "_backup", ".orig")


def read_version(root: Path) -> str:
    man = json.loads((root / "fusion_addin" / "MeshSolid" / "MeshSolid.manifest").read_text())
    return str(man["version"])


def _skip(path: Path) -> bool:
    if path.name in EXCLUDE_FILE_NAMES or path.suffix in EXCLUDE_SUFFIXES:
        return True
    if any(s in path.name for s in EXCLUDE_NAME_SUBSTR):
        return True
    return any(p in EXCLUDE_DIR_NAMES for p in path.parts)


def _add_file(z: zipfile.ZipFile, root: Path, prefix: str, path: Path) -> None:
    if _skip(path.relative_to(root) if path.is_relative_to(root) else path):
        return
    if not path.is_file():
        return
    rel = path.relative_to(root).as_posix()
    z.write(path, f"{prefix}/{rel}")


def _add_tree(z: zipfile.ZipFile, root: Path, prefix: str, rel: str) -> None:
    src = root / rel
    if src.is_file():
        _add_file(z, root, prefix, src)
        return
    for p in src.rglob("*"):
        if p.is_dir():
            continue
        rp = p.relative_to(root)
        if _skip(rp):
            continue
        if "user_scans" in rp.parts:
            continue
        z.write(p, f"{prefix}/{rp.as_posix()}")


def install_txt(version: str) -> str:
    return f"""MeshSolid {version} — Windows install (Fusion add-in + engine)
Built: {date.today().isoformat()}

WHAT THIS IS
  Mesh body or STL/OBJ -> meshsolid engine (separate Python process) -> STEP imported
  as a plain B-rep body, with an optional review step and user parameters that RECORD
  recognised design values (gear, thread, impeller, snapped dims). Those parameters
  do NOT drive the imported B-rep geometry.

REQUIREMENTS
  - Windows 10/11, Autodesk Fusion
  - Python 3.12 or 3.13 from python.org (64-bit; the py launcher is enough)
  - Internet for the first pip install (~180 MB of wheels)
  - CPU is mandatory and always works. GPU (NVIDIA Warp) is optional.

INSTALL
  1. Unzip this archive to a stable folder, e.g. C:\\meshsolid\\MeshSolid-{version}
  2. PowerShell from that folder:
       powershell -ExecutionPolicy Bypass -File fusion_addin\\install_windows.ps1
     Options: -Python <python.exe>, -VenvDir, -AddinsDir, -CheckOnly, -QuickCheck,
     -NoAddinCopy, -FreeCAD, -NoFusion, -Gpu
     or simply double-click INSTALL_MeshSolid.bat (finds/installs Python 3.13, adds -Gpu).
  3. Fusion: Utilities > Add-Ins (Shift+S) > Add-Ins tab > MeshSolid > Run
     (optionally tick Run on Startup). A MESHSOLID {version} panel with a large
     "Mesh to Solid" button appears on the UTILITIES and MESH tabs (also still in the
     Utilities > ADD-INS dropdown).

CPU / GPU
  Default install is CPU-only. For optional GPU closest-point acceleration:
       powershell -ExecutionPolicy Bypass -File fusion_addin\\install_windows.ps1 -Gpu
  Engine --device auto uses CUDA when usable, otherwise CPU. report.json records
  the device under "device".

FAILURES
  Failures never import a STEP. The dialog names the case: engine not ready,
  cancelled, timed out (stage), crashed (stage + log tail), no report, solid
  outside deviation limits, or the engine failure text, plus the job folder under
  %LOCALAPPDATA%\\meshsolid\\jobs.

NOT YET VERIFIED ON A LIVE FUSION / WINDOWS SESSION FROM THIS PACKAGE BUILD
  This Linux packaging host did not run Fusion. On first Windows session, check:
  1. Add-in loads; MESHSOLID panel on UTILITIES and MESH tabs (else see
     %LOCALAPPDATA%\\meshsolid\\addin_ui.log).
  2. "engine not ready" if settings.json is wrong; clears when fixed.
  3. Mesh body export size in mm matches a known STL (job folder mesh_body.stl).
  4. Small prismatic STL: progress updates, cancel works, review dialog, Import -> 1 body.
  5. User parameters appear under Modify > Change Parameters with ms_ prefix
     (they record values; they do not drive the body).
  6. A long run keeps Fusion responsive enough through the progress loop.

See fusion_addin/README.md for full detail, and CHANGELOG.md for what is new in this
release and the known open issues. Private user scan meshes are not in this zip.
"""


def pack(root: Path, out_dir: Path) -> Path:
    root = root.resolve()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    version = read_version(root)
    prefix = f"MeshSolid-{version}"
    zpath = out_dir / f"MeshSolid-{version}-windows.zip"
    if zpath.exists():
        zpath.unlink()
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for rel in INCLUDE_FILES:
            _add_file(z, root, prefix, root / rel)
        for rel in INCLUDE_TREES:
            _add_tree(z, root, prefix, rel)
        z.writestr(f"{prefix}/INSTALL.txt", install_txt(version))
    return zpath


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out = pack(root, root / "dist")
    print(out)


if __name__ == "__main__":
    main()
