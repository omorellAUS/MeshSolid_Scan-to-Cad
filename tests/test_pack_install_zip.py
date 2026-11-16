"""Packer for the versioned Windows install zip: required install files present, private meshes absent."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def man_version():
    return json.loads((ROOT / "fusion_addin/MeshSolid/MeshSolid.manifest").read_text())["version"]


def test_pack_install_zip_contains_addin_engine_and_excludes_private(tmp_path):
    from scripts.pack_install_zip import pack

    zpath = pack(ROOT, tmp_path)
    assert zpath.is_file() and zpath.suffix == ".zip"
    assert "MeshSolid-" in zpath.name and "windows" in zpath.name

    with zipfile.ZipFile(zpath) as z:
        names = set(z.namelist())
    # required
    assert any(n.endswith("fusion_addin/MeshSolid/MeshSolid.py") for n in names)
    assert any(n.endswith("fusion_addin/install_windows.ps1") for n in names)
    assert any(n.endswith("fusion_addin/README.md") for n in names)
    assert any(n.endswith("meshsolid/__init__.py") for n in names)
    assert any(n.endswith("requirements.txt") for n in names)
    assert any(n.endswith("requirements-gpu.txt") for n in names)
    assert any(n.endswith("INSTALL.txt") for n in names)
    assert any(n.endswith("CHANGELOG.md") for n in names)
    cl = zipfile.ZipFile(zpath).read(next(n for n in names if n.endswith("CHANGELOG.md"))).decode()
    assert "1.0.2" in cl and "What's new" in cl and "Known open issues" in cl
    assert any(n.endswith("scripts/bench_device.py") for n in names)
    bat = next(n for n in names if n.endswith("/INSTALL_MeshSolid.bat"))
    assert bat.count("/") == 1                                   # top of the versioned folder, next to INSTALL.txt
    btxt = zipfile.ZipFile(zpath).read(bat).decode()
    assert "\r\n" in btxt and f"MeshSolid-{man_version()}" in btxt and "0.1.1" not in btxt
    assert "-Gpu" in btxt and "winget install" in btxt and "--scope user" in btxt
    for icon in ("MeshSolid/32x32.png", "MeshSolid/16x16@2x.png", "MeshSolid/64x64.png",
                 "Metrology/32x32.png", "Metrology/16x16@2x.png", "Metrology/64x64.png"):
        assert any(n.endswith("fusion_addin/MeshSolid/resources/" + icon) for n in names), icon
    allowed = ("/resources/MeshSolid/", "/resources/Metrology/")
    assert not [n for n in names if "/resources/" in n and not any(a in n for a in allowed)]
    assert any(n.endswith("fusion_addin/MeshSolid/mslib/ui_options.py") for n in names)
    # version matches manifest
    man = json.loads((ROOT / "fusion_addin/MeshSolid/MeshSolid.manifest").read_text())
    assert man["version"] in zpath.name
    # private / junk absent
    joined = "\n".join(names)
    assert "user_scans" not in joined
    assert ".venv/" not in joined and not any("/.venv/" in n for n in names)
    assert "__pycache__" not in joined
    assert not any(n.endswith("settings.json") for n in names)
    # INSTALL must not claim parameters drive the B-rep, and must not claim live Fusion tested
    install = next(n for n in names if n.endswith("INSTALL.txt"))
    text = zipfile.ZipFile(zpath).read(install).decode()
    assert "do not drive" in text.lower() or "does not drive" in text.lower()
    assert "not yet verified" in text.lower() or "not run inside fusion" in text.lower()
    assert "cpu" in text.lower()
