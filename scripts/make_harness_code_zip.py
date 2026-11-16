"""Build the harness code zip that is synced to the PC (%LOCALAPPDATA%\\meshsolid\\harness\\code).

    python scripts/make_harness_code_zip.py OUT.zip

Contains meshsolid/ and scripts/ as committed at HEAD (git archive, so no stray caches or part files) plus
meshsolid/_build_commit.txt holding the short commit (+'-dirty' if tracked files differ from HEAD). The PC copy
has no .git, so metrology_harness.git_commit() and the robustness harness read the commit from that file."""
from __future__ import annotations

import io
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build_commit() -> str:
    c = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                       check=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True, check=True).stdout.strip()
    return c + ("-dirty" if dirty else "")


def main(out: str) -> str:
    commit = build_commit()
    arch = subprocess.run(["git", "-C", str(ROOT), "archive", "--format=zip", "HEAD", "meshsolid", "scripts"],
                          capture_output=True, check=True).stdout
    with zipfile.ZipFile(io.BytesIO(arch)) as src, zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as dst:
        for info in src.infolist():
            if info.filename != "meshsolid/_build_commit.txt":
                dst.writestr(info, src.read(info))
        dst.writestr("meshsolid/_build_commit.txt", commit)
    print(f"{out}: commit {commit}")
    return commit


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "harness_code.zip")
