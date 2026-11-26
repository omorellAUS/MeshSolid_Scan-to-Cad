"""Filesystem-safe part ids for the Fusion add-in (no meshsolid import).

Kept in step with ``meshsolid.names.safe_part_id``. ``KK&K Stage 1`` becomes
``KKandK_Stage_1`` so an ampersand never reaches a file name or a shell.
"""
from __future__ import annotations

import re

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")
_DUP = re.compile(r"_+")


def safe_part_id(display_name: str) -> str:
    text = str(display_name or "").replace("&", "and")
    text = _UNSAFE.sub("_", text)
    text = _DUP.sub("_", text).strip("._")
    return text or "part"
