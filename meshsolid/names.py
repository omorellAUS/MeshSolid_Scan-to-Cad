"""Filesystem-safe part ids. Display names may contain '&'; file names never do."""
from __future__ import annotations

import re

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")
_DUP = re.compile(r"_+")


def safe_part_id(display_name: str) -> str:
    """Turn a display name into a file-name id.

    ``KK&K Stage 1`` becomes ``KKandK_Stage_1``. The display name itself is left
    unchanged for reports and the user interface; only paths use this id.
    """
    text = str(display_name or "").replace("&", "and")
    text = _UNSAFE.sub("_", text)
    text = _DUP.sub("_", text).strip("._")
    return text or "part"
