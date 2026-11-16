"""Scan noise estimate (plan step 4): patch-local fits on a scan, plus its flat end faces.

    python scripts/scan_noise.py SCAN.stl [SCAN2.stl ...] --out noise.json [--names a,b,c]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from meshsolid import scan_noise as SN  # noqa: E402
from meshsolid.metrology import load_scan, rotation_axis  # noqa: E402


def one(path):
    scan = load_scan(path)
    axis = rotation_axis(scan)
    out = {"scan": str(path), "vertices": int(len(scan.vertices)), "whole_scan": SN.estimate(scan)}
    for end in ("low", "high"):
        m = SN.end_face_mask(scan, axis, end)
        if m.sum() > 200:
            est = SN.estimate(scan, region_mask=m)
            out[f"end_face_{end}"] = {"vertices": int(m.sum()), "flatness_global_plane": SN.flatness(scan, m),
                                      "patches": {k: v for k, v in est.items() if k != "rows"}}
    out["whole_scan"].pop("rows")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("scans", nargs="+")
    ap.add_argument("--names", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    names = a.names.split(",") if a.names else [Path(s).parent.name for s in a.scans]
    res = {}
    for n, s in zip(names, a.scans):
        res[n] = one(s)
        print(n, json.dumps({k: v for k, v in res[n].items() if k != "scan"}), flush=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
