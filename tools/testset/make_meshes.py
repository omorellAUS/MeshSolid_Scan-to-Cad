"""Build engine inputs for every case of a user test set (reads out/user/<set>/inventory.json):
  <case>.fine.stl  CAD tessellation of the STEP truth (already written by inventory.py under truth/)
  <case>.scan.stl  scan-like: isotropic remesh (edge scaled to part size, 0.4 mm floor), Gaussian normal
                   noise sigma 0.025 mm (Artec-like), crisp edges not preserved (meshsolid.synthetic_screw.scanify)
  user STL         used as-is (primary input when paired with a STEP)
Writes out/user/<set>/cases.json (the run list). Usage: python tools/testset/make_meshes.py drone"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from meshsolid.synthetic_screw import scanify  # noqa: E402

SET = sys.argv[1] if len(sys.argv) > 1 else "drone"
OUT = C.out_dir(SET)
MESH = OUT / "meshes"
MESH.mkdir(parents=True, exist_ok=True)
MAX_SCAN_TRIS = 150_000   # 700k-tri scans of the 0.3-0.5 m skins timed out in segmentation (600 s); see README
NOISE = 0.025


def scan_edge(area):
    # isotropic triangle area ~ 0.433 e^2; keep <= MAX_SCAN_TRIS triangles, never finer than 0.4 mm
    return float(max(0.4, np.sqrt(area / (0.433 * MAX_SCAN_TRIS))))


def main():
    inv = json.loads((OUT / "inventory.json").read_text())
    recs = [r for r in inv["records"] if "error" not in r]
    cases = []
    names = {}

    def cid(group, slug, kind):
        base = f"{group}__{slug}__{kind}"
        names[base] = names.get(base, 0) + 1
        return base if names[base] == 1 else f"{base}{names[base]}"

    for t in [r for r in recs if r["kind"] == "step"]:
        fine = OUT / "truth" / (Path(t["file"]).stem + ".fine.stl")
        m = trimesh.load(fine, process=True)
        e = scan_edge(m.area)
        sc = MESH / (Path(t["file"]).stem + ".scan.stl")
        if not sc.exists():
            s, _ = scanify(m, edge=e, noise=NOISE, seed=7, pose=False)
            s.export(sc)
        base = dict(group=t["group"], product=t["product"], truth_step=t["file"])
        cases.append(base | {"id": cid(t["group"], t["slug"], "fine"), "input": str(fine), "input_kind": "cad_fine"})
        cases.append(base | {"id": cid(t["group"], t["slug"], "scan"), "input": str(sc), "input_kind": "scan_like",
                             "scan_edge_mm": round(e, 3), "scan_noise_mm": NOISE})
        print(t["product"], "edge", round(e, 3), flush=True)
    for s in [r for r in recs if r["kind"] == "stl"]:
        p = next((C.ROOT / "tests" / "user_scans" / SET).rglob(s["file"]))
        mt = s.get("match")
        cases.append({"id": cid(s["group"], s["slug"], "userstl"), "group": s["group"], "product": s["product"],
                      "truth_step": mt["step"] if mt else None, "pair_how": mt["how"] if mt else None,
                      "truth_solid": mt["solid"] if mt else None, "T_stl_to_step": mt["T_stl_to_step"] if mt else None,
                      "pair_p2s_p95": mt["p2s_p95"] if mt else None,
                      "segment_of_candidates": s.get("segment_of_candidates"),
                      "input": str(p.relative_to(C.ROOT)), "input_kind": "user_stl"})
    # budget: skip the second of each L/R mirror twin (same fingerprint within 0.5%) and the CAD-fine run of a
    # STEP that has a user STL (the user STL is already a CAD tessellation); both are recorded, not run
    twin_of = {}
    for t in inv.get("twins", []):
        twin_of.setdefault(t["b"], t["a"])
    by_file = {r["file"]: r for r in recs}
    matched_steps = {s["match"]["step"] for s in recs if s["kind"] == "stl" and s.get("match")}
    for c in cases:
        src = c["truth_step"] if c["input_kind"] != "user_stl" else Path(c["input"]).name
        if src in twin_of:
            c["skip"] = f"mirror twin of {by_file[twin_of[src]]['product']} ({twin_of[src]})"
        elif c["input_kind"] == "cad_fine" and c["truth_step"] in matched_steps:
            c["skip"] = "STEP has a user STL (CAD tessellation) as primary input"
    # one representative per unique part first (user STL, or CAD-fine for a STEP without STL), then scan-like
    pri = {"user_stl": 0, "cad_fine": 1, "scan_like": 2}
    cases.sort(key=lambda c: (pri[c["input_kind"]], c["id"]))
    (OUT / "cases.json").write_text(json.dumps(cases, indent=1))
    print(len(cases), "cases")


if __name__ == "__main__":
    main()
