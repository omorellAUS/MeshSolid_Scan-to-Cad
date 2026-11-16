"""Inventory a user test set: every STEP (truth) and STL (user mesh) under tests/user_scans/<set>/ (recursive),
with geometric fingerprints, STL->STEP pairing, near-duplicate / mirror grouping, previews and a contact sheet.
Usage: python tools/testset/inventory.py drone        -> out/user/drone/inventory.json, previews/, contact_sheet.png"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

SET = sys.argv[1] if len(sys.argv) > 1 else "drone"
IN = C.ROOT / "tests" / "user_scans" / SET
OUT = C.out_dir(SET)
(OUT / "previews").mkdir(parents=True, exist_ok=True)
(OUT / "truth").mkdir(parents=True, exist_ok=True)


def group_of(p):
    rel = p.relative_to(IN).parts
    return rel[0] if rel[0] in ("vtol",) else "drone"


def guess(name, ft, ext, nsol):
    n = name.upper()
    fs = sum(v for k, v in ft.items() if k in ("bspline", "bezier", "offset", "revolution", "extrusion"))
    tot = max(sum(ft.values()), 1)
    shape = "freeform NURBS skin" if fs / tot > 0.3 else ("mostly prismatic" if ft.get("plane", 0) / tot > 0.5 else "mixed analytic")
    if "MODIFIER" in n:
        what = "slicer modifier volume(s) (local infill/wall override for 3D printing)"
    elif "AIL" in n:
        what = "aileron (control surface, airfoil section)"
    elif "WINGTIP" in n:
        what = "wingtip segment"
    elif "WING ROOT" in n:
        what = "wing root segment (airfoil, spar sockets)"
    elif "WING" in n:
        what = "wing panel segment (airfoil section, spar channels)"
    elif "NOSE" in n:
        what = "fuselage nose cone"
    elif "HATCH" in n:
        what = "fuselage hatch cover (curved thin shell)"
    elif "MOTOR MOUNT" in n:
        what = "motor mount (bolt pattern, prismatic)"
    elif "BOOM" in n:
        what = "VTOL boom mount (tube clamp)"
    elif "FUS" in n:
        what = "fuselage section (curved thin shell)"
    else:
        what = "airframe part"
    return f"{what}; {shape}; {nsol} solid(s); {ext[0]:.0f}x{ext[1]:.0f}x{ext[2]:.0f} mm"


def fingerprint(m):
    """Pose-invariant: volume, area, PCA extents (sorted)."""
    P = m.vertices - m.vertices.mean(0)
    w, V = np.linalg.eigh(np.cov(P.T))
    ext = np.ptp(P @ V, axis=0)
    return {"volume": float(abs(m.volume)) if m.is_volume else float(abs(m.volume)), "area": float(m.area),
            "pca_extents": sorted(map(float, ext), reverse=True)}


def render(m, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    if len(m.faces) > 60000:
        keep = np.sort(np.random.default_rng(0).choice(len(m.faces), 60000, replace=False))   # preview only
        m = trimesh.Trimesh(m.vertices, m.faces[keep], process=False)
    fig = plt.figure(figsize=(3.2, 3.2), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    light = np.array([0.4, -0.5, 0.75]); light /= np.linalg.norm(light)
    s = 0.3 + 0.7 * np.abs(m.face_normals @ light)
    ax.add_collection3d(Poly3DCollection(m.vertices[m.faces], facecolors=np.c_[np.outer(s, [0.55, 0.65, 0.85]), np.ones(len(s))],
                                         edgecolors="none", linewidths=0))
    lo, hi = m.bounds; c, r = (lo + hi) / 2, (hi - lo).max() / 2
    ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_box_aspect((1, 1, 1)); ax.view_init(28, -60); ax.set_axis_off()
    ax.set_title(title, fontsize=8)
    fig.subplots_adjust(0, 0, 1, 0.92)
    fig.savefig(path); plt.close(fig)


def step_record(p):
    t = time.time()
    s = C.U.read_step(p)
    sols = C.solids(s)
    lo, hi = C.bbox(s)
    ext = [hi[i] - lo[i] for i in range(3)]
    diag = float(np.linalg.norm(ext))
    ft = C.face_types(s)
    name = C.product_name(p)
    lin = max(0.005, min(0.05, diag * 2e-5))
    m, K = C.tessellate_labeled(s, lin=lin, ang=0.1)
    stl = OUT / "truth" / (p.stem + ".fine.stl")
    m.export(stl)
    zone_area = {}
    for k in np.unique(K):
        z = C.zone_of(k); zone_area[z] = zone_area.get(z, 0.0) + float(m.area_faces[K == k].sum())
    tot = sum(zone_area.values()) or 1.0
    per = []
    for so in sols:
        a, b = C.bbox(so)
        per.append({"volume": round(C.U.volume(so), 2), "n_faces": C.U.count(so, C.TopAbs_FACE), "valid": C.U.check(so),
                    "bbox": [round(b[i] - a[i], 2) for i in range(3)], "face_types": C.face_types(so)})
    rec = {"file": p.name, "group": group_of(p), "kind": "step", "product": name, "slug": C.slug(name), "units": C.units(p),
           "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "n_solids": len(sols),
           "n_shells": C.U.count(s, C.TopAbs_SHELL), "n_faces": C.U.count(s, C.TopAbs_FACE), "face_types": ft,
           "zone_area_frac": {k: round(v / tot, 3) for k, v in zone_area.items()},
           "bbox_min": [round(x, 3) for x in lo], "bbox": [round(x, 3) for x in ext], "volume": round(C.U.volume(s), 2),
           "area": round(C.U.area(s), 2), "valid": C.U.check(s), "solids": per, "fine_stl_lin_defl": lin,
           "fine_stl_tris": int(len(m.faces)), "fine_stl_watertight": bool(m.is_watertight),
           "fingerprint": fingerprint(m)}
    rec["guess"] = guess(name, ft, ext, len(sols))
    rec["preview"] = "previews/" + p.stem + ".png"
    render(m, OUT / rec["preview"], f"{name} [STEP]")
    rec["seconds"] = round(time.time() - t, 1)
    return rec


def stl_record(p):
    raw = p.read_bytes()
    hdr = raw[:80].split(b"\0")[0].decode(errors="ignore").strip()
    name = hdr[6:].strip() if hdr.lower().startswith("solid") else p.stem
    m = trimesh.load(p, process=True)
    body = m.split(only_watertight=False)
    rec = {"file": p.name, "group": group_of(p), "kind": "stl", "product": name, "slug": C.slug(name),
           "sha256": hashlib.sha256(raw).hexdigest(),
           "tris": int(len(m.faces)), "watertight": bool(m.is_watertight), "n_bodies": int(len(body)),
           "bbox_min": [round(x, 3) for x in m.bounds[0]], "bbox": [round(x, 3) for x in m.extents],
           "volume": round(float(abs(m.volume)), 2), "area": round(float(m.area), 2),
           "geom_hash": hashlib.sha256(np.round(np.sort(m.vertices.view(np.float64).reshape(-1, 3), axis=0), 3).tobytes()).hexdigest()[:16],
           "fingerprint": fingerprint(m)}
    rec["preview"] = "previews/" + p.stem + ".png"
    render(m, OUT / rec["preview"], f"{name} [STL]")
    return rec


def rel(a, b):
    return abs(a - b) / max(abs(a), abs(b), 1e-9)


def fp_dist(a, b):
    fa, fb = a["fingerprint"], b["fingerprint"]
    return max(rel(fa["volume"], fb["volume"]), rel(fa["area"], fb["area"]),
               *[rel(x, y) for x, y in zip(fa["pca_extents"], fb["pca_extents"])])


def same_pose(a, b, tol=0.5):
    return all(abs(x - y) < tol for x, y in zip(a["bbox_min"] + a["bbox"], b["bbox_min"] + b["bbox"]))


def main():
    recs = []
    for p in sorted(IN.rglob("*.step")) + sorted(IN.rglob("*.stl")):
        try:
            recs.append(step_record(p) if p.suffix == ".step" else stl_record(p))
            print(recs[-1]["kind"], recs[-1]["file"], recs[-1]["product"], recs[-1].get("seconds", ""), flush=True)
        except Exception as ex:  # noqa: BLE001
            recs.append({"file": p.name, "kind": p.suffix[1:], "error": repr(ex)}); print("ERR", p, ex, flush=True)
    steps = [r for r in recs if r["kind"] == "step" and "error" not in r]
    stls = [r for r in recs if r["kind"] == "stl" and "error" not in r]
    # STL <-> STEP pairing by pose-invariant fingerprint (mirror images share it; disambiguate by pose / name)
    pairs = []
    for s in stls:
        cands = sorted(((fp_dist(s, t), t) for t in steps), key=lambda x: x[0])
        best = None
        for d, t in cands:
            if d > 0.02:
                break
            if same_pose(s, t, tol=max(0.5, 1e-3 * max(t["bbox"]))):
                best = (d, t, "same_pose"); break
            if best is None:
                best = (d, t, "fingerprint_only (pose/mirror differs)")
        s["match"] = None if best is None else {"step": best[1]["file"], "product": best[1]["product"],
                                                 "fp_dist": round(best[0], 5), "how": best[2]}
        if best:
            pairs.append((s["file"], best[1]["file"]))
    for t in steps:
        t["matched_stls"] = [s["file"] for s in stls if s.get("match") and s["match"]["step"] == t["file"]]
    # near-duplicates / mirror twins among all meshes (fingerprint within 0.5%)
    allr = steps + stls
    twins = []
    for i, a in enumerate(allr):
        for b in allr[i + 1:]:
            if a["kind"] == b["kind"] and fp_dist(a, b) < 0.005:
                twins.append({"a": a["file"], "a_name": a["product"], "b": b["file"], "b_name": b["product"],
                              "kind": a["kind"], "fp_dist": round(fp_dist(a, b), 5), "same_pose": same_pose(a, b)})
    exact_dups = {}
    for r in allr:
        exact_dups.setdefault(r["sha256"], []).append(r["file"])
    geo_dups = {}
    for r in stls:
        geo_dups.setdefault(r["geom_hash"], []).append(r["file"])
    inv = {"set": SET, "records": recs, "pairs": pairs, "twins": twins,
           "exact_duplicates": [v for v in exact_dups.values() if len(v) > 1],
           "geometric_duplicates": [v for v in geo_dups.values() if len(v) > 1]}
    (OUT / "inventory.json").write_text(json.dumps(inv, indent=1))
    contact_sheet(recs)
    print("pairs", len(pairs), "twins", len(twins))


def contact_sheet(recs):
    from PIL import Image
    ims = [r for r in recs if "preview" in r]
    ims.sort(key=lambda r: (r.get("group", ""), r["kind"], r["product"]))
    n = len(ims); cols = 10; rows = (n + cols - 1) // cols
    W = 320
    sheet = Image.new("RGB", (cols * W, rows * W), "white")
    for i, r in enumerate(ims):
        im = Image.open(OUT / r["preview"]).convert("RGB").resize((W, W))
        sheet.paste(im, ((i % cols) * W, (i // cols) * W))
    sheet.save(OUT / "contact_sheet.png")


if __name__ == "__main__":
    main()
