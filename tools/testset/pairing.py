"""STL <-> STEP pairing at solid level, verified by rigid alignment (proper or mirrored), plus print-segment,
L/R-mirror-twin and drone-vs-VTOL variant notes. Rewrites the 'match' fields in out/user/<set>/inventory.json
and adds 'pairing'. Usage: python tools/testset/pairing.py drone"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

SET = sys.argv[1] if len(sys.argv) > 1 else "drone"
IN = C.ROOT / "tests" / "user_scans" / SET
OUT = C.out_dir(SET)


def fp(m):
    P = m.vertices - m.vertices.mean(0)
    w, V = np.linalg.eigh(np.cov(P.T))
    return {"volume": float(abs(m.volume)), "area": float(m.area), "pca": sorted(np.ptp(P @ V, axis=0).tolist(), reverse=True)}


def rel(a, b):
    return abs(a - b) / max(abs(a), abs(b), 1e-9)


def close(a, b, tol=0.03):
    return rel(a["volume"], b["volume"]) < tol and rel(a["area"], b["area"]) < tol and all(rel(x, y) < tol for x, y in zip(a["pca"][:2], b["pca"][:2]))


def icp(A, B, mirror=False, iters=40):
    """rigid transform B->A (optionally with a mirror); returns (T, rms, p95) on the trimmed-free residual."""
    tA = cKDTree(A)
    def pca(X):
        c = X.mean(0); w, V = np.linalg.eigh(np.cov((X - c).T)); return c, V[:, ::-1]
    ca, Ua = pca(A); cb, Ub = pca(B)
    best = None
    M = np.diag([-1.0, 1, 1]) if mirror else np.eye(3)
    Bm = B @ M.T
    cb, Ub = pca(Bm)
    for s in [(a, b, c) for a in (1, -1) for b in (1, -1) for c in (1, -1)]:
        R = Ua @ np.diag(s) @ Ub.T
        if np.linalg.det(R) < 0:
            continue
        t = ca - R @ cb
        for _ in range(iters):
            Bt = Bm @ R.T + t
            d, j = tA.query(Bt, workers=-1)
            k = d < np.percentile(d, 90)
            P, Q = Bt[k], A[j[k]]
            pc, qc = P.mean(0), Q.mean(0)
            Uu, _, Vt = np.linalg.svd((P - pc).T @ (Q - qc))
            Ri = (Uu @ Vt).T
            if np.linalg.det(Ri) < 0:
                Vt[-1] *= -1; Ri = (Uu @ Vt).T
            R = Ri @ R; t = Ri @ (t - pc) + qc
        d = tA.query(Bm @ R.T + t, workers=-1)[0]
        if best is None or np.median(d) < best[1]:
            T = np.eye(4); T[:3, :3] = R @ M; T[:3, 3] = t
            best = (T, float(np.median(d)), float(np.sqrt(np.mean(d ** 2))), float(np.percentile(d, 95)))
    return best


def base_name(n):
    n = n.upper().replace("AILERON", "AIL").replace("MODIFIER2", "MODIFIER").replace("BOTOM", "BOTTOM")
    n = re.sub(r"\bV2\b|\bVTOL\b|\bCLEAN\b", " ", n)
    n = re.sub(r"(\d)\s*([LR])\b", r"\1 \2", n)
    return " ".join(n.split())


def side(n):
    m = re.search(r"\b([LR])\b", base_name(n))
    return m.group(1) if m else None


def family(n):
    """name with segment number and side removed: 'AIL 1 R' -> 'AIL'"""
    b = base_name(n)
    b = re.sub(r"\b[LR]\b", " ", b); b = re.sub(r"\b\d+\b", " ", b); b = b.replace("MODIFIER", " ")
    b = re.sub(r"(\D)\d+\b", r"\1", b)
    return " ".join(b.split())


def main():
    inv = json.loads((OUT / "inventory.json").read_text())
    recs = [r for r in inv["records"] if "error" not in r]
    steps = [r for r in recs if r["kind"] == "step"]
    stls = [r for r in recs if r["kind"] == "stl"]
    solids = []    # (step rec, solid index or -1 for whole, fp, sample points)
    for t in steps:
        s = C.U.read_step(next(IN.rglob(t["file"])))
        sols = C.solids(s)
        whole, _ = C.tessellate_labeled(s, lin=0.02, ang=0.1)
        solids.append((t, -1, fp(whole), trimesh.sample.sample_surface(whole, 60000, seed=1)[0], whole))
        if len(sols) > 1:
            ms = []
            for i, so in enumerate(sols):
                m, _ = C.tessellate_labeled(so, lin=0.02, ang=0.1)
                ms.append(m)
                solids.append((t, i, fp(m), trimesh.sample.sample_surface(m, 60000, seed=1)[0], m))
            if 2 < len(sols) <= 6:   # print part = two STEP solids glued (e.g. hatch L+R halves)
                for i in range(len(sols)):
                    for j in range(i + 1, len(sols)):
                        m = trimesh.util.concatenate([ms[i], ms[j]])
                        solids.append((t, f"{i}+{j}", fp(m), trimesh.sample.sample_surface(m, 60000, seed=1)[0], m))
        print("step", t["product"], len(sols), flush=True)
    pairing = []
    for s in stls:
        m = trimesh.load(next(IN.rglob(s["file"])), process=True)
        f = fp(m)
        B = trimesh.sample.sample_surface(m, 30000, seed=2)[0]
        cands = [x for x in solids if rel(f["volume"], x[2]["volume"]) < 0.05 and rel(f["area"], x[2]["area"]) < 0.05]
        best = None
        for t, i, ft, A, tm in cands:
            for mir in (False, True):
                T, med, _, _ = icp(A, B, mir)
                # verify with point-to-surface distance (nearest-sample distance is dominated by sample spacing)
                d = np.abs(C.p2s(trimesh.transform_points(B[::3], T), tm, n=600000, seed=3))
                diag = float(np.linalg.norm(ft["pca"]))
                thr = max(0.2, 1e-3 * diag)
                frac = float(np.mean(d < thr))
                cand = {"step": t["file"], "product": t["product"], "solid": i, "mirrored": mir,
                        "p2s_rms": round(float(np.sqrt(np.mean(d ** 2))), 4), "p2s_p95": round(float(np.percentile(d, 95)), 4),
                        "frac_within": round(frac, 4), "thr_mm": round(thr, 3), "vol_rel": round(rel(f["volume"], ft["volume"]), 4),
                        "area_rel": round(rel(f["area"], ft["area"]), 4),
                        # ICP on multi-shell print STLs sometimes stops just short; volume+area equal to 0.1% backs it up
                        "ok": bool(frac > 0.95 or (frac > 0.9 and rel(f["volume"], ft["volume"]) < 1e-3 and rel(f["area"], ft["area"]) < 1e-3)),
                        "T_stl_to_step": np.round(T, 6).tolist()}
                same_name = base_name(t["product"]).replace("MODIFIER", "").split() == base_name(s["product"]).split()
                key = (cand["ok"], round(frac, 2), not mir, same_name, -cand["p2s_p95"])   # prefer unmirrored, same name on ties
                if best is None or key > best[0]:
                    best = (key, cand)
        best = best and best[1]
        if best and best["ok"]:
            best["how"] = ("mirror of " if best["mirrored"] else "") + ("whole STEP" if best["solid"] == -1 else f"STEP solid #{best['solid']}")
            s["match"] = best
        else:
            s["match"] = None
            s["match_near_miss"] = best and {k: v for k, v in best.items() if k != "T_stl_to_step"}
        # print-segment of a STEP body? same family name, smaller volume
        fam = [(t, i, ft) for t, i, ft, _, _ in solids if family(t["product"]) == family(s["product"]) and t["group"] == s["group"]]
        s["family"] = family(s["product"])
        s["segment_of_candidates"] = sorted({t["product"] for t, i, ft in fam if ft["volume"] > 1.1 * f["volume"]})
        print("stl", s["product"], "->", s["match"] and s["match"]["how"] + " " + s["match"]["product"], flush=True)
        pairing.append({"stl": s["file"], "stl_name": s["product"], "group": s["group"],
                        "match": s["match"] and {k: v for k, v in s["match"].items() if k != "T_stl_to_step"},
                        "near_miss": s.get("match_near_miss"), "family": s["family"], "segment_of_candidates": s["segment_of_candidates"]})
    for t in steps:
        t["matched_stls"] = [s["file"] for s in stls if s.get("match") and s["match"]["step"] == t["file"]]
        t["family"] = family(t["product"])
    # family volume sums: are the STL segments of a family the cut-up STEP body?
    fams = {}
    for r in recs:
        fams.setdefault((r["group"], family(r["product"]), side(r["product"])), []).append(r)
    fam_notes = []
    for (g, fa, sd), rs in sorted(fams.items(), key=lambda x: (x[0][0], x[0][1], x[0][2] or "")):
        st = [r for r in rs if r["kind"] == "step"]; sl = [r for r in rs if r["kind"] == "stl"]
        if st and sl:
            fam_notes.append({"group": g, "family": fa, "side": sd, "steps": [(r["product"], r["volume"]) for r in st],
                              "stls": [(r["product"], r["volume"]) for r in sl], "stl_volume_sum": round(sum(r["volume"] for r in sl), 1)})
    # drone vs VTOL variants: same family, compare fingerprints
    variants = []
    for a in [r for r in recs if r["group"] == "vtol"]:
        for b in [r for r in recs if r["group"] == "drone" and r["kind"] == a["kind"]]:
            if family(a["product"]) == family(b["product"]) and side(a["product"]) in (side(b["product"]), None):
                fa_, fb_ = a["fingerprint"], b["fingerprint"]
                d = max(rel(fa_["volume"], fb_["volume"]), rel(fa_["area"], fb_["area"]))
                variants.append({"vtol": a["product"], "drone": b["product"], "kind": a["kind"], "vol_vtol": a["volume"], "vol_drone": b["volume"],
                                 "max_rel_diff_vol_area": round(d, 4), "verdict": "near-identical (<1%)" if d < 0.01 else "variant"})
    inv["pairing"] = pairing; inv["family_notes"] = fam_notes; inv["vtol_vs_drone"] = variants
    inv["pairs"] = [(p["stl"], p["match"]["step"]) for p in pairing if p["match"]]
    (OUT / "inventory.json").write_text(json.dumps(inv, indent=1))
    print("pairs", len(inv["pairs"]))


if __name__ == "__main__":
    main()
