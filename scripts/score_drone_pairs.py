"""Score drone_pairs_all engine outputs against matched STEP truth portions. Numbers only."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "testset"))
import common as C  # noqa: E402
from meshsolid import occ_util as U  # noqa: E402

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/workspace/meshsolid_testset/drone_pairs_all")
IN = ROOT / "tests" / "user_scans" / "drone"
N = 80_000


def truth_shape(step_file, solid_sel):
    p = next(IN.rglob(step_file))
    s = U.read_step(p)
    if solid_sel in (-1, None, "", "-1"):
        return s
    from OCP.BRep import BRep_Builder
    from OCP.TopoDS import TopoDS_Compound
    idx = [int(x) for x in str(solid_sel).split("+")]
    sols = C.solids(s)
    comp = TopoDS_Compound(); bb = BRep_Builder(); bb.MakeCompound(comp)
    for i in idx:
        bb.Add(comp, sols[i])
    return comp


def align(A, B):
    tA = cKDTree(A)
    def pca(X):
        c = X.mean(0); _, V = np.linalg.eigh(np.cov((X - c).T)); return c, V[:, ::-1]
    ca, Ua = pca(A); cb, Ub = pca(B)
    best = None
    for s in [(a, b, c) for a in (1, -1) for b in (1, -1) for c in (1, -1)]:
        R = Ua @ np.diag(s) @ Ub.T
        if np.linalg.det(R) < 0:
            continue
        t = ca - R @ cb
        for _ in range(25):
            Bt = B @ R.T + t
            d, j = tA.query(Bt, workers=-1)
            k = d < np.percentile(d, 90)
            P, Q = Bt[k], A[j[k]]
            pc, qc = P.mean(0), Q.mean(0)
            Uu, _, Vt = np.linalg.svd((P - pc).T @ (Q - qc))
            Ri = (Uu @ Vt).T
            if np.linalg.det(Ri) < 0:
                Vt[-1] *= -1; Ri = (Uu @ Vt).T
            R = Ri @ R; t = Ri @ (t - pc) + qc
        med = float(np.median(tA.query(B @ R.T + t, workers=-1)[0]))
        if best is None or med < best[1]:
            T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t
            best = (T, med)
    return best[0]


def stats(d):
    d = np.asarray(d, float)
    return {
        "rms_mm": round(float(np.sqrt(np.mean(d ** 2))), 4),
        "p95_mm": round(float(np.percentile(d, 95)), 4),
        "max_mm": round(float(np.max(d)), 4),
        "n_samples": int(len(d)),
    }


def score_row(row):
    runs = OUT / "runs"
    rep_p = runs / f"{row['id']}.report.json"
    step_p = runs / f"{row['id']}.step"
    out = dict(row)
    out["units"] = "mm"
    out["sample_method"] = f"trimesh surface sample n={N} seed=11; point-to-mesh via common.p2s; rigid PCA+ICP align of rebuilt samples onto truth"
    if not rep_p.exists():
        out.update(status="missing", failure="no report")
        return out
    rep = json.loads(rep_p.read_text())
    out["status"] = rep.get("status")
    out["failure"] = rep.get("failure") or rep.get("reason")
    out["last_stage"] = rep.get("last_stage")
    out["engine_seconds"] = round(float(rep.get("seconds") or 0), 1)
    out["valid_solid"] = None
    out["n_solids"] = None
    out["n_faces"] = None
    out["vol_err_pct"] = None
    out["truth_to_solid"] = None
    out["solid_to_truth"] = None
    if not step_p.exists() or out["status"] != "ok":
        return out
    try:
        rebuilt = U.read_step(step_p)
        sols = C.solids(rebuilt)
        out["n_solids"] = len(sols)
        out["valid_solid"] = len(sols) >= 1 and all(U.is_valid(s) for s in sols)
        rb_m, _ = C.tessellate_labeled(rebuilt, lin=0.015, ang=0.1)
        from OCP.TopAbs import TopAbs_FACE
        out["n_faces"] = int(sum(1 for _ in C.explore(rebuilt, TopAbs_FACE)))
        truth = truth_shape(row["truth_step"], row["truth_solid"])
        tr_m, _ = C.tessellate_labeled(truth, lin=0.015, ang=0.1)
        Pt = trimesh.sample.sample_surface(tr_m, N, seed=11)[0]
        Pr = trimesh.sample.sample_surface(rb_m, N, seed=12)[0]
        T = align(Pt, Pr)
        Pr_a = trimesh.transform_points(Pr, T)
        # truth -> solid (samples on truth, distance to rebuilt mesh)
        d_ts = np.abs(C.p2s(Pt, rb_m.apply_transform(T) if False else trimesh.Trimesh(Pr_a, rb_m.faces), n=400_000, seed=3))
        # simpler: nearest to aligned rebuilt samples as proxy + p2s on transformed rebuilt mesh
        rb_a = rb_m.copy()
        rb_a.apply_transform(T)
        d_ts = np.abs(C.p2s(Pt, rb_a, n=400_000, seed=3))
        d_st = np.abs(C.p2s(Pr_a, tr_m, n=400_000, seed=4))
        out["truth_to_solid"] = stats(d_ts)
        out["solid_to_truth"] = stats(d_st)
        vt = float(sum(U.volume(s) for s in C.solids(truth)))
        vr = float(sum(U.volume(s) for s in sols))
        if vt > 1e-6:
            out["vol_err_pct"] = round(100.0 * (vr - vt) / abs(vt), 3)
            out["truth_volume_mm3"] = round(vt, 3)
            out["rebuilt_volume_mm3"] = round(vr, 3)
    except Exception as e:
        out["score_error"] = f"{type(e).__name__}: {e}"
    return out


def main():
    rows = json.loads((OUT / "engine_rows.json").read_text())
    scored = [score_row(r) for r in rows]
    (OUT / "pair_results.json").write_text(json.dumps(scored, indent=2))
    # CSV
    flat = []
    for r in scored:
        flat.append({
            "id": r["id"], "group": r["group"], "stl_name": r["stl_name"],
            "truth_product": r["truth_product"], "truth_solid": r["truth_solid"],
            "pair_how": r["pair_how"], "pair_mirrored": r["pair_mirrored"],
            "status": r.get("status"), "failure": (r.get("failure") or "")[:180],
            "valid_solid": r.get("valid_solid"), "n_solids": r.get("n_solids"),
            "n_faces": r.get("n_faces"), "vol_err_pct": r.get("vol_err_pct"),
            "truth_to_solid_rms_mm": (r.get("truth_to_solid") or {}).get("rms_mm"),
            "truth_to_solid_p95_mm": (r.get("truth_to_solid") or {}).get("p95_mm"),
            "solid_to_truth_rms_mm": (r.get("solid_to_truth") or {}).get("rms_mm"),
            "solid_to_truth_p95_mm": (r.get("solid_to_truth") or {}).get("p95_mm"),
            "engine_seconds": r.get("engine_seconds"), "wall_seconds": r.get("wall_seconds"),
            "build_method": r.get("build_method"), "units": "mm",
            "sample_method": r.get("sample_method"),
        })
    with (OUT / "pair_results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(flat[0])); w.writeheader(); w.writerows(flat)
    # MD summary
    from collections import Counter
    c = Counter(r.get("status") for r in scored)
    lines = ["# Drone / VTOL matched-pair results", "",
             f"Out: `{OUT}` (baseline `/workspace/meshsolid_testset/drone` untouched).",
             f"Cases: {len(scored)}. Status counts: {dict(c)}.",
             "Deviation units: mm. Sample method: surface samples + point-to-mesh; rebuilt rigidly aligned to truth.",
             "Success = rebuilt solid from STL compared to matched STEP portion (not importing source STEP).",
             "",
             "| id | status | valid | faces | vol err % | truth→solid p95 | solid→truth p95 | eng s | wall s | failure |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in flat:
        lines.append(
            f"| {r['id']} | {r['status']} | {r['valid_solid']} | {r['n_faces']} | {r['vol_err_pct']} | "
            f"{r['truth_to_solid_p95_mm']} | {r['solid_to_truth_p95_mm']} | {r['engine_seconds']} | "
            f"{r['wall_seconds']} | {(r['failure'] or '')[:80]} |"
        )
    (OUT / "pair_results.md").write_text("\n".join(lines) + "\n")
    print("SCORED", len(scored), dict(c))


if __name__ == "__main__":
    main()
