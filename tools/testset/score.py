"""Score every engine run of a user test set against its STEP truth; numbers only (no geometry) ->
results/testset_<set>.json. Usage: python tools/testset/score.py drone

Per case: status, failure, last stage, valid single solid, face count, volume error, truth-surface->solid
rms/p95/max (overall and by truth zone: planar / cylindrical / conical_toroidal / freeform), solid->truth,
input(scan)->solid rms/p95, snapped dimensions, per-body stats for multi-body truths, run time.
User STLs that are not in the STEP pose are rigidly aligned to the truth first (PCA + ICP; recorded)."""
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
OUT = C.out_dir(SET)
RUNS = OUT / "runs"
N_TRUTH = 150_000


def truth_samples(step_file, solid_sel=-1):
    p = next((C.ROOT / "tests" / "user_scans" / SET).rglob(step_file))
    s = C.U.read_step(p)
    if solid_sel not in (-1, None):      # the user STL is one STEP solid, or two glued ones ("i+j")
        from OCP.BRep import BRep_Builder
        from OCP.TopoDS import TopoDS_Compound
        idx = [int(x) for x in str(solid_sel).split("+")]
        sols = C.solids(s)
        comp = TopoDS_Compound(); bb = BRep_Builder(); bb.MakeCompound(comp)
        for i in idx:
            bb.Add(comp, sols[i])
        s = comp
    m, K = C.tessellate_labeled(s, lin=0.01, ang=0.1)
    P, fi = trimesh.sample.sample_surface(m, N_TRUTH, seed=11)
    Z = np.array([C.zone_of(k) for k in K])[fi]
    # body label per sample: nearest truth solid
    sols = C.solids(s)
    body = np.zeros(len(P), int)
    if len(sols) > 1:
        dmin = np.full(len(P), np.inf)
        for i, so in enumerate(sols):
            mb, _ = C.tessellate_labeled(so, lin=0.02, ang=0.2)
            Q = trimesh.sample.sample_surface(mb, 200_000, seed=i)[0]
            d, _ = cKDTree(Q).query(P, workers=-1)
            body[d < dmin] = i; dmin = np.minimum(d, dmin)
    return s, m, P, Z, body, [C.U.volume(so) for so in sols]


def align_to(A, B):
    """rigid (proper) transform taking points B onto A: PCA sign candidates + trimmed ICP. Returns (4x4, median)."""
    tA = cKDTree(A)
    def pca(X):
        c = X.mean(0); w, V = np.linalg.eigh(np.cov((X - c).T)); return c, V[:, ::-1]
    ca, Ua = pca(A); cb, Ub = pca(B)
    best = None
    for s in [(a, b, c) for a in (1, -1) for b in (1, -1) for c in (1, -1)]:
        R = Ua @ np.diag(s) @ Ub.T
        if np.linalg.det(R) < 0:
            continue
        t = ca - R @ cb
        for _ in range(30):
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
            best = (R, t, med)
    T = np.eye(4); T[:3, :3] = best[0]; T[:3, 3] = best[1]
    return T, best[2]


def score_case(c, truth_cache, inv_by_file):
    rp = RUNS / f"{c['id']}.report.json"
    out = {k: c.get(k) for k in ("id", "group", "product", "input_kind", "truth_step", "truth_solid", "pair_how", "pair_p2s_p95",
                                  "segment_of_candidates", "scan_edge_mm", "scan_noise_mm")}
    if c.get("skip"):
        out["status"] = "skipped"; out["skip"] = c["skip"]; return out
    if not rp.exists():
        out["status"] = "not_run"; return out
    import subprocess
    if subprocess.run(["pgrep", "-f", f"testset {SET} {c['id']}$"], capture_output=True).returncode == 0:
        out["status"] = "running"; return out
    rep = json.loads(rp.read_text())
    b = rep.get("stages", {}).get("build", {}) if isinstance(rep.get("stages"), dict) else {}
    out |= {"status": rep.get("status"), "failure": (rep.get("failure") or "")[:400] or None, "last_stage": rep.get("last_stage"),
            "seconds": round(rep.get("seconds", 0) or 0, 1), "build_method": b.get("method"),
            "build_reason": (b.get("reason") or "")[:300] or None, "n_regions": len(rep.get("regions", []) or []),
            "tol": (rep.get("tolerance") or {}).get("tol"), "noise_est": (rep.get("tolerance") or {}).get("noise_sigma_est"),
            "recognizers": {k: (rep.get(k) or {}).get("detected") if isinstance(rep.get(k), dict) else None
                            for k in ("gear", "screw", "impeller", "revolve", "threads", "freeform")},
            "engine_dev": {k: (rep.get("deviation") or {}).get(k, {}) and {kk: round(vv, 4) for kk, vv in (rep["deviation"][k]).items() if kk in ("rms", "p95", "max")}
                           for k in ("mesh_vertices_to_solid", "solid_to_mesh")} if rep.get("deviation") else None,
            "gate_passed": ((rep.get("deviation") or {}).get("acceptance") or {}).get("passed"),
            "dimensions": [{"name": d["name"], "type": d["type"], "value": round(d["value"], 4), "fit": round(d["fit_value"], 4)}
                           for d in (rep.get("dimensions") or [])][:40]}
    try:
        tl = [ln for ln in (RUNS / "runs.tsv").read_text().splitlines() if ln.split("\t")[0] == c["id"]]
        if tl:
            out["wall_seconds"] = int(tl[-1].split("secs=")[1]); out["rc"] = int(tl[-1].split("rc=")[1].split("\t")[0])
    except FileNotFoundError:
        pass
    rc = out.get("rc")
    log = RUNS / f"{c['id']}.log"
    logtxt = log.read_text(errors="ignore") if log.exists() else ""
    killed = rc in (-9, 124, 137) or (rc is None and "TIMEOUT" in logtxt) or \
        (rc is None and "Traceback" not in logtxt and "status:" not in logtxt and out.get("seconds", 0) == 0)
    if killed and out["status"] not in ("ok", "ok_no_output", "failed"):
        # SIGTERM is not handled while OCC holds the GIL: the runner's SIGKILL after the grace period ends it
        out["status"] = "timeout"; out["failure"] = f"timeout ({out.get('wall_seconds')} s) during stage '{out.get('last_stage')}'"
    elif rc not in (0, 2, None) and "Traceback" in logtxt:
        exl = [ln for ln in logtxt.splitlines() if re.match(r"^[A-Za-z_][\w.]*(Error|Exception|NullObject|Failure|Signal)\b", ln)]
        exc = exl[-1] if exl else "exception (see log)"
        frames = [ln.strip() for ln in logtxt.splitlines() if ln.strip().startswith('File "') and "/meshsolid/" in ln]
        out["status"] = "exception"; out["failure"] = exc[:300]; out["exception_at"] = frames[-1][-160:] if frames else None
    sp = RUNS / f"{c['id']}.step"
    if not sp.exists():
        out["solid"] = None
        return out
    s = C.U.read_step(sp)
    sol = C.solids(s)
    lo, hi = C.bbox(s)
    diag = float(np.linalg.norm(np.subtract(hi, lo)))
    vol = C.U.volume(s)
    out |= {"valid": C.U.check(s), "n_solids": len(sol), "n_faces": C.U.count(s, C.TopAbs_FACE), "face_types": C.face_types(s),
            "volume": round(vol, 2), "valid_single_solid": bool(C.U.check(s) and len(sol) == 1)}
    sm = C.U.shape_to_trimesh(s, lin=max(0.003, diag * 1e-5), ang=0.05)
    inp = trimesh.load(C.ROOT / c["input"], process=False)
    Vin = inp.vertices[np.random.default_rng(3).choice(len(inp.vertices), min(150_000, len(inp.vertices)), replace=False)]
    out["input_to_solid"] = C.stats(C.p2s(Vin, sm))
    if not c.get("truth_step"):
        return out
    key = (c["truth_step"], str(c.get("truth_solid", -1)))
    if key not in truth_cache:
        truth_cache[key] = truth_samples(c["truth_step"], c.get("truth_solid", -1))
    ts, tm, P, Z, body, bvols = truth_cache[key]
    tv = round(float(sum(bvols)), 2) if c.get("truth_solid", -1) not in (-1, None) else inv_by_file[c["truth_step"]]["volume"]
    out["truth_volume"] = tv
    out["volume_err_pct"] = round(100 * (vol - tv) / tv, 3) if tv else None
    # user STL in a different pose: bring the solid into the truth frame
    if c["input_kind"] == "user_stl" and c.get("T_stl_to_step") is not None:
        # same rigid (or mirror) transform that registered the user STL onto the STEP in pairing.py
        sm.apply_transform(np.array(c["T_stl_to_step"]))
        out["aligned_to_truth"] = {"pairing_p2s_p95": c.get("pair_p2s_p95"), "mirrored": "mirror" in (c.get("pair_how") or "")}
    d = C.p2s(P, sm)
    out["truth_to_solid"] = C.stats(d)
    out["truth_to_solid_by_zone"] = {z: C.stats(d[Z == z]) for z in np.unique(Z)}
    Ps = trimesh.sample.sample_surface(sm, 100_000, seed=4)[0]
    out["solid_to_truth"] = C.stats(C.p2s(Ps, tm))
    if len(bvols) > 1:
        out["per_body"] = [{"body": i, "truth_volume": round(v, 2), "truth_to_solid": C.stats(d[body == i])} for i, v in enumerate(bvols)]
    return out


def main():
    inv = json.loads((OUT / "inventory.json").read_text())
    inv_by_file = {r["file"]: r for r in inv["records"]}
    cases = json.loads((OUT / "cases.json").read_text())
    cache, res = {}, []
    for c in sorted(cases, key=lambda c: c.get("truth_step") or "~"):
        try:
            r = score_case(c, cache, inv_by_file)
        except Exception as ex:  # noqa: BLE001
            r = {"id": c["id"], "score_error": repr(ex)[:300]}
        res.append(r); print(r["id"], r.get("status"), r.get("volume_err_pct"), (r.get("truth_to_solid") or {}).get("rms"), flush=True)
    keep = ("file", "group", "kind", "product", "units", "n_solids", "n_faces", "face_types", "zone_area_frac", "bbox", "volume",
            "area", "valid", "solids", "guess", "tris", "watertight", "n_bodies", "match", "matched_stls", "sha256")
    doc = {"set": SET, "inventory": [{k: r.get(k) for k in keep if k in r} for r in inv["records"]],
           "twins": inv["twins"], "exact_duplicates": inv["exact_duplicates"], "geometric_duplicates": inv["geometric_duplicates"],
           "cases": res}
    (C.ROOT / "results" / f"testset_{SET}.json").write_text(json.dumps(doc, indent=1, default=float))


if __name__ == "__main__":
    main()
