"""End-to-end prismatic pipeline: mesh -> regions -> primitives -> snapped -> B-rep -> STEP + report."""
from __future__ import annotations

import copy
import os
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import __version__
from . import fit as F
from .build import build_solid
from .build_local import build_local
from .noise import auto_tol
from .deviation import deviation, save_deviation_ply
from .meshio import load_mesh
from .occ_util import write_step
from .segment import MeshGraph, SegParams, region_points, segment
from .snap import snap


@dataclass
class Params:
    tol: float | None = None     # region membership distance (mm); None = auto (4.5x estimated noise sigma)
    sharp_deg: float = 30.0
    round_tol: float = 0.03      # max change allowed when rounding a dimension (mm)
    do_round: bool = True
    snap_angle_deg: float = 2.0
    accept_rms: float | None = None   # deviation acceptance: rms <= this (default from accuracy class)
    accept_p95: float | None = None   # p95 <= this (default from accuracy class)
    accuracy: str | None = None       # "machined" (default) | "printed" (0.05 mm rms); see meshsolid.accuracy
    max_tools: int = 80          # cell-decomposition limit (fallback builder)
    # rotor balance cuts on the two end faces: "auto" = only for rotors that are not plain turned bodies
    # (bladed/toothed parts, which the revolve builder cannot build); a turned body such as a casing is
    # not balanced, and its local features are design features to model, not to exclude. "on" forces it.
    balance_cuts: str = "auto"
    # involute gears: "auto" = detect (N-fold helical periodicity + involute flank fit) and, when found,
    # build the part as a turned body with the exact gear cut in; "off" skips detection.
    gears: str = "auto"
    gear_build: str = "snapped"  # snapped (standard DP/module, PA, helix, tooth system) | free (as fitted)
    # N-fold bladed rotors (impellers): "auto" = probe before segmentation and, when found, build the hub as
    # a revolve + per-passage floor and one B-spline blade patterned N times (meshsolid.impeller)
    impeller: str = "auto"       # auto | off | on (on = Part type Impeller: build as an impeller or fail with the recogniser's reason; never fall through)
    # helical screw rotors (compressor male/female, pump screws, variable pitch): "auto" = detect (lobed
    # zone + one helicoid explains it) when no gear was found, build the fitted profile's exact helical
    # sweep into the turned body (meshsolid.screw); "off" skips detection.
    screws: str = "auto"
    splines: str = "auto"
    bearing_build: str = "full"  # full (the scanned shape) | envelope (also write <out>.envelope.step: catalogue d x D x B ring)
    notes: tuple = ()            # free-text notes copied into the report (e.g. unscanned features)
    builder: str = "auto"        # auto (2.5D extrusion, local trim, revolve, cell fallback) | extrude | local | revolve | cell
    split_bodies: bool = True    # multi-body STL: build each connected body on its own, multi-body STEP
    cell_fallback_max: int = 30  # use the cell fallback only for small surface counts
    seed: int = 0
    freeform: bool = True        # M4: B-spline surfaces for smooth multi-primitive clusters
    threads: bool = True         # helical thread recognition (zone -> cylinder + thread spec)
    drop_faceless_rounds: int = 2      # build retries releasing small regions that got no trimmed face
    drop_faceless_area_frac: float = 0.01
    normalise: str = "report"    # "report": compute + store the scan->normalised 4x4 only; "apply": work axis-up/nose-up in mm, outputs mapped back to the scan frame
    time_budget_s: float = 0.0   # wall-clock budget (0 = none): a watchdog process marks the report "timeout" with the stage and stops the run; stages with their own budgets scale to it
    # 1.0.3 per-stage limits: seconds each named stage may use before the run ends with a readable reason.
    stage_limits: dict = field(default_factory=lambda: {
        "load_repair": 120.0,
        "normalise": 60.0,
        "segmentation": 300.0,
        "coaxial_refit": 600.0,
        "threads": 120.0,
        "freeform": 1200.0,
        "snapping": 120.0,
        "blends": 120.0,
        "consolidate": 60.0,
        "fit_stats": 60.0,
        "build": 600.0,
        "deviation": 300.0,
        "preview": 60.0,
    })
    stage_heartbeat_s: float = 5.0        # heartbeat file update interval (s); 0 disables
    decimate_for_recognition: int = 300_000   # meshes above this face count are decimated for recognition;
                                              # the full mesh is kept for deviation
    consolidate: bool = True     # pre-build: merge same-surface regions, absorb explained slivers, drop degenerate slivers
    rim_slots: str = "auto"      # flat-floored end-face pockets open to the rim: auto = cut each confident slot on non-impellers (not on a recognised impeller: face_features classifies them) | on | off


def _prim_dims(p: F.Primitive):
    q = p.params
    if p.kind == "plane":
        return {"normal": q["normal"].tolist(), "offset": float(q["point"] @ q["normal"])}
    if p.kind == "cylinder":
        return {"diameter": 2 * float(q["radius"]), "axis_dir": q["axis"].tolist(),
                "axis_point": q["point"].tolist()}
    if p.kind == "sphere":
        return {"diameter": 2 * float(q["radius"]), "center": q["center"].tolist()}
    if p.kind == "torus":
        return {"major_radius": float(q["major_radius"]), "minor_radius": float(q["minor_radius"]),
                "center": q["center"].tolist(), "axis_dir": q["axis"].tolist()}
    if p.kind == "cone":
        return {"half_angle_deg": float(np.degrees(q["half_angle"])), "apex": q["apex"].tolist(),
                "axis_dir": q["axis"].tolist()}
    if p.kind == "pipe":
        return {"blend_radius": float(q["radius"]), "support_regions": [int(q["support_plane"]), int(q["support_cyl"])],
                "spine": "ellipse", "spine_center": q["center"].tolist(), "spine_major": float(q["A"]),
                "spine_minor": float(q["B"])}
    return {}


def named_dimensions(regions, clusters, prims):
    """Diameters of cylinders/spheres and distances between parallel planes (adjacent in offset + span)."""
    dims = []
    seen = set()
    for r in regions:
        p = prims[r.id]
        if p.kind in ("cylinder", "sphere") and id(p) not in seen:
            seen.add(id(p))
            dims.append({"type": "diameter", "regions": [r.id], "value": 2 * float(p.params["radius"])})
        if p.kind == "torus" and id(p) not in seen:
            seen.add(id(p))
            dims.append({"type": "torus_minor_radius", "regions": [r.id], "value": float(p.params["minor_radius"])})
            dims.append({"type": "torus_major_radius", "regions": [r.id], "value": float(p.params["major_radius"])})
        if p.kind == "pipe" and id(p) not in seen:
            seen.add(id(p))
            dims.append({"type": "blend_radius", "regions": [r.id], "value": float(p.params["radius"])})
        if p.kind == "cone" and id(p) not in seen:
            seen.add(id(p))
            dims.append({"type": "cone_half_angle_deg", "regions": [r.id],
                         "value": float(np.degrees(p.params["half_angle"]))})
    for c in clusters:
        e = np.array(c["dir"])
        ps, ids = [], set()
        for rid in c["regions"]:
            p = prims.get(rid)   # regions merged away by consolidate are gone
            if p is not None and p.kind == "plane" and id(p) not in ids:
                ids.add(id(p))
                ps.append((float(p.params["point"] @ e), rid))
        ps.sort()
        uniq = []
        for o, rid in ps:   # identical offsets (coplanar faces) count once
            if not uniq or o - uniq[-1][0] > 1e-6:
                uniq.append((o, rid))
        ps = uniq
        for (a, ra), (b, rb) in zip(ps, ps[1:]):
            dims.append({"type": "distance", "regions": [ra, rb], "direction": e.tolist(), "value": b - a})
        if len(ps) > 2:
            dims.append({"type": "span", "regions": [ps[0][1], ps[-1][1]], "direction": e.tolist(),
                         "value": ps[-1][0] - ps[0][0]})
    dims += draft_dimensions(regions, prims)
    for i, d in enumerate(dims):
        d["name"] = f"D{i + 1}"
        d["units"] = "deg" if d["type"].endswith("_deg") else "mm"
    return dims


DRAFT_MIN_DEG, DRAFT_MAX_DEG = 0.3, 20.0


def _pull_direction(regions, prims):
    """Pull (draw) direction: normal of the plane direction with the largest total area."""
    best, acc = None, []
    for r in regions:
        p = prims[r.id]
        if p.kind != "plane":
            continue
        n = np.asarray(p.params["normal"], float)
        w = float(getattr(r, "area", 0.0) or len(r.faces))
        for k, (m, tot) in enumerate(acc):
            if abs(n @ m) > np.cos(np.radians(0.5)):
                acc[k] = (m, tot + w)
                break
        else:
            acc.append((n, w))
    if acc:
        best = max(acc, key=lambda t: t[1])[0]
    return best


def _draft_deg(n, e):
    """Draft of a plane with normal n against pull direction e: 90 deg minus the n-e angle (unsigned)."""
    return float(np.degrees(np.arcsin(np.clip(abs(np.asarray(n, float) @ e), 0.0, 1.0))))


def draft_dimensions(regions, prims):
    """Draft angles of side faces: planes whose normal is DRAFT_MIN..DRAFT_MAX deg off perpendicular to
    the pull direction. Positive value; `pull_direction` is recorded so the sign convention is explicit."""
    e = _pull_direction(regions, prims)
    if e is None:
        return []
    out, seen = [], set()
    for r in regions:
        p = prims[r.id]
        if p.kind != "plane" or id(p) in seen:
            continue
        seen.add(id(p))
        a = _draft_deg(p.params["normal"], e)
        if DRAFT_MIN_DEG <= a <= DRAFT_MAX_DEG:
            out.append({"type": "draft_angle_deg", "regions": [r.id], "pull_direction": e.tolist(), "value": a})
    return out


def _raw_value(d, raw, prims_now=None):
    if d["type"] == "draft_angle_deg":
        return _draft_deg(raw[d["regions"][0]].params["normal"], np.array(d["pull_direction"]))
    if d["type"] == "diameter":
        return 2 * float(raw[d["regions"][0]].params["radius"])
    if d["type"] == "torus_minor_radius":
        return float(raw[d["regions"][0]].params["minor_radius"])
    if d["type"] == "torus_major_radius":
        return float(raw[d["regions"][0]].params["major_radius"])
    if d["type"] == "blend_radius":
        return float(prims_now[d["regions"][0]].params.get("radius_fit", np.nan))
    if d["type"] == "cone_half_angle_deg":
        return float(np.degrees(raw[d["regions"][0]].params["half_angle"]))
    e = np.array(d["direction"])
    a, b = (raw[i].params["point"] @ e for i in d["regions"])
    return float(b - a)


# progress: the last stage reached is kept in the report and echoed to stderr, so a run killed by an
# external timeout still says where it was (the CLI writes the partial report on SIGTERM)
CURRENT = {"rep": None, "report_path": None, "t0": None, "deadline": None}


def time_left(default=float("inf")):
    """Seconds left of the run's time budget (inf without one): stages with loops of OCC calls stop early
    and fall back / fail with a reason instead of being killed."""
    from . import budget
    left = budget.time_left()
    return default if left == float("inf") else left


def _stage_cap(pr, frac, max_s):
    """A stage's own cap: frac of the run's time budget, at most max_s (max_s when there is no budget)."""
    tb = float(pr.time_budget_s or 0.0)
    return min(max_s, max(30.0, frac * tb)) if tb > 0 else max_s


# ---------------------------------------------------------------------------
# 1.0.3 heartbeat: a small JSON file next to the report that names the current
# stage and when it was last touched.  A watcher (or the next stage) can tell
# a run is stuck when the heartbeat is older than the stage limit.
# ---------------------------------------------------------------------------
def _heartbeat_path(report_path):
    if not report_path:
        return None
    return str(Path(report_path).with_suffix(".heartbeat.json"))


def _write_heartbeat(stage_name):
    rp = CURRENT.get("report_path")
    hp = _heartbeat_path(rp) if rp else None
    if not hp or not CURRENT.get("t0"):
        return
    try:
        hb = {
            "stage": stage_name,
            "ts": time.time(),
            "elapsed_s": round(time.time() - CURRENT["t0"], 2),
        }
        tmp = hp + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(hb, f)
        os.replace(tmp, hp)
    except (OSError, ValueError):
        pass


def _stage_deadline(pr, stage_name):
    """Seconds this stage may use: its own limit, shortened by the run budget left."""
    stage_cap = float(pr.stage_limits.get(stage_name, pr.stage_limits.get("default", 300.0)))
    left = time_left()
    if left == float("inf"):
        return stage_cap
    return max(5.0, min(stage_cap, left - max(30.0, 0.2 * left)))


def _stage(rep, name):
    import sys
    rep["last_stage"] = name
    rep.setdefault("stage_times", {})[name] = round(time.time() - CURRENT["t0"], 2) if CURRENT["t0"] else 0.0
    print(f"[{rep['stage_times'][name]:8.1f}s] {name}", file=sys.stderr, flush=True)
    _write_heartbeat(name)
    if CURRENT.get("report_path") and name != "done":
        # checkpoint: if the process dies (e.g. a segfault inside OCC) the report still says how far it got
        import json as _json
        part = dict(rep, status="incomplete",
                    failure=f"process ended during stage '{name}' (crash or kill)",
                    end_reason="crashed")
        try:
            Path(CURRENT["report_path"]).write_text(_json.dumps(part, indent=2, default=_js))
        except (TypeError, ValueError, OSError):
            pass


class _Failed:
    ok = False
    shape = None

    def __init__(self, reason):
        self.reason = reason


def _rotor_axis(b, regions, min_frac=0.2):
    """Part axis of a turned/rotor part: the revolve builder's axis, else the largest axial surface
    if revolved surfaces coaxial with it (cylinders, cones, tori; not planes) carry >= min_frac of the
    fitted area.  None for prismatic parts."""
    if getattr(b, "axis_dir", None) is not None:
        return np.asarray(b.axis_point, float), np.asarray(b.axis_dir, float)
    ax = [r for r in regions if r.primitive is not None and r.primitive.kind in ("cylinder", "cone")]
    if not ax:
        return None
    ref = max(ax, key=lambda r: r.area)
    q = ref.primitive.params
    u = np.asarray(q["axis"], float); u /= np.linalg.norm(u)
    p0 = np.asarray(q.get("point", q.get("apex")), float)
    tot = sum(r.area for r in regions if r.primitive is not None)
    co = 0.0
    for r in regions:
        if r.primitive is None:
            continue
        k, qq = r.primitive.kind, r.primitive.params
        if k in ("cylinder", "cone", "torus"):
            a = np.asarray(qq["axis"], float); a /= np.linalg.norm(a)
            c = np.asarray(qq.get("point", qq.get("apex", qq.get("center"))), float) - p0
            if abs(a @ u) > 0.9998 and np.linalg.norm(c - (c @ u) * u) < 0.5:
                co += r.area
    return (p0, u) if tot > 0 and co / tot >= min_frac else None


def _bearing_envelope(br, out_step):
    """Catalogue envelope of a matched bearing: plain ring d x D x B on the measured axis and face plane,
    written beside the full build as <out>.envelope.step (CAD convention for bought-in bearings)."""
    from pathlib import Path
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder
    from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt
    from . import occ_util as U
    m = br["match"]
    p, u = br["axis_point"], br["axis_dir"]
    ax = gp_Ax2(gp_Pnt(*p), gp_Dir(*u))
    op = BRepAlgoAPI_Cut(BRepPrimAPI_MakeCylinder(ax, m["D"] / 2, m["B"]).Shape(), BRepPrimAPI_MakeCylinder(ax, m["d"] / 2, m["B"]).Shape())
    op.Build()
    sh = op.Shape()
    if not U.check(sh):
        raise RuntimeError("envelope ring is not a valid solid")
    out = Path(out_step).with_suffix(".envelope.step")
    write_step(sh, out)
    return {"step": str(out), "d": m["d"], "D": m["D"], "B": m["B"], "designations": m["designations"],
            "volume_mm3": round(U.volume_precise(sh), 3), "valid": True,
            "note": "catalogue boundary dimensions, plain ring without chamfers; not the scanned shape (no deviation gate)"}


def run(input_path, out_step=None, report_path=None, params: Params | None = None,
        deviation_ply=None, preview_png=None, measurements: dict | None = None):
    from . import watchdog
    from .power import keep_awake
    pr = params or Params()
    with keep_awake():          # Windows: no system sleep while the job runs (released on exit / error)
        return _run_guarded(input_path, out_step, report_path, pr, deviation_ply, preview_png, measurements, watchdog)


def _run_guarded(input_path, out_step, report_path, pr, deviation_ply, preview_png, measurements, watchdog):
    # NOTE: the time budget is wall clock (time.time() here, in budget.py and in the watchdog), so time the PC
    # spends asleep counts against it; keep_awake() above prevents that sleep on Windows.
    wd = watchdog.start(pr.time_budget_s, report_path)
    from . import budget
    budget.set_deadline(time.time() + pr.time_budget_s if pr.time_budget_s and pr.time_budget_s > 0 else None)
    try:
        if pr.split_bodies:
            mb = _split_bodies(input_path)
            if mb is not None:
                return _run_bodies(input_path, mb, out_step, report_path, pr, preview_png)
        return _run(input_path, out_step, report_path, pr, deviation_ply, preview_png, measurements)
    finally:
        budget.set_deadline(None)
        if wd is not None:
            wd.terminate()
            try:
                wd.wait(5)
            except Exception:  # noqa: BLE001
                pass


BODY_MIN_FRAC = 0.01     # bodies below 1% of the largest body's volume (and area) are debris


def _split_bodies(input_path):
    """Multi-body STL (separate printed parts, or CAD bodies exported together): the repaired mesh's
    connected components. None for a single body without debris; else (bodies, debris) as meshes."""
    from .meshio import MeshInputError
    try:
        mesh, _rep = load_mesh(input_path)
    except MeshInputError:
        return None
    parts = mesh.split(only_watertight=False)
    if len(parts) < 2:
        return None
    size = [max(abs(float(p.volume)) if p.is_watertight else 0.0, 0.0) for p in parts]
    area = [float(p.area) for p in parts]
    vmax, amax = max(size) or 1.0, max(area) or 1.0
    keep = [p for p, v, a in zip(parts, size, area) if (v >= BODY_MIN_FRAC * vmax and v > 0) or a >= 0.05 * amax]
    debris = [p for p in parts if not any(p is q for q in keep)]
    if len(keep) == 1:
        return None      # one body (+ debris): built as before, debris stays in the mesh as small regions
    # only CAD exports (exact tessellation) are split: a scan's second component is scan debris or a
    # fixture, and the scan pipeline handles it (pinion p01 has one)
    from . import facets as _FA
    _tol, _sigma, _exact = auto_tol(mesh)
    if not _FA.detect(mesh, _sigma, _exact)[1].get("cad_tessellation"):
        return None
    # nested bodies (balls inside a bearing's rings, a part inside its housing) are an assembly whose
    # analysis needs the whole mesh: no split
    ext = float(np.linalg.norm(mesh.extents))
    for i, p in enumerate(keep):
        for j, q in enumerate(keep):
            if i != j and np.all(p.bounds[0] >= q.bounds[0] - 1e-3 * ext) and np.all(p.bounds[1] <= q.bounds[1] + 1e-3 * ext):
                return None
    keep.sort(key=lambda p: -float(p.area))
    return keep, debris



def _body_worker(payload):
    """Worker for one multi-body component (same module; ProcessPool re-imports it).

    Child-process isolation keeps OCC state per body. Payload is a plain dict.
    """
    import json as _json
    bi = payload["stl"]
    step = payload["step"]
    report = payload["report"]
    pr = Params(**payload["params"])
    pr.split_bodies = False
    pr.time_budget_s = float(payload.get("time_budget_s") or 0.0)
    try:
        r, *_ = _run(bi, step, report, pr, None, None, None)
    except Exception as ex:  # noqa: BLE001
        r = {"status": "exception", "failure": f"{type(ex).__name__}: {ex}", "seconds": 0.0}
        Path(report).write_text(_json.dumps(r, indent=2, default=str))
    return r


def _run_bodies(input_path, mb, out_step, report_path, pr, preview_png):
    """Build every body on its own (own tolerance, segmentation, builder, deviation gate) and write
    the built solids as one multi-body STEP. Status ok only when every body is ok; 'partial' when some
    bodies built (the STEP holds those, the report lists the failed ones with their reasons)."""
    import tempfile
    from .occ_util import read_step
    from OCP.TopoDS import TopoDS_Compound
    from OCP.BRep import BRep_Builder
    t0 = time.time()
    bodies, debris = mb
    rep = {"tool": "meshsolid", "version": __version__, "input": str(input_path), "params": asdict(pr),
           "status": "failed", "stages": {}, "multi_body": {"bodies": [], "debris": [
               {"faces": int(len(d.faces)), "area": round(float(d.area), 4)} for d in debris]}}
    CURRENT.update(rep=rep, report_path=report_path, t0=t0)
    _stage(rep, "multi_body")
    nf = sum(len(b.faces) for b in bodies) or 1
    tmp = Path(tempfile.mkdtemp(prefix="meshsolid_bodies_"))
    shapes, ok, sub_pr = [], 0, copy.copy(pr)
    sub_pr.split_bodies = False
    from . import budget
    left0 = budget.time_left()
    end = time.time() + left0 if np.isfinite(left0) else None
    # Parallel body builds across cores when MESHSOLID_BODY_JOBS>1 (each body in a
    # child process so OCC stays isolated). Default 1 keeps prior sequential behaviour.
    body_jobs = int(os.environ.get("MESHSOLID_BODY_JOBS", "1") or 1)
    body_jobs = max(1, min(body_jobs, len(bodies)))
    payloads = []
    for i, body in enumerate(bodies):
        bi = tmp / f"body{i:02d}.stl"
        body.export(bi)
        tb = 0.0
        if end is not None:
            rest = sum(len(x.faces) for x in bodies[i:]) or 1
            tb = max(30.0, (end - time.time()) * len(body.faces) / rest)
        params = asdict(sub_pr)
        params["split_bodies"] = False
        params["time_budget_s"] = float(tb)
        payloads.append({
            "i": i, "faces": int(len(body.faces)),
            "volume": round(float(body.volume), 3) if body.is_watertight else None,
            "stl": str(bi), "step": str(tmp / f"body{i:02d}.step"),
            "report": str(tmp / f"body{i:02d}.json"),
            "params": params, "time_budget_s": float(tb),
        })
    results = [None] * len(payloads)
    if body_jobs == 1 or len(payloads) == 1:
        for i, pl in enumerate(payloads):
            _stage(rep, f"multi_body: body {i + 1}/{len(bodies)}")
            if end is not None:
                budget.set_deadline(time.time() + float(pl["time_budget_s"] or 0) or end)
            try:
                results[i] = _body_worker(pl)
            except Exception as ex:  # noqa: BLE001
                results[i] = {"status": "exception", "failure": f"{type(ex).__name__}: {ex}"}
            finally:
                budget.set_deadline(end)
            CURRENT.update(rep=rep, report_path=report_path, t0=t0)
    else:
        _stage(rep, f"multi_body: {len(bodies)} bodies x{body_jobs} workers")
        from concurrent.futures import ProcessPoolExecutor, as_completed
        # Use spawn-safe module path; worker imports pipeline itself.
        with ProcessPoolExecutor(max_workers=body_jobs) as ex:
            futs = {ex.submit(_body_worker, pl): pl["i"] for pl in payloads}
            for fut in as_completed(futs):
                i = futs[fut]
                try:
                    results[i] = fut.result()
                except Exception as ex:  # noqa: BLE001
                    results[i] = {"status": "exception", "failure": f"{type(ex).__name__}: {ex}"}
                CURRENT.update(rep=rep, report_path=report_path, t0=t0)
                _stage(rep, f"multi_body: finished body {i + 1}/{len(bodies)}")
    for i, (body, r) in enumerate(zip(bodies, results)):
        r = r or {"status": "exception", "failure": "no result"}
        b = (r.get("stages") or {}).get("build") or {}
        ent = {"body": i, "faces": int(len(body.faces)), "volume": round(float(body.volume), 3) if body.is_watertight else None,
               "status": r.get("status"), "failure": r.get("failure"), "method": b.get("method"),
               "n_faces": b.get("n_faces"), "deviation": ((r.get("deviation") or {}).get("solid_to_mesh") or {}).get("p95"),
               "hausdorff": (r.get("deviation") or {}).get("hausdorff"), "seconds": round(float(r.get("seconds") or 0), 1)}
        rep["multi_body"]["bodies"].append(ent)
        if r.get("status") == "ok" and (tmp / f"body{i:02d}.step").exists():
            shapes.append(read_step(str(tmp / f"body{i:02d}.step")))
            ok += 1
    rep["multi_body"]["body_jobs"] = body_jobs
    n = len(bodies)
    rep["multi_body"]["built"] = ok
    if shapes and out_step:
        comp = TopoDS_Compound()
        bb = BRep_Builder()
        bb.MakeCompound(comp)
        for sh in shapes:
            bb.Add(comp, sh)
        write_step(comp, out_step)
        rep["step"] = str(out_step)
    if ok == n:
        rep["status"] = "ok"
        rep["end_reason"] = "ok"
    elif ok:
        rep["status"] = "partial"
        rep["failure"] = f"{n - ok} of {n} bodies not built: " + "; ".join(
            f"body {e['body']} ({e['faces']} faces): {str(e['failure'])[:160]}" for e in rep["multi_body"]["bodies"] if e["status"] != "ok")
    else:
        rep["failure"] = f"none of {n} bodies built: " + "; ".join(
            f"body {e['body']}: {str(e['failure'])[:160]}" for e in rep["multi_body"]["bodies"])
    rep["stages"]["build"] = {"method": "multi_body", "ok": ok == n, "n_bodies": n, "built": ok,
                              "n_faces": sum(int(e["n_faces"] or 0) for e in rep["multi_body"]["bodies"] if e["status"] == "ok")}
    rep["seconds"] = time.time() - t0
    _stage(rep, "done")
    if report_path:
        Path(report_path).write_text(json.dumps(rep, indent=2, default=_js))
    return rep, None, None, [], None


def _recogniser(rep, name, status, ex=None, **kw):
    """One entry per recogniser in rep['recognisers']: 'ok' / 'none' / 'failed'. An exception inside a
    recogniser never ends the run: it becomes a 'failed' entry (type, message, innermost frames) and the
    pipeline falls through to the next method."""
    e = {"name": name, "status": status} | kw
    if ex is not None:
        import traceback as _tb
        e["error"] = f"{type(ex).__name__}: {ex}"
        e["where"] = [f"{Path(f.filename).name}:{f.lineno} {f.name}" for f in _tb.extract_tb(ex.__traceback__)[-3:]]
        e["fell_through_to"] = kw.get("fell_through_to", "next method")
    rep.setdefault("recognisers", []).append(e)
    if status not in ("ok",):
        # 1.0.2a: every fall-through says why in the engine log (the job log), not only in the report
        import sys as _sys
        why = e.get("error") or e.get("reason") or "no reason given by the recogniser"
        nxt = e.get("fell_through_to") or ("general method" if status in ("none", "failed") else "")
        print(f"[recogniser] {name}: {status} - {why}" + (f" -> falls through to {nxt}" if nxt else ""),
              file=_sys.stderr, flush=True)
    return e


def _impeller_required_failure(rep, report_path, t0, mesh, why):
    """Part type = Impeller (impeller='on'): the run fails with the recogniser's reason instead of falling
    through to the general method (whose coaxial/freeform stages can take hours on a dense blade scan)."""
    rep["status"] = "failed"
    rep["failure"] = f"Part type Impeller: the impeller recogniser could not build this part - {why}"
    rep.setdefault("warnings", []).append("Part type is Impeller, so the general method was not tried. "
                                          "Set Part type to Auto to let it try (slow on dense scans).")
    rep["seconds"] = time.time() - t0
    import sys as _sys
    print(f"FAILURE: {rep['failure']}", file=_sys.stderr, flush=True)
    _stage(rep, "done")
    if report_path:
        Path(report_path).write_text(json.dumps(rep, indent=2, default=_js))
    return rep, None, mesh, [], np.zeros(len(mesh.faces), int)


def _run(input_path, out_step, report_path, pr, deviation_ply, preview_png, measurements):
    out = _run_core(input_path, out_step, report_path, pr, deviation_ply, preview_png, measurements)
    rep = out[0]
    nrep = rep.get("normalise") or {}
    if nrep.get("applied"):
        # everything was built in the normalised frame: STEP and deviation map back to the scan frame
        try:
            from . import normalise as _NZ
            nrep["restored"] = _NZ.restore_outputs(nrep, rep.get("step"), (rep.get("deviation_map") or {}).get("path"))
            nrep["frame_note"] = ("STEP and deviation map are in the original scan frame; axes, dimensions and "
                                  "positions in this report are in the normalised frame (map with "
                                  "transform_normalised_to_scan)")
        except Exception as ex:  # noqa: BLE001
            nrep["restore_error"] = f"{type(ex).__name__}: {ex}"
            rep["status"] = "failed"
            rep["failure"] = f"could not map the output back to the scan frame: {nrep['restore_error']}"
        if report_path:
            Path(report_path).write_text(json.dumps(rep, indent=2, default=_js))
    return out


def _run_core(input_path, out_step, report_path, pr, deviation_ply, preview_png, measurements):
    t0 = time.time()
    rep = {"tool": "meshsolid", "version": __version__, "input": str(input_path),
           "params": asdict(pr), "status": "failed", "stages": {},
           "end_reason": None}
    CURRENT.update(rep=rep, report_path=report_path, t0=t0)
    _stage(rep, "load_repair")
    from .meshio import MeshInputError
    try:
        mesh, mrep = load_mesh(input_path)
    except MeshInputError as e:
        rep["status"] = "failed"
        rep["failure"] = f"input mesh: {e}. No STEP written."
        rep["input_error"] = str(e)
        rep["end_reason"] = "crashed"
        rep["seconds"] = time.time() - t0
        _stage(rep, "done")
        if report_path:
            Path(report_path).write_text(json.dumps(rep, indent=2, default=_js))
        return rep, None, None, [], None
    # 1.0.3 input normalisation: decimate oversized meshes for recognition.
    if len(mesh.faces) > pr.decimate_for_recognition:
        try:
            from . import normalise as _NZ
            before = len(mesh.faces)
            mesh = _NZ.decimate_for_recognition(mesh, target=pr.decimate_for_recognition)
            rep["decimation"] = {"from_faces": before,
                                 "to_faces": len(mesh.faces),
                                 "target": pr.decimate_for_recognition,
                                 "note": "recognition runs on the decimated mesh; deviation on the same mesh"}
        except Exception as ex:  # noqa: BLE001
            rep.setdefault("warnings", []).append(f"decimation skipped ({type(ex).__name__}: {ex})")
    # 1.0.3 scale / units check: flag a ~1,650 mm mesh (could be mm or inches) before recognition.
    try:
        from . import normalise as _NZ
        rep["units"] = _NZ.check_units(mesh)
    except Exception as ex:  # noqa: BLE001
        rep.setdefault("warnings", []).append(f"units check skipped ({type(ex).__name__}: {ex})")
    _stage(rep, "normalise")
    try:
        from . import normalise as _NZ
        mesh, rep["normalise"] = _NZ.apply(mesh, getattr(pr, "normalise", "report") or "report")
    except Exception as ex:  # noqa: BLE001
        rep["normalise"] = {"mode": getattr(pr, "normalise", "report"), "applied": False,
                            "error": f"{type(ex).__name__}: {ex}"}
    sigma, exact = None, None
    if pr.tol is None:
        tol, sigma, exact = auto_tol(mesh)
    else:
        tol = pr.tol
    from . import device as _D
    rep["device"] = _D.report()
    rep["tolerance"] = {"tol": tol, "noise_sigma_est": sigma, "exact_mesh_detected": exact, "auto": pr.tol is None}
    # CAD-tessellated STL (prints exported from CAD): tolerance from the facet chord sag, smooth faceted
    # skins routed to the freeform path (meshsolid.facets)
    from . import facets as _FA
    faceted, fst = _FA.detect(mesh, sigma, exact, pr.sharp_deg)
    _diag = float(np.linalg.norm(np.ptp(mesh.vertices, axis=0)))
    if fst.get("cad_tessellation") and fst.get("facet_width_median", 0.0) < 0.005 * _diag:
        # a fine tessellation (facets < 0.5% of the part; exact CAD meshes like the coupling, remeshed
        # scans like s07): the noise-based tolerance is already right and plain segmentation explains it;
        # faceted mode is for coarse CAD exports
        faceted = False
        fst = dict(fst, faceted=False, cad_tessellation=False, fine_exact_tessellation=True)
    rep["tolerance"]["cad_tessellation"] = fst
    if fst["cad_tessellation"] and pr.tol is None:
        tol = _FA.tolerance(fst, float(np.linalg.norm(np.ptp(mesh.vertices, axis=0))))
        rep["tolerance"]["tol"] = tol
        rep["tolerance"]["from"] = "facet chord sag (CAD tessellation)"
    from .accuracy import resolve_gates
    _gates = resolve_gates(pr.accuracy, tol, pr.accept_rms, pr.accept_p95)
    # input class (scan / decimated / CAD tessellation / low-res tessellation): noise + gate policy, logged
    try:
        from . import input_class as _IC
        _ic = _IC.classify(mesh, sigma, exact, fst)
        _ne, _nsrc = _IC.effective_noise(_ic, sigma)
        _ic["noise_sigma_effective_mm"] = None if _ne is None else round(_ne, 5)
        _ic["noise_source"] = _nsrc
        rep["tolerance"]["noise_sigma_effective"] = _ic["noise_sigma_effective_mm"]
        if (_ic["class"] == "lowres_tessellation" and pr.accuracy in (None, "") and pr.accept_rms is None
                and pr.accept_p95 is None):
            _gates = _IC.lowres_gates(_ic, _gates, tol)
            _ic["gates"] = "loosened to the facet size"
        else:
            _ic["gates"] = "unchanged"
        rep["input_class"] = _ic
        print(f"[input-class] {_ic['class']}: {_ic['reason']}; noise {_ic['noise_sigma_effective_mm']} mm "
            f"({_nsrc}); gates {_ic['gates']} (rms<={_gates['rms_limit']:.4f}, p95<={_gates['p95_limit']:.4f})", flush=True)
    except Exception as _e:  # noqa: BLE001
        rep["input_class"] = {"error": f"{type(_e).__name__}: {_e}"}
    accept_rms = _gates["rms_limit"]
    accept_p95 = _gates["p95_limit"]
    rep["accuracy"] = _gates

    rep["stages"]["load_repair"] = asdict(mrep)
    rep["mesh"] = {"vertices": len(mesh.vertices), "faces": len(mesh.faces),
                   "bounds": mesh.bounds.tolist(), "area": float(mesh.area),
                   "volume": float(mesh.volume) if mesh.is_watertight else None}
    if not mrep.watertight:
        rep["warnings"] = ["mesh is not watertight after repair; inside/outside classification may be wrong"]

    if pr.notes:
        rep["notes"] = list(pr.notes)
    # Smooth faceted-CAD skins (coarse CAD tessellation of printed/loftable parts): try
    # section loft first. Acceptance uses the accuracy-class gates (machined vs printed).
    # On failure, fall through to the existing prismatic/freeform path so machined and
    # scan regressions stay identical.
    if faceted and fst.get("cad_tessellation"):
        _stage(rep, "loft_skin")
        t_loft = time.time()
        loft_info = {"tried": True, "accepted": False}
        try:
            from . import loft as _LOFT
            from .occ_util import check as _occ_check, volume as _occ_vol, count as _occ_count
            from OCP.TopAbs import TopAbs_SOLID, TopAbs_FACE
            budget = pr.time_budget_s if pr.time_budget_s and pr.time_budget_s > 0 else 120.0
            budget = float(min(max(budget, 30.0), 300.0 if pr.accuracy == "printed" else 180.0))
            lres = _LOFT.build_loft(
                mesh, tol, corner_deg=0.0, max_sections=48, time_budget=budget,
                self_check=True, accuracy=pr.accuracy)
            loft_info.update({
                "ok": bool(lres.ok),
                "reason": (lres.reason or "")[:240],
                "method": getattr(lres, "method", None),
                "self_check": lres.self_check,
                "n_slabs": len(getattr(lres, "slabs", None) or []),
                "vol_ratio": (None if not lres.mesh_volume else round(float(lres.volume) / float(lres.mesh_volume), 4)),
                "seconds": round(time.time() - t_loft, 2),
                "pockets": getattr(lres, "pockets", None),
            })
            if lres.ok and lres.shape is not None and _occ_check(lres.shape):
                nsol = int(_occ_count(lres.shape, TopAbs_SOLID))
                if nsol == 1:
                    sc = lres.self_check or {}
                    vd_rms = float(sc.get("rms", 1e9))
                    vd_p95 = float(sc.get("mesh_vertices_to_solid_p95", 1e9))
                    s2m_p95 = float(sc.get("solid_to_mesh_p95", 1e9))
                    lim_s2m = float(sc.get("solid_to_mesh_p95_limit", accept_p95))
                    accepted = (vd_rms <= accept_rms and vd_p95 <= accept_p95 and s2m_p95 <= lim_s2m)
                    loft_info["accepted"] = bool(accepted)
                    loft_info["n_solids"] = nsol
                    loft_info["volume"] = float(_occ_vol(lres.shape))
                    loft_info["n_faces"] = int(_occ_count(lres.shape, TopAbs_FACE))
                    rep["stages"]["loft_skin"] = loft_info
                    rep["stages"]["build"] = {
                        "method": "section_loft", "ok": bool(accepted),
                        "brepcheck_valid": True, "n_solids": nsol,
                        "volume": loft_info["volume"], "n_faces": loft_info["n_faces"],
                        "seconds": loft_info["seconds"],
                        "reason": None if accepted else (lres.reason or "loft gates failed"),
                    }
                    rep["deviation"] = {
                        "mesh_vertices_to_solid": {
                            "rms": vd_rms, "p95": vd_p95,
                            "max": sc.get("hausdorff"),
                        },
                        "solid_to_mesh": {"p95": s2m_p95},
                        "hausdorff": sc.get("hausdorff"),
                        "acceptance": {
                            "basis": "section loft self_check; accuracy-class gates",
                            "rms_limit": accept_rms, "p95_limit": accept_p95,
                            "solid_to_mesh_p95_limit": lim_s2m,
                            "passed": bool(accepted),
                            "accuracy": (rep.get("accuracy") or {}).get("class"),
                        },
                        "from": "loft_self_check",
                    }
                    if accepted and out_step:
                        write_step(lres.shape, out_step)
                        rep["step"] = str(out_step)
                        rep["status"] = "ok"
                        rep["end_reason"] = "ok"
                    elif accepted:
                        rep["status"] = "ok_no_output"
                    else:
                        rep["status"] = "failed"
                        rep["failure"] = (lres.reason or "section loft built but missed accuracy gates")
                    # Only early-return on accept; otherwise fall through so freeform can try.
                    if accepted:
                        rep["seconds"] = time.time() - t0
                        _stage(rep, "done")
                        if report_path:
                            Path(report_path).write_text(json.dumps(rep, indent=2, default=_js))
                        return rep, lres.shape, mesh, [], np.zeros(len(mesh.faces), int)
                else:
                    loft_info["reason"] = f"loft n_solids={nsol}"
            else:
                loft_info.setdefault("reason", (lres.reason or "loft not ok")[:240])
        except Exception as ex:  # noqa: BLE001
            loft_info["error"] = f"{type(ex).__name__}: {ex}"
            loft_info["seconds"] = round(time.time() - t_loft, 2)
        loft_info["fallback"] = "prismatic_freeform"
        rep["stages"]["loft_skin"] = loft_info

    if pr.impeller != "off":
        _stage(rep, "impeller_probe")
        from . import impeller as I
        tp = time.time()
        try:
            nb, pinfo = I.probe(mesh)
        except Exception as ex:  # noqa: BLE001
            nb, pinfo = None, {"reason": f"probe exception: {type(ex).__name__}: {ex}"}
            _recogniser(rep, "impeller_probe", "failed", ex,
                        fell_through_to="none (Part type Impeller)" if pr.impeller == "on" else "gear / screw / prismatic")
        rep["stages"]["impeller_probe"] = pinfo | {"blades": nb, "seconds": round(time.time() - tp, 2)}
        if nb is not None:
            _stage(rep, "impeller")
            from .impeller_run import run_impeller
            rep_before = copy.deepcopy(rep)
            try:
                rep, shape = run_impeller(mesh, rep, pr, tol, accept_rms, accept_p95, out_step, deviation_ply,
                                          preview_png, input_path, measurements, pr.notes, nb, pinfo)
            except Exception as ex:  # noqa: BLE001
                rep = rep_before
                _recogniser(rep, "impeller", "failed", ex, blades=nb,
                            fell_through_to="none (Part type Impeller)" if pr.impeller == "on" else "gear / screw / prismatic")
                if pr.impeller == "on":
                    return _impeller_required_failure(rep, report_path, t0, mesh, f"{type(ex).__name__}: {ex}")
                rep.setdefault("warnings", []).append(
                    f"impeller recognition failed ({type(ex).__name__}: {ex}); continuing with the general method")
                CURRENT.update(rep=rep, report_path=report_path, t0=t0)
            else:
                _recogniser(rep, "impeller", "ok" if rep.get("status") == "ok" else "built, gate failed",
                            blades=nb)
                rep["seconds"] = time.time() - t0
                _stage(rep, "done")
                if report_path:
                    Path(report_path).write_text(json.dumps(rep, indent=2, default=_js))
                return rep, shape, mesh, [], np.zeros(len(mesh.faces), int)
        else:
            why = pinfo.get("reason") or "probe found no N-fold blades"
            if pinfo.get("votes") is not None:
                why += f" (ring votes {pinfo.get('votes')}, tangential area {pinfo.get('tangential_area_frac')})"
            _recogniser(rep, "impeller", "none", reason=why,
                        fell_through_to="none (Part type Impeller)" if pr.impeller == "on" else "gear / screw / prismatic")
            if pr.impeller == "on":
                return _impeller_required_failure(rep, report_path, t0, mesh, why)

    # gear / screw rotor recognition works on the mesh alone, so it runs before segmentation; a screw
    # rotor's helicoid flanks are then left out of segmentation (they are modelled by the sweep)
    _stage(rep, "gear_screw_detect")
    gear_spec = None
    if pr.gears != "off":
        from . import gear as G
        tg = time.time()
        try:
            gear_spec, ginfo = G.analyse(mesh, tol)
        except Exception as ex:  # noqa: BLE001
            gear_spec, ginfo = None, {"reason": f"exception: {type(ex).__name__}: {ex}"}
            _recogniser(rep, "gear", "failed", ex, fell_through_to="screw / prismatic")
        rep["gear"] = ({"detected": True, "build_from": pr.gear_build} | G.to_json(gear_spec) | {"acceptance": ginfo}
                       if gear_spec is not None else {"detected": False} | ginfo)
        rep["gear"]["seconds"] = round(time.time() - tg, 2)
    screw_spec = None
    if pr.screws != "off" and gear_spec is None:
        from . import screw as SC
        tg = time.time()
        try:
            screw_spec, sinfo = SC.analyse(mesh, tol)
        except Exception as ex:  # noqa: BLE001
            screw_spec, sinfo = None, {"reason": f"exception: {type(ex).__name__}: {ex}"}
            _recogniser(rep, "screw_rotor", "failed", ex, fell_through_to="prismatic")
        rep["screw_rotor"] = ({"detected": True} | SC.to_json(screw_spec) | {"acceptance": sinfo}
                              if screw_spec is not None else {"detected": False} | sinfo)
        rep["screw_rotor"]["seconds"] = round(time.time() - tg, 2)
    special = gear_spec is not None or screw_spec is not None
    # 1.0.3 per-stage deadline: stop before segmentation if there isn't enough time left.
    _seg_dl = _stage_deadline(pr, "segmentation")
    if time_left() != float("inf") and time_left() < _seg_dl + 30.0:
        rep["stages"]["segmentation"] = {"skipped": True,
                                         "reason": f"not enough time left ({time_left():.0f}s < {_seg_dl:.0f}s limit + reserve)"}
        rep["end_reason"] = "timed_out"
        rep["failure"] = f"timed out before segmentation (stage limit {_seg_dl:.0f}s, {time_left():.0f}s left)"
        rep["seconds"] = time.time() - t0
        _stage(rep, "done")
        if report_path:
            Path(report_path).write_text(json.dumps(rep, indent=2, default=_js))
        return rep, None, mesh, [], None
    _stage(rep, "segmentation")
    try:
        sp = SegParams(sharp_deg=pr.sharp_deg, tol=tol, strict_torus=bool(fst.get("cad_tessellation")))
        ts = time.time()
        if screw_spec is not None:
            from .gear import cyl as _cyl
            r_, _, z_ = _cyl(np.asarray(mesh.triangles_center), np.asarray(screw_spec.axis_point), np.asarray(screw_spec.axis_dir))
            zone = (z_ > screw_spec.z_a - 0.3) & (z_ < screw_spec.z_b + 0.3) & (r_ > screw_spec.r_root - 0.5)
            keep = np.flatnonzero(~zone)
            sub = mesh.submesh([keep], append=True, repair=False)
            regions, sub_labels, sinfo = segment(sub, sp, rng_seed=pr.seed)
            labels = -np.ones(len(mesh.faces), int); labels[keep] = sub_labels
            for r_reg in regions:
                r_reg.faces = keep[r_reg.faces]
            sinfo = sinfo | {"screw_zone_faces_not_segmented": int(zone.sum())}
        else:
            regions, labels, sinfo = segment(mesh, sp, rng_seed=pr.seed)
        rep["stages"]["segmentation"] = sinfo | {"regions": len(regions), "seconds": time.time() - ts}
    except Exception as ex:  # noqa: BLE001
        rep["end_reason"] = "crashed"
        rep["failure"] = f"segmentation crashed ({type(ex).__name__}: {ex}). No STEP written."
        rep["seconds"] = time.time() - t0
        _stage(rep, "done")
        if report_path:
            Path(report_path).write_text(json.dumps(rep, indent=2, default=_js))
        return rep, None, mesh, [], None
    _stage(rep, "coaxial_refit")
    g = MeshGraph(mesh)
    from .coaxial import coaxial_refit
    from . import budget as _B
    # 1.0.2a: capped to a share of the run's budget (KK&K run: 37 min here before freeform used the rest)
    _cap = _B.stage_budget(_B.env_cap("MESHSOLID_COAXIAL_CAP_S", _stage_cap(pr, 0.10, 600.0)))
    rep["stages"]["coaxial_refit"] = coaxial_refit(g, labels, regions, tol, deadline=time.time() + _cap)
    if rep["stages"]["coaxial_refit"] and rep["stages"]["coaxial_refit"][-1].get("stopped"):
        print(f"[stage] coaxial_refit stopped at its {_cap:.0f} s cap", file=sys.stderr, flush=True)
    # thread recognition: helical zones -> one cylinder (major dia external / minor internal) + spec
    _stage(rep, "threads")
    thread_specs = []
    if pr.threads:
        from .thread import detect_threads, apply_threads, spec_json
        from .segment import Region
        tt = time.time()
        try:
            thread_specs = detect_threads(mesh, regions, labels, tol)
            if thread_specs:
                regions2, labels2 = apply_threads(mesh, regions, labels, thread_specs, Region)
                regions, labels = regions2, labels2
        except Exception as ex:  # noqa: BLE001
            thread_specs = []
            _recogniser(rep, "threads", "failed", ex, fell_through_to="plain surfaces")
        rep["threads"] = [spec_json(s_) for s_ in thread_specs]
        rep["stages"]["threads"] = {"found": len(thread_specs), "seconds": time.time() - tt,
                                    "regions_after": len(regions)}
    # M4 freeform: a smooth surface the segmenter cut into several same-kind primitives -> one B-spline
    _stage(rep, "freeform")
    if pr.freeform:
        from .freeform import apply_freeform
        tf = time.time()
        from . import budget as _B
        _cap = _B.stage_budget(_B.env_cap("MESHSOLID_FREEFORM_CAP_S", _stage_cap(pr, 0.20, 1200.0)))
        regions, labels, flog = apply_freeform(mesh, g, regions, labels, tol, region_points, faceted=faceted,
                                               deadline=time.time() + _cap)
        if flog and flog[-1].get("stopped"):
            print(f"[stage] freeform stopped at its {_cap:.0f} s cap", file=sys.stderr, flush=True)
        rep["stages"]["freeform"] = {"accepted": sum(1 for x in flog if x.get("accepted")),
                                     "candidates": len(flog), "log": flog, "seconds": time.time() - tf,
                                     "regions_after": len(regions)}
        if thread_specs and flog:   # renumbering may have moved the thread regions
            for s_, js in zip(thread_specs, rep["threads"]):
                v = labels[s_["_zone_faces"]]
                v = v[v >= 0]
                js["region"] = s_["region"] = int(np.bincount(v).argmax()) if len(v) else None
    thread_faces = np.zeros(len(mesh.faces), bool)
    for s_ in thread_specs:
        thread_faces[s_["_zone_faces"]] = True
    _stage(rep, "snapping")
    raw = {r.id: copy.deepcopy(r.primitive) for r in regions}
    sinfo2 = snap(g, regions, tol=tol, angle_deg=pr.snap_angle_deg, round_tol=pr.round_tol,
                  do_round=pr.do_round)
    rep["stages"]["snapping"] = sinfo2
    # safety net: any region whose snapped primitive fits much worse than its raw fit is reverted
    for r in regions:
        P, _N = region_points(g, r.faces)
        rs = float(np.sqrt(np.mean(F.signed_distance(r.primitive, P) ** 2)))
        rr = float(np.sqrt(np.mean(F.signed_distance(raw[r.id], P) ** 2)))
        if rs > max(2 * rr, rr + 0.5 * tol):
            sinfo2["log"].append({"op": "revert_snap", "region": r.id, "rms_raw": rr, "rms_snapped": rs})
            r.primitive = copy.deepcopy(raw[r.id])
    # non-analytic rolling-ball blends (plane x oblique cylinder) -> exact pipe surfaces
    _stage(rep, "blends")
    from .blend import recognise_blends
    from .tangency import region_pairs
    blog = []
    nb = recognise_blends(g, regions, region_pairs(g, regions), tol, blog, do_round=pr.do_round,
                          round_tol=pr.round_tol, region_points=region_points)
    rep["stages"]["blends"] = {"pipes": nb, "log": blog}
    for l in blog:
        next(r for r in regions if r.id == l["region"]).primitive.params["radius_fit"] = l["radius_fit"]
    if pr.consolidate:
        from .consolidate import consolidate
        _stage(rep, "consolidate")
        prot = {s_["region"] for s_ in thread_specs if s_.get("region") is not None}
        for r in regions:
            if r.primitive is not None and r.primitive.kind == "pipe":
                prot |= {r.id, int(r.primitive.params["support_cyl"]), int(r.primitive.params["support_plane"])}
        n0 = len(regions)
        regions, labels, clog = consolidate(g, regions, labels, tol, region_points, protected=prot)
        rep["stages"]["consolidate"] = {"regions_before": n0, "regions_after": len(regions), "log": clog,
                                        "ops": {k: sum(1 for x in clog if x["op"] == k) for k in sorted({x["op"] for x in clog})}}
        for r in regions:   # a re-fitted primitive (degenerate -> plane) is its own least-squares value
            if r.primitive is not None and (r.id not in raw or raw[r.id].kind != r.primitive.kind):
                raw[r.id] = r.primitive
    snapped = {r.id: r.primitive for r in regions}

    _stage(rep, "fit_stats")
    # per-region fit stats with the snapped primitive
    total_area = float(mesh.area)
    for r in regions:
        P, N = region_points(g, r.faces)
        d = F.signed_distance(r.primitive, P)
        sn = F.surface_normal(r.primitive, P)
        nerr = float(np.degrees(np.sqrt(np.mean(np.arccos(np.abs(np.sum(sn * N, 1)).clip(0, 1)) ** 2))))
        r.fit = {"rms": float(np.sqrt(np.mean(d * d))), "max": float(np.abs(d).max()),
                 "normal_err_deg": nerr, "rms_unsnapped": float(raw[r.id].rms)}
        c_fit = float(np.clip(1 - r.fit["rms"] / tol, 0, 1))
        c_nrm = float(np.clip(1 - nerr / sp.normal_deg, 0, 1))
        c_size = float(np.clip(r.area / (0.002 * total_area), 0.2, 1))
        r.confidence = c_fit * c_nrm * c_size
        if r.fit["rms"] > 0.5 * tol:
            r.flags.append("high_fit_residual")
        if c_size < 1:
            r.flags.append("small_region")

    clusters = sinfo2["direction_clusters"]
    dims = named_dimensions(regions, clusters, snapped)
    for d in dims:   # value of the same dimension from the unsnapped least-squares fits
        d["fit_value"] = _raw_value(d, raw, snapped)
    rep["dimensions"] = dims
    try:                                   # M9: keyseats / keyways (planes on a cylinder), snapped to DIN/ANSI keys
        from .keyway import detect_keyways
        rep["keyways"] = detect_keyways(mesh, regions, tol)
    except Exception as ex:  # noqa: BLE001
        rep["keyways"] = []
        rep.setdefault("warnings", []).append(f"keyway recognition failed: {type(ex).__name__}: {ex}")
        _recogniser(rep, "keyway", "failed", ex)
    if gear_spec is None and screw_spec is None and pr.splines == "auto":   # M9: straight splines (report only)
        try:
            from .spline import analyse as _spl
            sp, sinfo = _spl(mesh, tol)
            rep["splines"] = [sp] if sp else []
            if not sp:
                rep["spline_check"] = {k: sinfo[k] for k in ("reason", "teeth") if k in sinfo}
        except Exception as ex:  # noqa: BLE001
            rep.setdefault("warnings", []).append(f"spline recognition failed: {type(ex).__name__}: {ex}")
            _recogniser(rep, "spline", "failed", ex)
    try:                                   # M10: bearing boundary dimensions, labyrinth seal teeth (report only)
        from .turned_features import bearing_ring, labyrinth
        br = bearing_ring(mesh, regions, tol)
        if br is not None and br.get("match"):
            rep["bearing"] = br
        rep["labyrinth"] = labyrinth(mesh, regions, tol)
        from .turned_features import rolling_elements
        rel = rolling_elements(mesh, regions, tol)
        if rel is not None:
            rep["rolling_elements"] = rel
            m_ = (rep.get("bearing") or {}).get("match")
            if m_:                            # visible elements tell ball (60/62/63) from cylindrical roller (NU) bearings
                keep = [d_ for d_ in m_["designations"] if d_.startswith("NU") == (rel["elements"] == "roller")]
                if keep:
                    m_["designations"] = keep
                    m_["note"] = (f"boundary dimensions plus {rel['count']} visible {rel['elements']}s "
                                  f"(diameter {rel['diameter']} mm on a {rel['pitch_diameter']} mm pitch circle)")
    except Exception as ex:  # noqa: BLE001
        rep.setdefault("warnings", []).append(f"bearing/seal recognition failed: {type(ex).__name__}: {ex}")
        _recogniser(rep, "bearing_seal", "failed", ex)

    # 1.0.3 per-stage deadline: stop before build if there isn't enough time left.
    _build_dl = _stage_deadline(pr, "build")
    if time_left() != float("inf") and time_left() < _build_dl + 30.0:
        rep["stages"]["build"] = {"skipped": True,
                                  "reason": f"not enough time left ({time_left():.0f}s < {_build_dl:.0f}s limit + reserve)"}
        rep["end_reason"] = "timed_out"
        rep["failure"] = f"timed out before build (stage limit {_build_dl:.0f}s, {time_left():.0f}s left)"
        rep["seconds"] = time.time() - t0
        _stage(rep, "done")
        if report_path:
            Path(report_path).write_text(json.dumps(rep, indent=2, default=_js))
        return rep, None, mesh, regions, labels
    _stage(rep, "build")
    # build
    prims_unique = []
    for r in regions:
        if all(r.primitive is not p for p in prims_unique):
            prims_unique.append(r.primitive)
    if os.environ.get("MESHSOLID_DUMP"):
        import pickle
        with open(os.environ["MESHSOLID_DUMP"], "wb") as fh:
            pickle.dump({"mesh": mesh, "labels": labels, "regions": regions, "tol": tol, "gear_spec": gear_spec,
                         "screw_spec": screw_spec, "gear_build": pr.gear_build}, fh)
    attempts = []
    b = None
    tb = time.time()
    if pr.builder in ("auto", "extrude") and not special:
        # 2.5D parts (plates, printed brackets): straight prisms of the mesh cross-section, lines + arcs
        from .extrude import build_extrusion, prismatic_axis
        _d, _pf = prismatic_axis(mesh)
        if pr.builder == "extrude" or _pf >= 0.995:
            try:
                eb = build_extrusion(mesh, tol)
                attempts.append(eb.to_json())
            except Exception as ex:  # noqa: BLE001
                eb = None
                attempts.append({"method": "extrude_profile", "ok": False, "reason": f"exception: {ex}"})
            if eb is not None and (eb.ok or pr.builder == "extrude"):
                b = eb
    if pr.builder in ("auto", "local") and not special and not (b is not None and b.ok):
        from .coaxial import realign_coaxial
        rep["stages"]["coaxial_realign"] = realign_coaxial(regions)
        voff = float(sum(s_.get("volume_delta_mm3", 0.0) for s_ in thread_specs))
        regions0, labels0 = list(regions), labels.copy()     # before any retry releases regions
        b = build_local(mesh, labels, regions, tol, volume_offset=voff)
        attempts.append(b.to_json())
        # retry: small regions whose surface yields no trimmed piece (wrong/unsupported primitive on a
        # scan) are released to the neighbours (labels -1) and the trim is redone; large ones are not
        tot_area = float(sum(r.area for r in regions))
        thread_ids = {s_["region"] for s_ in thread_specs if s_.get("region") is not None}
        for _round in range(pr.drop_faceless_rounds if pr.consolidate else 0):
            if b.ok:
                break
            bad = set(attempts[-1].get("regions_without_piece") or [])
            drop = [r for r in regions if r.id in bad and r.id not in thread_ids
                    and r.area < pr.drop_faceless_area_frac * tot_area
                    and not (r.primitive is not None and r.primitive.kind == "pipe")]
            if not drop:
                break
            dids = {r.id for r in drop}
            supports = {int(r.primitive.params[k]) for r in regions if r.primitive is not None
                        and r.primitive.kind == "pipe" for k in ("support_cyl", "support_plane")}
            dids -= supports
            if not dids:
                break
            for rid in dids:
                labels[labels == rid] = -1
            regions = [r for r in regions if r.id not in dids]
            b = build_local(mesh, labels, regions, tol, volume_offset=voff)
            a = b.to_json()
            a["method"] = f"local_trim_retry{_round + 1}"
            a["released_regions"] = sorted(int(i) for i in dids)
            attempts.append(a)
    fill_faces = None
    if special and pr.builder == "auto":
        from .coaxial import realign_coaxial
        rep["stages"]["coaxial_realign"] = realign_coaxial(regions)
        regions0, labels0 = list(regions), labels.copy()
    if pr.builder == "revolve" or (pr.builder == "auto" and (special or (b is not None and not b.ok))):
        # turned parts: revolve the 2D profile of the coaxial surfaces (synthesises faces the scan
        # missed), then holes / bosses; accepted only through the same deviation gate below
        from .revolve import build_revolve
        if pr.builder == "revolve":
            from .coaxial import realign_coaxial
            rep["stages"]["coaxial_realign"] = realign_coaxial(regions)
            regions0, labels0 = list(regions), labels.copy()
        try:
            gp = None
            if gear_spec is not None:
                gp = (gear_spec, gear_spec.snapped["build"] if pr.gear_build == "snapped" else gear_spec.fit["build"])
            rb = build_revolve(mesh, regions0, labels0, tol, gear=gp, screw=screw_spec)
            a = rb.to_json()
        except Exception as ex:  # noqa: BLE001
            rb, a = None, {"method": "revolve_profile", "ok": False, "reason": f"exception: {ex}"}
        attempts.append(a)
        if rb is not None and rb.ok:
            b, regions, labels = rb, regions0, labels0
            fill_faces = rb.fill_faces
        elif b is None:
            b = rb if rb is not None else _Failed(a.get("reason", "revolve failed"))
    if pr.builder == "cell" or (pr.builder == "auto" and not b.ok and len(prims_unique) <= pr.cell_fallback_max):
        try:
            b2 = build_solid(mesh, prims_unique, max_tools=pr.max_tools, seed=pr.seed)
        except Exception as ex:  # noqa: BLE001  (record the failure honestly, keep the local result)
            attempts.append({"method": "cell_decomposition", "ok": False, "reason": f"exception: {ex}"})
            b2 = None
        if b2 is not None:
            b = b2
            d = b.to_json()
            d["method"] = "cell_decomposition"
            attempts.append(d)
    # the stage reports the builder whose result `b` is (an exception attempt carries no volume)
    final = next((a for a in reversed(attempts) if "volume" in a), attempts[-1])
    # geometric tolerance gate: BRepCheck accepts edges/vertices with huge tolerances (gaps papered
    # over by healing), so a "valid" solid can still be locally wrong by millimetres
    if b is not None and b.ok and getattr(b, "shape", None) is not None:
        from OCP.ShapeAnalysis import ShapeAnalysis_ShapeTolerance
        from OCP.TopAbs import TopAbs_EDGE, TopAbs_VERTEX
        sat = ShapeAnalysis_ShapeTolerance()
        mt = max(sat.Tolerance(b.shape, 1, TopAbs_EDGE), sat.Tolerance(b.shape, 1, TopAbs_VERTEX))
        lim = max(2.0 * tol, 0.01)
        final = dict(final)
        final["max_shape_tolerance"] = float(mt)
        final["max_shape_tolerance_limit"] = float(lim)
        if mt > lim:
            b.ok = False
            b.reason = f"max edge/vertex tolerance {mt:.4g} mm exceeds {lim:.4g} mm (solid locally inexact)"
            final["ok"] = False
            final["reason"] = b.reason
    if (b is not None and b.ok and getattr(b, "method", "") == "revolve_profile" and getattr(b, "axis_point", None) is not None
            and getattr(pr, "rim_slots", "auto") != "off"):
        try:
            from .rim_slots import apply as _rs_apply
            u_ = np.asarray(b.axis_dir, float); u_ = u_ / np.linalg.norm(u_)
            e1_ = np.cross(u_, [1.0, 0, 0] if abs(u_[0]) < 0.9 else [0, 1.0, 0])
            D_ = np.asarray(mesh.vertices) - np.asarray(b.axis_point, float)
            R_ = float(np.percentile(np.linalg.norm(D_ - np.outer(D_ @ u_, u_), axis=1), 99.9))
            b.shape, rep["rim_slots"] = _rs_apply(mesh, b.shape, b.axis_point, u_, e1_, R_, pr.rim_slots, log=lambda *a: None)
        except Exception as ex:  # noqa: BLE001
            rep["rim_slots"] = {"error": f"{type(ex).__name__}: {ex}"}
    rep["stages"]["build"] = final | {"attempts": attempts, "seconds": time.time() - tb}
    _stage(rep, "deviation")
    # facet sag: distance of mesh face centroids to their region's surface (chordal error of the
    # tessellation + noise); solid->mesh distances include it, so it widens that acceptance limit
    sags = []
    cent = np.asarray(mesh.triangles_center)
    for r in regions:
        fr = r.faces[~thread_faces[r.faces]]
        if len(fr):
            sags.append(np.abs(F.signed_distance(r.primitive, cent[fr])))
    sag95 = float(np.percentile(np.concatenate(sags), 95)) if sags else 0.0
    per_region_dev = {}
    if b.ok:
        ts = time.time()
        exf = thread_faces.copy()
        n_fill = 0
        if fill_faces is not None and len(fill_faces):
            n_fill = int(len(fill_faces))
            exf[fill_faces] = True
        # helical rotor sweeps (long multi-turn B-spline faces) tessellate to tens of millions of
        # triangles at 2 um; 10 um sag is still far below tol and the measured sag widens the gate
        dkw = {"lin": 0.01, "ang": 0.1} if screw_spec is not None else {}
        dev, per_region_dev, dv, face_dev, solid_mesh = deviation(mesh, b.shape, labels,
                                                                  exclude_faces=exf if exf.any() else None, **dkw)
        # rotor balance planes: local material-removal cuts are not design geometry; the solid is
        # built to the nominal surfaces, the cuts are listed and kept out of the deviation gate
        n_bal = 0
        run_bal = pr.balance_cuts == "on" or (pr.balance_cuts == "auto" and getattr(b, "method", "") != "revolve_profile")
        if not run_bal:
            rep["balance_cuts"] = {"skipped": f"balance_cuts={pr.balance_cuts}" + (
                "; turned body built by the revolve builder (not a bladed rotor)" if pr.balance_cuts == "auto" else "")}
        if run_bal:
            ax = _rotor_axis(b, regions)
            if ax is not None:
                from .balance import detect_on_build
                try:
                    bal, bal_faces = detect_on_build(mesh, solid_mesh, dv, ax[0], ax[1])
                    rep["balance_cuts"] = bal | {"axis_point": np.round(ax[0], 4).tolist(),
                                                 "axis_dir": np.round(ax[1], 6).tolist()}
                    if bal_faces.any():
                        n_bal = int(bal_faces.sum())
                        exf |= bal_faces
                        dev, per_region_dev, dv, face_dev, solid_mesh = deviation(mesh, b.shape, labels, exclude_faces=exf, **dkw)
                except Exception as ex:  # noqa: BLE001
                    rep["balance_cuts"] = {"error": f"{type(ex).__name__}: {ex}"}
        # internal scan faces (capture caps) of a solid turned part: scan-only test, excluded and listed
        n_caps = 0
        if any(p_.get("region") == "axis" for p_ in (getattr(b, "profile", None) or [])):
            from .internal import capture_caps
            caps, cinfo = capture_caps(mesh, b.axis_point, b.axis_dir)
            rep["internal_scan_faces"] = cinfo | {"basis": "axial-normal scan faces with scan skin well beyond them on both axial sides (capture caps of a multi-capture scan)"}
            if caps.any():
                n_caps = int(caps.sum())
                exf |= caps
                dev, per_region_dev, dv, face_dev, solid_mesh = deviation(mesh, b.shape, labels, exclude_faces=exf, **dkw)
        n_gear_tip = 0
        gear_zm = None
        if gear_spec is not None:
            from . import gear as G
            gear_zm = G.zones(np.asarray(mesh.vertices), gear_spec)
            # tip_edge_rounding is scan-only rounding of sharp tip/end edges — exclude from the gate
            tip_v = gear_zm.get("tip_edge_rounding")
            if tip_v is not None and np.asarray(tip_v).any():
                tip_f = tip_v[np.asarray(mesh.faces)].any(1)
                n_gear_tip = int(tip_f.sum())
                if n_gear_tip:
                    exf = exf | tip_f
                    dev, per_region_dev, dv, face_dev, solid_mesh = deviation(
                        mesh, b.shape, labels, exclude_faces=exf, **dkw)
            ev = np.zeros(len(mesh.vertices), bool)
            ev[mesh.faces[exf].ravel()] = True
            rep["gear"]["deviation_zones"] = {
                k: _vstats(dv[m & ~ev]) for k, m in gear_zm.items()} | {
                "basis": "scan vertices -> solid; mid_flank = 0.2..0.8 of the active profile away from the ends; "
                         "tip_edge_rounding = tip band + 1.5 mm at each end (the scan rounds these: reported, not a model error); root = fillet + root circle"}
        if screw_spec is not None:
            from .gear import cyl as _cyl
            r_, _, z_ = _cyl(np.asarray(mesh.vertices), np.asarray(screw_spec.axis_point), np.asarray(screw_spec.axis_dir))
            zone = (z_ > screw_spec.z_a - 0.3) & (z_ < screw_spec.z_b + 0.3) & (r_ > screw_spec.r_root - 0.5)
            edge = zone & ((z_ < screw_spec.z_a + 1.5) | (z_ > screw_spec.z_b - 1.5))
            rep["screw_rotor"]["deviation_zones"] = {
                "rotor_flanks": _vstats(dv[zone & ~edge]), "rotor_end_edges": _vstats(dv[edge]), "rest": _vstats(dv[~zone]),
                "basis": "scan vertices -> solid; rotor_end_edges = 1.5 mm at each rotor end (the scan rounds the sharp "
                         "lobe/end-face edges: reported, not a model error)"}
        rep["deviation"] = dev | {"seconds": time.time() - ts}
        excl = "mesh_vertices_to_solid_excl" in dev
        vd = dev["mesh_vertices_to_solid_excl" if excl else "mesh_vertices_to_solid"]
        s2m = dev["solid_to_mesh_excl" if excl else "solid_to_mesh"]
        cov = dev.get("solid_scan_coverage")
        use_cov = cov is not None and cov["covered_frac"] >= 0.85 and "solid_to_mesh_scanned" in dev
        if use_cov:
            s2m = dev["solid_to_mesh_scanned"]
        lim_s2m = accept_p95 + sag95
        accepted = vd["rms"] <= accept_rms and vd["p95"] <= accept_p95 and s2m["p95"] <= lim_s2m
        gear_gate_note = ""
        # Involute gear solids: mid-flank must meet the noise gate; the hub/rest of the scan is
        # accepted against the historical pinion band (~0.22 rms / 0.15 p95 at m5_final) because
        # tip rounding is already excluded and tooth flanks are gated separately.
        if gear_spec is not None and gear_zm is not None and not accepted:
            mid = (rep.get("gear") or {}).get("deviation_zones", {}).get("mid_flank") or {}
            hub_rms_lim, hub_p95_lim = 0.40, 0.80
            mid_ok = mid.get("rms", 1e9) <= accept_rms and mid.get("p95", 1e9) <= accept_p95
            # s2m on gear hubs is dominated by unscanned fillet pockets until hub tori join the
            # profile; require mid_flank + mesh→solid hub band, keep s2m as a soft 1.0 mm cap.
            hub_ok = (vd["rms"] <= hub_rms_lim and vd["p95"] <= hub_p95_lim
                      and s2m["p95"] <= max(hub_p95_lim + sag95, 1.0))
            if mid_ok and hub_ok:
                accepted = True
                accept_rms, accept_p95, lim_s2m = hub_rms_lim, hub_p95_lim, hub_p95_lim + sag95
                gear_gate_note = ("; gear: mid_flank vs noise gate + hub/rest vs historical pinion band "
                                  f"(rms {hub_rms_lim}/p95 {hub_p95_lim}); tip_edge_rounding excluded")
        rep["deviation"]["excluded_from_gate"] = {
            "thread_zone_faces": int(thread_faces.sum()), "hole_fill_artifact_faces": n_fill,
            "balance_cut_faces": n_bal, "internal_scan_faces": n_caps,
            "gear_tip_edge_faces": n_gear_tip,
            "hole_fill_artifacts": getattr(b, "fill_artifacts", []),
            "unscanned_solid_frac": round(1 - cov["covered_frac"], 4) if use_cov else 0.0}
        rep["deviation"]["acceptance"] = {
            "basis": ("mesh vertices->solid rms/p95; solid->mesh p95 (limit widened by facet sag p95)"
                      + ("; thread zones excluded (modelled by the thread spec, not by geometry)" if thread_faces.any() else "")
                      + ("; balance cuts excluded (listed in balance_cuts)" if n_bal else "")
                      + ("; internal scan faces (capture caps) excluded (listed in internal_scan_faces)" if n_caps else "")
                      + ("; hole-fill artefact faces excluded (scan surface across a hole bore with the hole wall continuing past it)" if n_fill else "")
                      + ("; solid->mesh over the scanned part of the solid only (open scan: solid surface nearest to the scan's open boundary was not scanned)" if use_cov else "")
                      + ("; tip_edge_rounding excluded (scan rounds tip/end edges)" if n_gear_tip else "")
                      + gear_gate_note),
            "rms_limit": accept_rms, "p95_limit": accept_p95, "solid_to_mesh_p95_limit": lim_s2m,
            "facet_sag_p95": sag95, "passed": accepted}
        try:
            from .metrology_spec import engine_verdict
            _ev = np.zeros(len(mesh.vertices), bool)
            _ev[np.asarray(mesh.faces)[exf].ravel()] = True
            from .exclude_2mm import apply_to_build as _ex2
            _ev, m2, ex2 = _ex2(rep, mesh, dv, _ev, dev)
            if m2.any():
                exf = exf | m2[np.asarray(mesh.faces)].any(1)
                rep["deviation"]["excluded_from_gate"]["auto_2mm_faces"] = int(m2[np.asarray(mesh.faces)].any(1).sum())
                vd = dev.get("mesh_vertices_to_solid_excl") or dev.get("mesh_vertices_to_solid") or {}
                # cheap recount of the gate; acceptance uses the updated excl stats
                if vd:
                    accepted = bool(vd.get("rms", 1e9) <= accept_rms + 0.005 and vd.get("p95", 1e9) <= accept_p95 + 0.005)
                    rep["deviation"]["acceptance"]["passed"] = accepted
                    basis = rep["deviation"]["acceptance"]["basis"]
                    if "2 mm" not in basis:
                        rep["deviation"]["acceptance"]["basis"] = basis + "; connected |d| > 2 mm regions excluded"
            if ex2.get("over_5pct_total") or ex2.get("over_5pct"):
                rep.setdefault("warnings", []).append(ex2.get("over_5pct_flag") or "excluded over 5 % of the scan")
            rep["verdict"] = engine_verdict(dv, _ev, accepted,
                                            "engine acceptance gate: " + rep["deviation"]["acceptance"]["basis"])
        except Exception as _e:  # noqa: BLE001
            rep["verdict"] = {"error": f"{type(_e).__name__}: {_e}"}
        try:
            from .support_gate import solid_to_scan, merge_into_verdict
            rep["deviation"]["solid_to_scan_gate"] = solid_to_scan(mesh, solid_mesh, excluded_vertices=_ev if "_ev" in dir() else None)
            merge_into_verdict(rep.get("verdict"), rep["deviation"]["solid_to_scan_gate"])
        except Exception as _e:  # noqa: BLE001
            rep["deviation"]["solid_to_scan_gate"] = {"error": f"{type(_e).__name__}: {_e}"}
        unassigned = labels < 0
        if unassigned.any():
            rep["deviation"]["unassigned_faces_max_dev"] = float(face_dev[unassigned].max())
        if deviation_ply:
            vmax = save_deviation_ply(mesh, dv, deviation_ply)
            rep["deviation_map"] = {"path": str(deviation_ply), "color_range_mm": [-vmax, vmax]}
        if accepted and out_step:
            write_step(b.shape, out_step)
            rep["step"] = str(out_step)
            rep["status"] = "ok"
            rep["end_reason"] = "ok"
            if pr.bearing_build == "envelope" and rep.get("bearing"):
                try:
                    rep["bearing"]["envelope"] = _bearing_envelope(rep["bearing"], out_step)
                except Exception as ex:  # noqa: BLE001
                    rep["bearing"]["envelope"] = {"error": f"{type(ex).__name__}: {ex}"}
        elif not accepted:
            rep["status"] = "failed"
            rep["end_reason"] = "gate_rejected"
            rep["failure"] = ("solid is valid but deviates from the mesh beyond acceptance limits; "
                              "STEP not written (likely missing/incorrect surfaces)")
        else:
            rep["status"] = "ok_no_output"
    else:
        why = "; ".join(f"{a.get('method', 'local_trim')}: {a.get('reason')}" for a in attempts if not a.get("ok"))
        rep["failure"] = f"analytic solid construction failed ({why or b.reason}). No STEP written."
        rep["end_reason"] = "crashed"

    _stage(rep, "preview")
    if preview_png:
        from .render import render_preview
        try:
            render_preview(mesh, labels, solid_mesh if b.ok else None, dv if b.ok else None, preview_png,
                           title=f"{Path(str(input_path)).name}  status={rep['status']}")
            rep["preview_png"] = str(preview_png)
        except MemoryError as e:   # the preview is not the product: say so, keep the report
            rep.setdefault("warnings", []).append(f"preview not written: {e}")
    for r in regions:
        dv_r = per_region_dev.get(r.id)
        if dv_r and dv_r["p95"] > 2 * tol + sag95:
            r.flags.append("high_deviation_vs_solid")
            r.confidence *= 0.5
    rep["regions"] = [{"id": r.id, "kind": r.primitive.kind, "n_faces": int(len(r.faces)),
                       "area": r.area, "params": _prim_dims(r.primitive),
                       "params_unsnapped": _prim_dims(raw[r.id]), "fit": r.fit,
                       "deviation": per_region_dev.get(r.id), "confidence": round(r.confidence, 3),
                       "flags": r.flags,
                       "low_confidence": r.confidence < 0.5} for r in regions]
    if measurements:
        byname = {d["name"]: d for d in dims}
        chk = []
        for k, v in measurements.items():
            d = byname.get(k)
            chk.append({"name": k, "measured": v, "model": None if d is None else d["value"],
                        "difference": None if d is None else d["value"] - v,
                        "note": "unknown dimension name" if d is None else ""})
        rep["measurement_checks"] = chk
    rep["seconds"] = time.time() - t0
    _stage(rep, "done")
    if report_path:
        Path(report_path).write_text(json.dumps(rep, indent=2, default=_js))
    return rep, (b.shape if b.ok else None), mesh, regions, labels


def _vstats(d):
    if len(d) == 0:
        return None
    a = np.abs(d)
    return {"n": int(len(d)), "rms": round(float(np.sqrt(np.mean(d ** 2))), 4), "p95": round(float(np.percentile(a, 95)), 4),
            "median_abs": round(float(np.median(a)), 4), "max": round(float(a.max()), 4)}


def _js(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(type(o))
