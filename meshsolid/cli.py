import argparse
import json
import os
import sys
from pathlib import Path

from .pipeline import Params, run


def _install_timeout_report():
    """On SIGTERM (user cancel / external timeout), write a partial report and exit 124.
    The end reason is cancelled. Never reports success."""
    import signal
    from . import pipeline as P

    def h(signum, frame):
        rep = P.CURRENT.get("rep")
        if rep is not None and P.CURRENT.get("report_path"):
            rep["status"] = "cancelled"
            rep["end_reason"] = "cancelled"
            rep["failure"] = f"cancelled by user (signal {signum}) during stage '{rep.get('last_stage')}'"
            Path(P.CURRENT["report_path"]).write_text(json.dumps(rep, indent=2, default=P._js))
        print(f"CANCELLED during stage {rep.get('last_stage') if rep else None}", file=sys.stderr, flush=True)
        os._exit(124)
    signal.signal(signal.SIGTERM, h)


def _keep_job_file(path: Path) -> bool:
    """STEP, reports, logs, and the preview PNG.

    The Fusion review reads the preview as soon as the CLI returns, so that PNG stays.
    Copied meshes, deviation PLY and the heartbeat do not.
    """
    name = path.name.lower()
    if "heartbeat" in name or name.endswith(".stage") or name.endswith(".tmp"):
        return False
    if name.endswith((".step", ".stp", ".log")):
        return True
    if name.endswith(".png") and "preview" in name:
        return True
    if name.endswith(".json") and "report" in name:
        return True
    return False


def prune_job_folder(report_path, preview_path=None, deviation_ply_path=None):
    """Drop copied meshes, deviation PLY and the heartbeat.

    STEP, reports, logs and the preview PNG stay. A shared batch folder (more than one
    report) only drops this run's deviation ply and heartbeat."""
    if not report_path:
        return []
    report = Path(report_path)
    folder = report.parent
    if not folder.is_dir():
        return []
    def keep(p: Path) -> bool:
        try:
            if p.resolve() == report.resolve():
                return True
        except OSError:
            pass
        return _keep_job_file(p)

    reports = [p for p in folder.iterdir() if p.is_file() and keep(p) and p.suffix.lower() == ".json"]
    single = len(reports) <= 1
    deleted = []
    targets = []
    if single:
        targets = [p for p in folder.iterdir() if p.is_file() and not keep(p)]
    else:
        # Shared batch folder: only this run's deviation ply and heartbeat. Previews stay
        # for the same reason as a single job folder (the review reads them).
        if deviation_ply_path:
            targets.append(Path(deviation_ply_path))
        targets.append(Path(str(report.with_suffix(".heartbeat.json"))))
        targets.append(Path(str(report.with_suffix(".stage"))))
    seen = set()
    for p in targets:
        key = str(p)
        if key in seen or not p.is_file():
            continue
        if p.resolve().parent != folder.resolve() and not single:
            # named intermediates may sit beside the report; don't follow a path outside the folder
            if p.parent.resolve() != folder.resolve():
                continue
        seen.add(key)
        try:
            p.unlink()
            deleted.append(str(p))
        except OSError:
            pass
    return deleted


def format_build_line(rep) -> str:
    """One log line. Impeller builds used to print solids=None; the count lives on the build record."""
    b = dict((rep.get("stages") or {}).get("build") or {})
    imp = ((rep.get("impeller") or {}).get("build") or {})
    if b.get("n_solids") is None and imp.get("n_solids") is not None:
        b["n_solids"] = imp["n_solids"]
    if b.get("n_faces") is None and imp.get("n_faces") is not None:
        b["n_faces"] = imp["n_faces"]
    valid = b.get("brepcheck_valid", b.get("valid", imp.get("valid", b.get("ok"))))
    volume = b.get("volume", 0) or 0
    n_solids = b.get("n_solids")
    if n_solids is None:
        n_solids = imp.get("n_solids")
    if n_solids is None:
        n_solids = 0
    valid_solid = b.get("valid_solid", imp.get("valid_solid"))
    return (f"regions: {len(rep.get('regions', []))}  build: valid={valid} valid_solid={valid_solid} "
            f"solids={n_solids} "
            f"faces={b.get('n_faces')} volume={float(volume):.3f} (mesh {rep.get('mesh', {}).get('volume')})")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m meshsolid",
                                 description="Mesh (STL/OBJ) -> analytic B-rep solid (STEP) for prismatic parts")
    ap.add_argument("input")
    ap.add_argument("-o", "--output", help="STEP output (only written if the solid validates)")
    ap.add_argument("--report", help="JSON report path")
    ap.add_argument("--deviation-ply", help="write colour-mapped deviation PLY")
    ap.add_argument("--preview", help="write preview PNG (input | solid | deviation)")
    ap.add_argument("--tol", type=float, default=None, help="fit tolerance mm (default: auto from noise estimate)")
    ap.add_argument("--accuracy", choices=["machined", "printed"], default=None,
                    help="deviation gate class: machined=1.5/3*tol (default); printed=0.05/0.10 mm absolute for FDM/SLA drone parts. The mesh classifier does not set or tighten this gate.")
    ap.add_argument("--mesh-source", choices=["scan", "cad_export"], default=None,
                    help="declared mesh source. Overrides parts.csv. The classifier is only a cross-check.")
    ap.add_argument("--parts", default=None, help="parts.csv; the source column for this mesh hash is authoritative")
    ap.add_argument("--balance-depth-cap", type=float, default=5.0,
                    help="balance-cut depth cap in mm (own cap, not the 5%% exclusion cap)")
    ap.add_argument("--balance-area-cap", type=float, default=2500.0,
                    help="balance-cut total-area cap in mm^2 (own cap, not the 5%% exclusion cap)")
    ap.add_argument("--fuse-bench", action="store_true",
                    help="opt-in: time OpenCASCADE parallel and glue options on the impeller fuse. "
                         "Off by default. The default solid is unchanged.")
    ap.add_argument("--sharp-deg", type=float, default=30.0)
    ap.add_argument("--round-tol", type=float, default=0.03)
    ap.add_argument("--no-split-bodies", action="store_true",
                    help="treat a multi-body STL as one mesh (default: build each connected body, multi-body STEP)")
    ap.add_argument("--builder", choices=["auto", "extrude", "local", "revolve", "cell"], default="auto")
    ap.add_argument("--rim-slots", choices=["auto", "on", "off"], default="auto",
                    help="flat-floored end-face pockets open to the rim: auto = cut each confident slot on non-impellers; on a recognised impeller auto does not cut (end-face features go through face_features)")
    ap.add_argument("--balance-cuts", choices=["auto", "on", "off"], default="auto",
                    help="rotor balance cuts on the end faces: list them and keep them out of the gate")
    ap.add_argument("--gears", choices=["auto", "off"], default="auto",
                    help="involute gear recognition (exact helical/spur gear cut into a turned body)")
    ap.add_argument("--gear-build", choices=["snapped", "free"], default="snapped",
                    help="build the gear from the standard (snapped) values or from the free fit")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda", "warp-cpu"], default=None,
                    help="compute device: auto (CUDA via NVIDIA Warp when available, else CPU), cpu, cuda "
                         "(error if no GPU), warp-cpu (the GPU code path on the CPU, for testing). Default: "
                         "MESHSOLID_DEVICE or auto")
    ap.add_argument("--bearing-build", choices=["full", "envelope"], default="full",
                    help="envelope: for a recognised catalogue bearing also write <out>.envelope.step (plain d x D x B ring)")
    ap.add_argument("--screws", choices=["auto", "off"], default="auto",
                    help="helical screw rotor recognition (exact helical sweep of the fitted transverse profile)")
    ap.add_argument("--impeller", choices=["auto", "off", "on"], default="auto",
                    help="N-fold bladed rotor path (hub revolve + passage floor + one patterned B-spline blade); "
                         "on = the part IS an impeller: fail with the recogniser's reason instead of falling through "
                         "to the general method")
    ap.add_argument("--normalise", choices=["report", "apply"], default="report",
                    help="pre-recognition normalise stage: 'report' stores the scan->normalised 4x4 only; 'apply' builds "
                         "axis-up / nose-up in mm and maps the STEP + deviation map back to the scan frame")
    ap.add_argument("--time-budget", type=float, default=float(os.environ.get("MESHSOLID_TIME_BUDGET", 0) or 0),
                    help="wall-clock budget in s (0 = none); past it the report says timeout + stage (watchdog)")
    ap.add_argument("--note", action="append", default=[], help="note copied into the report (repeatable)")
    ap.add_argument("--max-tools", type=int, default=80, help="max surfaces for cell decomposition")
    ap.add_argument("--no-round", action="store_true")
    ap.add_argument("--measure", action="append", default=[], metavar="NAME=VALUE",
                    help="physical measurement to check against a named dimension (e.g. D3=40.02)")
    a = ap.parse_args(argv)
    from . import device as D
    try:
        D.set_device(a.device)
    except (RuntimeError, ValueError) as ex:
        print("status: failed")
        print("FAILURE:", ex)
        return 2
    _install_timeout_report()
    meas = {}
    for m in a.measure:
        k, v = m.split("=")
        meas[k] = float(v)
    p = Params(tol=a.tol, sharp_deg=a.sharp_deg, round_tol=a.round_tol, do_round=not a.no_round, max_tools=a.max_tools, builder=a.builder, split_bodies=not a.no_split_bodies,
               balance_cuts=a.balance_cuts, rim_slots=a.rim_slots, gears=a.gears, gear_build=a.gear_build,
               impeller=a.impeller, normalise=a.normalise, screws=a.screws, bearing_build=a.bearing_build, notes=tuple(a.note),
               time_budget_s=a.time_budget, accuracy=a.accuracy, mesh_source=a.mesh_source, parts_csv=a.parts,
               balance_depth_cap_mm=a.balance_depth_cap, balance_area_cap_mm2=a.balance_area_cap,
               fuse_bench=a.fuse_bench)
    rep, *_ = run(a.input, a.output, a.report, p, a.deviation_ply, a.preview, meas or None)
    if rep.get("input_error"):
        print("status: failed")
        print("FAILURE:", rep["failure"])
        return 2
    b = rep["stages"].get("build", {})
    print(f"status: {rep['status']}  end_reason: {rep.get('end_reason')}")
    print(f"tol: {rep['tolerance']['tol']:.4f} (noise est {rep['tolerance']['noise_sigma_est']})  method: {b.get('method')}")
    print(format_build_line(rep))
    if "deviation" in rep:
        v = rep["deviation"]["mesh_vertices_to_solid"]
        print("deviation vertices->solid (all vertices, reference): max={:.4f} rms={:.4f} p95={:.4f}".format(
            float(v.get("max") or float("nan")), float(v["rms"]), float(v["p95"])))
        ve = rep["deviation"].get("mesh_vertices_to_solid_excl")
        if ve:
            print("deviation vertices->solid (excl. zones, GATE): max={:.4f} rms={:.4f} p95={:.4f}".format(
                float(ve.get("max") or float("nan")), float(ve["rms"]), float(ve["p95"])))
        s2 = rep["deviation"]["solid_to_mesh"]
        acc = rep.get("accuracy") or {}
        passed = (rep.get("deviation") or {}).get("acceptance", {}).get("passed")
        print("deviation solid->mesh: max={:.4f} rms={:.4f} p95={:.4f}  accepted={}  accuracy={} (rms<={}, p95<={})".format(
            float(s2.get("max") or float("nan")), float(s2.get("rms") or float("nan")), float(s2["p95"]),
            passed, acc.get("class", "machined"), acc.get("rms_limit"), acc.get("p95_limit")))
        d = rep["deviation"].get("mesh_to_solid")
        if d:
            print("deviation mesh(area samples)->solid: max={:.4f} mean={:.4f} rms={:.4f} p95={:.4f}".format(
                float(d["max"]), float(d["mean"]), float(d["rms"]), float(d["p95"])))
    for d in rep.get("dimensions", []):
        print(f"  {d['name']}: {d['type']} {d['value']:.4f} (fit {d['fit_value']:.4f}) regions {d['regions']}")
    summary = ((rep.get("deviation") or {}).get("acceptance") or {}).get("summary")
    if summary:
        print(summary)
    plain = (rep.get("verdict") or {}).get("plain") if isinstance(rep.get("verdict"), dict) else None
    if plain and plain != summary:
        print(plain)
    if rep.get("failure"):
        print("FAILURE:", rep["failure"])
    for c in rep.get("measurement_checks", []):
        print("  measure", c)
    if a.report:
        deleted = prune_job_folder(a.report, a.preview, a.deviation_ply)
        if deleted:
            print(f"pruned: deleted {len(deleted)} file(s); kept STEP, reports and logs")
    return 0 if rep["status"] == "ok" else 2


if __name__ == "__main__":
    sys.exit(main())
