"""Runs the meshsolid engine as a separate process (its own Python with OCP/trimesh/scipy; Fusion's
embedded Python cannot load those, and an engine crash must not take Fusion down) and reads its report.
No adsk import, so it is unit-testable outside Fusion."""
import json
import os
import subprocess
import time
from dataclasses import dataclass, field

from .outcome import CANCELLED, TIMEOUT, explain

SETTINGS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "settings.json")


@dataclass
class EngineSettings:
    python: str = ""            # engine interpreter, e.g. C:\meshsolid\venv\Scripts\python.exe
    engine_dir: str = ""        # folder containing the meshsolid package (added to PYTHONPATH)
    timeout_s: int = 5400
    extra_args: list = field(default_factory=list)

    @classmethod
    def load(cls, path=SETTINGS_FILE):
        if not os.path.exists(path):
            return cls()
        with open(path, encoding="utf-8-sig") as f:      # Windows PowerShell 5.1 writes a BOM
            d = json.load(f)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def save(self, path=SETTINGS_FILE):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.__dict__, f, indent=1)

    def problems(self):
        out = []
        if not self.python or not os.path.exists(self.python):
            out.append(f"engine Python not found: {self.python!r} (set it in settings.json)")
        if not self.engine_dir or not os.path.isdir(os.path.join(self.engine_dir, "meshsolid")):
            out.append(f"meshsolid package not found under {self.engine_dir!r}")
        return out


@dataclass
class EngineJob:
    mesh_path: str
    out_dir: str
    measurements: dict = field(default_factory=dict)     # e.g. {"bore_d": 14.29}
    notes: list = field(default_factory=list)
    options: dict = field(default_factory=dict)          # e.g. {"impeller": "auto", "tol": 0.02}

    def paths(self):
        base = os.path.join(self.out_dir, os.path.splitext(os.path.basename(self.mesh_path))[0])
        return {"step": base + ".step", "report": base + ".report.json", "preview": base + ".preview.png",
                "log": base + ".engine.log"}

    def command(self, s: EngineSettings):
        p = self.paths()
        cmd = [s.python, "-u", "-m", "meshsolid", self.mesh_path, "-o", p["step"], "--report", p["report"],
               "--preview", p["preview"]]
        # Auto CUDA via Warp when an NVIDIA GPU is present; falls back to CPU cleanly.
        # Time budget leaves headroom under the add-in kill timeout so the watchdog can
        # write a partial report before Fusion's outer timer kills the process.
        opts = dict(self.options)
        if "device" not in opts and not any(a == "--device" or a.startswith("--device=") for a in s.extra_args):
            opts["device"] = "auto"
        if "time_budget" not in opts and not any(
                a == "--time-budget" or a.startswith("--time-budget=") for a in s.extra_args):
            opts["time_budget"] = max(30, int(s.timeout_s) - 60) if s.timeout_s else 0
        for k, v in opts.items():
            if v is None or v == "" or (k == "time_budget" and float(v) <= 0):
                continue
            cmd += [f"--{k.replace('_', '-')}", str(v)]
        for k, v in self.measurements.items():
            cmd += ["--measure", f"{k}={v}"]
        for n in self.notes:
            cmd += ["--note", n]
        return cmd + list(s.extra_args)


@dataclass
class MetrologyJob:
    """Run `python -m meshsolid.metrology SCAN SOLID --out DIR` in the engine process."""
    scan_path: str
    solid_path: str
    out_dir: str
    options: dict = field(default_factory=dict)

    def paths(self):
        return {"log": os.path.join(self.out_dir, "metrology.engine.log"),
                "report": os.path.join(self.out_dir, "report.json"),
                "report_png": os.path.join(self.out_dir, "report.png"),
                "heatmap": os.path.join(self.out_dir, "heatmap.png"),
                "legend": os.path.join(self.out_dir, "legend.png"),
                "csv": os.path.join(self.out_dir, "deviation.csv"),
                "report_pdf": os.path.join(self.out_dir, "report.pdf")}

    def command(self, s: EngineSettings):
        o = dict(self.options)
        cmd = [s.python, "-u", "-m", "meshsolid.metrology", self.scan_path, self.solid_path,
               "--out", self.out_dir]
        if o.get("range"):
            cmd += ["--range", str(o["range"][0]), str(o["range"][1])]
        cmd.append("--symmetric" if o.get("symmetric", True) else "--no-symmetric")
        if o.get("pass_tol") is not None:
            cmd += ["--pass", str(o["pass_tol"])]
        if o.get("bands") is not None:
            cmd += ["--bands", str(int(o["bands"]))]
        if o.get("band_json"):
            cmd += ["--band-json", str(o["band_json"])]
        if o.get("exclude"):
            tags = o["exclude"] if isinstance(o["exclude"], str) else ",".join(o["exclude"])
            if tags:
                cmd += ["--exclude", tags]
        if o.get("max_dist"):
            cmd += ["--max-dist", str(o["max_dist"])]
            if o.get("cap_color"):
                cmd += ["--cap-color", str(o["cap_color"])]
        if o.get("align"):
            cmd.append("--align")
        if o.get("device"):
            cmd += ["--device", str(o["device"])]
        if o.get("lin") is not None:
            cmd += ["--lin", str(o["lin"])]
        if o.get("n_probes") is not None:
            cmd += ["--probes", str(int(o["n_probes"]))]
        for pt in o.get("probe") or []:
            cmd += ["--probe", ",".join(str(x) for x in pt)]
        if o.get("section"):
            org, nrm = o["section"]
            cmd += ["--section", ",".join(str(x) for x in org) + ":" + ",".join(str(x) for x in nrm)]
        if o.get("no_map"):
            cmd.append("--no-map")
        if o.get("no_zones"):
            cmd.append("--no-zones")
        if o.get("no_histogram"):
            cmd.append("--no-histogram")
        if o.get("no_section"):
            cmd.append("--no-section-plot")
        if o.get("zones"):
            cmd += ["--zones", str(o["zones"])]
        for f in o.get("exclude_faces") or []:
            cmd += ["--exclude-faces", str(f)]
        if o.get("convert_report"):
            cmd += ["--convert-report", str(o["convert_report"])]
        if o.get("auto_exclude") is False:
            cmd.append("--no-auto-exclude")
        if o.get("accept_pct") is not None:
            cmd += ["--accept-pct", str(o["accept_pct"])]
        for key, flag in (("title", "--title"), ("part_name", "--part-name"), ("operator", "--operator")):
            if o.get(key):
                cmd += [flag, str(o[key])]
        return cmd + list(s.extra_args)


MESH_EXT = (".stl", ".obj", ".ply")


@dataclass
class BatchJob:
    """Engine batch mode over a folder (python -m meshsolid.batch): every mesh is converted in its own
    engine process; meshes with a same-name STEP are scored against it.  Results stay in out_dir."""
    in_dir: str
    out_dir: str
    options: dict = field(default_factory=dict)
    recursive: bool = False

    def meshes(self):
        out = []
        for root, _, files in os.walk(self.in_dir):
            out += [os.path.join(root, f) for f in files if f.lower().endswith(MESH_EXT)]
            if not self.recursive:
                break
        return sorted(out)

    def paths(self):
        return {"log": os.path.join(self.out_dir, "batch.engine.log"), "summary": os.path.join(self.out_dir, "summary.json"),
                "csv": os.path.join(self.out_dir, "summary.csv")}

    def command(self, s: EngineSettings):
        cmd = [s.python, "-u", "-m", "meshsolid.batch", self.in_dir, "--out", self.out_dir, "--timeout", str(s.timeout_s)]
        if self.recursive:
            cmd.append("--recursive")
        opts = dict(self.options)
        # Parallel engine processes across cores; each mesh remains its own process.
        if "jobs" not in opts and not any(a == "--jobs" or a.startswith("--jobs=") for a in s.extra_args):
            import os
            opts["jobs"] = int(os.environ.get("MESHSOLID_BATCH_JOBS", "0") or 0) or max(1, (os.cpu_count() or 2) // 2)
        # Forward device=auto into engine_args for each child meshsolid.
        engine_forward = []
        if "device" not in opts and not any(a == "--device" or a.startswith("--device=") for a in s.extra_args):
            engine_forward += ["--device", "auto"]
        for k, v in opts.items():
            cmd += [f"--{k.replace('_', '-')}", str(v)]
        return cmd + engine_forward + list(s.extra_args)

    def settings(self, s: EngineSettings):
        """The whole batch may take timeout_s per mesh."""
        return EngineSettings(s.python, s.engine_dir, s.timeout_s * max(1, len(self.meshes())) + 600, list(s.extra_args))


def batch_summary_text(job: BatchJob):
    """Short text for the add-in's message box from summary.json (or why there is none)."""
    p = job.paths()["summary"]
    if not os.path.exists(p):
        return f"No batch summary was written; see {job.paths()['log']}"
    with open(p, encoding="utf-8") as f:
        d = json.load(f)
    lines = [f"{d['ok']} of {d['meshes']} meshes converted ({d['failed']} failed); {d['scored']} scored against a STEP."]
    if d.get("median_solid_vs_truth_rms") is not None:
        lines.append(f"Median solid vs STEP rms: {d['median_solid_vs_truth_rms']} mm")
    for r in d["rows"][:15]:
        extra = f", vs STEP rms {r['solid_vs_truth_rms']} mm" if r.get("solid_vs_truth_rms") is not None else ""
        lines.append(f"  {r['name']}: {r['status']}{extra}")
    if len(d["rows"]) > 15:
        lines.append(f"  ... {len(d['rows']) - 15} more")
    lines.append(f"Results: {job.out_dir} (summary.csv, one STEP + report per mesh)")
    return "\n".join(lines)


class EngineRun:
    """Popen wrapper the add-in polls from its UI loop (poll() returns quickly; the add-in calls
    adsk.doEvents() between polls and can cancel)."""

    def __init__(self, job: EngineJob, settings: EngineSettings):
        self.job, self.s = job, settings
        os.makedirs(job.out_dir, exist_ok=True)
        self.log_path = job.paths()["log"]
        self._log = open(self.log_path, "w", encoding="utf-8", errors="replace")
        self.t0 = time.time()
        self.code = None
        try:
            self.proc = subprocess.Popen(job.command(settings), stdout=self._log, stderr=subprocess.STDOUT,
                                         env=_engine_env(settings), cwd=job.out_dir,
                                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except OSError as e:                 # interpreter missing / not executable
            self._log.write(f"could not start the engine: {e}\n"); self._log.close()
            self.proc = None; self.code = 127

    def poll(self):
        """None while running; the exit code when done.  Past the timeout the engine is killed and
        poll() returns outcome.TIMEOUT."""
        if self.code is not None:
            return self.code
        rc = self.proc.poll()
        if rc is None and time.time() - self.t0 > self.s.timeout_s:
            self._kill(); rc = TIMEOUT
        if rc is not None:
            self.code = rc
            if not self._log.closed:
                self._log.close()
        return rc

    def cancel(self):
        if self.code is None:
            self._kill(); self.code = CANCELLED
            if not self._log.closed:
                self._log.close()

    def _kill(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait()

    def elapsed(self):
        return time.time() - self.t0

    def last_stage(self):
        """Last '[  t s] stage' line of the engine log (progress text)."""
        try:
            with open(self.log_path, encoding="utf-8", errors="replace") as f:
                lines = [l.strip() for l in f if l.startswith("[")]
            return lines[-1] if lines else "starting"
        except OSError:
            return "starting"


def _engine_env(s):
    env = dict(os.environ)
    env["PYTHONPATH"] = s.engine_dir + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.pop("PYTHONHOME", None)              # Fusion's embedded Python must not leak into the engine
    # Leave MESHSOLID_DEVICE alone when the user set it; otherwise let --device auto decide
    # (CUDA via Warp when an NVIDIA GPU is present, else CPU).
    return env


def selfcheck(s: EngineSettings, quick=True, timeout=300):
    """Run `python -m meshsolid.selfcheck` in the engine Python.  -> (ok, lines).  ok is False with the
    reason when the interpreter is missing, a package does not import, or (full mode) the synthetic
    round trip fails."""
    probs = s.problems()
    if probs:
        return False, probs
    import tempfile
    fd, jp = tempfile.mkstemp(suffix=".json"); os.close(fd)
    cmd = [s.python, "-m", "meshsolid.selfcheck", "--json", jp] + (["--quick"] if quick else [])
    try:
        p = subprocess.run(cmd, env=_engine_env(s), capture_output=True, text=True, timeout=timeout,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except subprocess.TimeoutExpired:
        return False, [f"engine self-check did not finish within {timeout} s"]
    except OSError as e:
        return False, [f"could not start the engine Python {s.python!r}: {e}"]
    try:
        with open(jp, encoding="utf-8") as f:
            res = json.load(f)
    except (OSError, ValueError):
        err = (p.stderr or p.stdout or "").strip().splitlines()[-8:]
        return False, [f"engine self-check failed (exit {p.returncode}) and wrote no result"] + err
    finally:
        if os.path.exists(jp):
            os.unlink(jp)
    lines = [f"engine Python {res.get('python')} at {res.get('executable')}"]
    lines += [f"error: {e}" for e in res.get("errors", [])] + [f"warning: {w}" for w in res.get("warnings", [])]
    if res.get("roundtrip"):
        rt = res["roundtrip"]
        lines.append(f"synthetic box: status {rt.get('status')}, {rt.get('n_solids')} solid, {rt.get('seconds')} s")
    return bool(res.get("ok")), lines


def read_result(job: EngineJob, rc=None, timeout_s=None):
    """Engine outcome for the add-in: the report, the plain-words outcome (lib.outcome.explain) and the
    file paths.  `step` is set only if the engine wrote one (it writes STEP only when the solid
    validates and passes its deviation gate)."""
    p = job.paths()
    rep = None
    if os.path.exists(p["report"]):
        try:
            with open(p["report"], encoding="utf-8") as f:
                rep = json.load(f)
        except ValueError:
            rep = None
    out = explain(rc, rep, p["log"], timeout_s=timeout_s, step_path=p["step"] if rep and rep.get("status") == "ok" else None)
    return {"status": (rep or {}).get("status"), "report": rep, "outcome": out, "step": out["step"],
            "preview": p["preview"] if os.path.exists(p["preview"]) else None,
            "report_path": p["report"], "log": p["log"], "out_dir": job.out_dir}
