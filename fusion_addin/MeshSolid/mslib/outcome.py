"""What happened in an engine run, in plain words, for the add-in's dialogs.  No adsk import.

Never reports success that did not happen: `importable` is True only when the engine's report says
status "ok" and the STEP file it names exists.  Every other outcome gets a title, a message that
says what is known (stage, limits, log tail) and, where there is one, a next step."""
import os
import re

CANCELLED, TIMEOUT = -1001, -1002          # EngineRun.poll() codes for an add-in cancel / timeout kill


def _f(x, nd=3):
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return "?"


def log_tail(path, n=15):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = [l.rstrip() for l in f if l.strip() and "Step File Name" not in l and "WorkSession" not in l]
        return lines[-n:]
    except OSError:
        return []


def deviation_line(rep):
    dev = rep.get("deviation") or {}
    acc = dev.get("acceptance") or {}
    v = dev.get("mesh_vertices_to_solid_excl") or dev.get("mesh_vertices_to_solid") or {}
    if not v:
        return None
    s = f"deviation rms {_f(v.get('rms'))} mm, p95 {_f(v.get('p95'))} mm"
    if acc:
        s += (f" (limits {_f(acc.get('rms_limit'))} / {_f(acc.get('p95_limit'))} mm: "
              f"{'within' if acc.get('passed') else 'OUTSIDE'})")
    return s


def verdict_line(rep):
    """Honest target verdict beside the engine's own acceptance gate, each with the exclusion %. None for old
    reports without a verdict."""
    vd = (rep or {}).get("verdict") or {}
    t, e = vd.get("target") or {}, vd.get("engine_limit") or {}
    if not t:
        return None
    m = vd.get("metrics") or {}
    why = ""
    if t.get("result") != "PASS":
        names = {"rms_uncapped": f"rms {_f(m.get('rms_uncapped'))} > 0.03 mm",
                 "pass_pct_v1": f"{_f(m.get('pass_pct_v1'), 1)} % within ±0.05 mm < 90 %",
                 "pct_excluded": f"{_f(m.get('pct_excluded'), 1)} % excluded > 5 %"}
        why = " (" + "; ".join(names.get(k, k) for k in t.get("failed") or []) + ")"
    return (f"Target verdict: {t.get('label', t.get('result'))}{why}  |  engine-limit result: "
            f"{e.get('label', e.get('result'))}")


def explain(rc, rep, log_path, timeout_s=None, step_path=None):
    """-> dict(kind, title, message, hint, importable, step)."""
    tail = log_tail(log_path)
    tail_txt = ("\n\nLast engine log lines:\n" + "\n".join(tail)) if tail else ""
    stage = (rep or {}).get("last_stage")
    if rc == CANCELLED:
        return dict(kind="cancelled", title="Cancelled", importable=False, step=None,
                    message=f"You cancelled the run during stage '{stage or 'start'}'. Nothing was imported.",
                    hint="")
    if rc == TIMEOUT:
        return dict(kind="timeout", title="Engine timed out", importable=False, step=None,
                    message=(f"The engine was stopped after {timeout_s} s, during stage '{stage or 'unknown'}'. "
                             f"Nothing was imported.{tail_txt}"),
                    hint="Raise timeout_s in settings.json, or decimate the mesh first.")
    if rep is None:
        return dict(kind="no_report", title="Engine failed to start or crashed", importable=False, step=None,
                    message=(f"The engine exited with code {rc} without writing a report.{tail_txt}"),
                    hint="Run the engine self-check (install_windows.ps1 -CheckOnly) to test the engine Python.")
    status = rep.get("status")
    failure = rep.get("failure")
    dl = deviation_line(rep)
    vl = verdict_line(rep)
    if vl:
        dl = (dl + "\n" + vl) if dl else vl
    step = rep.get("step") or step_path
    if status == "ok" and step and os.path.exists(step):
        return dict(kind="ok", title="Solid ready", importable=True, step=step,
                    message=("The engine built one valid solid within its own deviation limits (engine-limit result); "
                             "the target verdict below is the honest accuracy check." if vl else
                             "The engine built one valid solid within its deviation limits.") + (f"\n{dl}" if dl else ""),
                    hint="")
    if status == "ok":
        return dict(kind="missing_step", title="STEP file missing", importable=False, step=None,
                    message=f"The report says ok but the STEP file is not there ({step}). Nothing was imported.",
                    hint="Check that the job folder is writable and not cleaned by another program.")
    if status == "incomplete" and any(l.startswith("Traceback (most recent call last)") for l in log_tail(log_path, 400)):
        err = next((l for l in reversed(tail) if not l.startswith(" ")), "unknown error")
        hint = ("The mesh could not be read or repaired: check that it is a valid STL/OBJ with triangles."
                if stage == "load_repair" else "This is an engine bug; the report and log in the job folder show where.")
        return dict(kind="engine_error", title="Engine error", importable=False, step=None,
                    message=(f"The engine stopped with an error during stage '{stage}': {err}. Nothing was imported."
                             f"{tail_txt}"),
                    hint=hint)
    if status == "incomplete":
        return dict(kind="crash", title="Engine stopped unexpectedly", importable=False, step=None,
                    message=(f"The engine process ended during stage '{stage}' without finishing (exit code {rc}). "
                             "The cause is not recorded; out of memory is a common one on large meshes."
                             f"{tail_txt}"),
                    hint="Close other programs and retry, or decimate the mesh first.")
    if status == "timeout":
        return dict(kind="timeout", title="Engine timed out", importable=False, step=None,
                    message=f"{failure or 'The engine was stopped by a time limit.'} Nothing was imported.{tail_txt}",
                    hint="Raise timeout_s in settings.json.")
    if status == "ok_no_output":
        return dict(kind="no_output", title="No STEP requested", importable=False, step=None,
                    message="The engine ran without a STEP output path. This is an add-in bug; please report it.",
                    hint="")
    if rep.get("input_error"):
        return dict(kind="bad_input", title="Mesh could not be used", importable=False, step=None,
                    message=f"The engine could not use the input mesh: {rep['input_error']}. Nothing was imported.",
                    hint="Export the mesh again as binary STL or OBJ with triangles (Fusion: right-click the "
                         "mesh body > Save As Mesh; FreeCAD: File > Export).")
    gate = bool(((rep.get("deviation") or {}).get("acceptance") or {})) and not \
        rep["deviation"]["acceptance"].get("passed", False)
    if gate:
        return dict(kind="gate", title="Solid outside its deviation limits", importable=False, step=None,
                    message=("The engine built a solid, but it deviates from the mesh beyond the acceptance "
                             "limits, so it wrote no STEP and nothing was imported.\n" + (dl or "") +
                             (f"\n{failure}" if failure else "")),
                    hint="Open the preview and the report to see where it deviates.")
    return dict(kind="failed", title="Engine could not build a solid", importable=False, step=None,
                message=f"{failure or f'status {status}'}\nNothing was imported.{tail_txt}",
                hint=_hint(failure))


def _hint(failure):
    f = (failure or "").lower()
    if "no bore" in f:
        return "Enter the design bore as a measurement: bore_d=<mm>."
    if "watertight" in f or "open" in f:
        return "Close holes in the mesh (Mesh > Repair in Fusion) and retry."
    return "See the report and log in the job folder."


MEASURE_LABELS = {"bore_d": "Bore diameter (mm)"}


def measure_label(name):
    return MEASURE_LABELS.get(name, name.replace("_", " ") + " (mm)")


def missing_measurements(outcome, report=None, already=None):
    """Named measurements the engine said it needs (e.g. bore_d when the impeller scan has no bore).

    Picks names out of `--measure name=` / `name=<mm>` in the outcome or report, and the known
    'no bore' case. Skips names already present in `already`. Each item is {name, label}."""
    already = already or {}
    texts = [((outcome or {}).get("message") or ""),
             ((outcome or {}).get("hint") or ""),
             str((report or {}).get("failure") or "")]
    keys, seen = [], set()

    def add(k):
        k = (k or "").strip()
        if k and k not in already and k not in seen:
            seen.add(k)
            keys.append(k)

    blob = " ".join(texts)
    for m in re.finditer(r"--measure\s+([A-Za-z_][A-Za-z0-9_]*)\s*=", blob):
        add(m.group(1))
    for m in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)=<", blob):
        add(m.group(1))
    low = blob.lower()
    if "no bore" in low or "design bore" in low:
        add("bore_d")
    return [{"name": k, "label": measure_label(k)} for k in keys]
