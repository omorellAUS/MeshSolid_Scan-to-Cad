# Design: Fusion install package + full drone STL/STEP pair test

Date: 2026-11-09 (Australia/Perth). Repo: quadremesh-fusion @ wip-local-trim. Local-only (no remotes).

## Scope (this milestone)

1. **Versioned Fusion/Windows install zip** (lossless of installable code + docs; no private meshes).
2. **Audit + inventory** of all 27 STEP / 52 STL drone+VTOL inputs with verified pairing classification.
3. **Comprehensive pair test** of every user-provided matched STL against its matched STEP truth (both mirrors), in a **new** output directory; keep `/workspace/meshsolid_testset/drone` as the untouched baseline.
4. Unmatched STLs listed and optionally scored against the input mesh only (never invent a STEP match).
5. **Not in scope:** finishing section-loft, claiming loft success, live Fusion/Windows/GPU runs on this Linux box, uploading parts, remote pushes, new Pages.

## Constraints (hard)

- Parameters created by the add-in **record** recognised design values next to the imported body; they **do not drive** the imported B-rep. Docs and UI text must keep saying so.
- CPU is mandatory; GPU (Warp) is optional via `-Gpu` / `requirements-gpu.txt`.
- Failures must report stage + reason; never import a STEP on failure.
- Do not claim live Fusion, Windows, or GPU were tested here (Linux box only). Outside-Fusion tests + packer self-check only.
- User parts under `tests/user_scans/` stay gitignored and uncommitted.
- WIP `meshsolid/loft.py` is preserved uncommitted; no success claim for loft.

## Approach (chosen)

**A. Package first, then pair runs (recommended).** Build `dist/MeshSolid-<version>-windows.zip` from a packer script with a unit test that asserts required files are present and private paths are absent. Start the full matched-pair engine runs detached into `/workspace/meshsolid_testset/drone_pairs_all/` while the zip is already deliverable.

Alternatives rejected for this chunk:
- **B. Pair runs first, package later:** risks losing the deliverable if a restore hits mid-run.
- **C. Single harness rewrite of tools/testset:** out of scope; reuse `pairing.py` inventory + a thin runner that calls the engine and scores against the matched STEP solid(s).

## Install zip contents

- `fusion_addin/` (MeshSolid add-in, README, `install_windows.ps1`)
- `freecad_addin/` (optional FreeCAD path the installer already supports)
- `meshsolid/` engine package (committed tree only)
- `requirements.txt`, `requirements-gpu.txt`, root `package.xml`, `scripts/bench_device.py`
- `INSTALL.txt` generated with version, install steps, CPU/GPU notes, "parameters do not drive B-rep", and explicit "not yet verified inside Fusion" checklist
- Exclusions: `.venv`, `__pycache__`, `settings.json`, `tests/user_scans/**`, `out/`, geometry under `results/`, `.git`

Version string = `fusion_addin/MeshSolid/MeshSolid.manifest` `version` (currently `0.1.0`), mirrored in FreeCAD `package.xml`.

## Pairing classification (from existing verified inventory)

Reuse `/workspace/meshsolid_testset/drone/inventory.json` (produced by `tools/testset/pairing.py`). Publish a human-readable artifact:

- **Matched (35):** STL → whole STEP / one solid / glued solids `i+j` / mirror of STEP solid. Include p2s_p95, frac_within, how.
- **Unmatched (17):** no verified STEP; note segment-of candidates (e.g. AIL 1/2 of AIL R) vs true orphans (no STEP sent).
- **Mirror twins (17):** fingerprint twins; pair test runs **both** sides when each has a match (user request), not only one representative.
- No unverified reference matches. Borderline matches (e.g. FRONT HATCH 1 p95≈0.22 mm) stay labelled with their measured p95.

## Pair test method

- New dir: `/workspace/meshsolid_testset/drone_pairs_all/` with `manifest.json`, `runs/`, `pair_results.csv`, `pair_results.md`.
- For each matched STL: run engine on the STL (CPU), write report+STEP on success; score rebuilt solid vs the **matched STEP portion** (not by importing the source STEP as success). Record: status, valid solid count, bidirectional deviation (mm) with sample counts/method, volume error when volumes are meaningful, runtime (engine `seconds` + wall), failure reason/stage.
- Concurrency: 2 workers; per-case wall budget ~400–600 s; watchdog/report on timeout.
- Unmatched STLs: listed in the artifact; optional input-only deviation if time remains; never claimed as STEP-matched.
- Baseline `/workspace/meshsolid_testset/drone` untouched. Distinguish "rerun this session" vs "prior baseline".

## Regressions

Re-run or cite prior green evidence for: M3, coupling, casings s04–s06, pinion p01, screw rotors, (impeller tolerances if not rebuilt). Label each as **rerun** or **prior result**.

## Verification on this box

- `pytest tests/test_addin.py`
- Packer unit test + zip inspection
- Installer syntax / `-CheckOnly` against the existing Linux venv where possible (not a Windows Fusion load)
- Pair runner smoke on one known-ok unmatched prismatic part is not a STEP-pair success; smoke a small matched part if one builds, else record failure honestly

## Deliverables for the parent

- Zip path under `dist/`
- Pairing + results CSV/MD path
- Counts, regression evidence, local commits, unresolved failures
- Exact remaining live Fusion/Windows test steps (from README checklist)
