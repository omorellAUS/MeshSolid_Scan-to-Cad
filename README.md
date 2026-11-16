# quadremesh-fusion / meshsolid — milestone 1 (prismatic path)

Mesh (STL/OBJ) → segmented regions → fitted analytic primitives → snapped design intent →
watertight Open CASCADE B-rep → STEP + JSON metrology report. Core library + CLI so a Fusion 360
add-in can call it later (subprocess or import). License intent: GPL (personal use).

**Honesty rule:** a STEP file is written only if the B-rep is a single solid, `BRepCheck_Analyzer`
is valid, and mesh↔solid deviation passes acceptance (rms ≤ 1.5·tol, p95 ≤ 3·tol both directions).
Otherwise `status: failed` with a reason, exit code 2, and no STEP.

## Run
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m meshsolid part.stl -o part.step --report report.json \
    --preview preview.png --deviation-ply deviation.ply [--tol 0.1] [--no-round] [--measure D3=40.02]
.venv/bin/python -m pytest            # fast suite (13 tests)
.venv/bin/python -m pytest -m slow    # real coupling mesh (expected: honest failure)
.venv/bin/python scripts/run_synthetic.py out   # truth-vs-recovered table + previews
```

## Pipeline (meshsolid/)
| module | does |
|---|---|
| `meshio.py` | load keeping connectivity; merge vertices, drop degenerate/duplicate faces, fill holes ≤32 edges, report non-manifold edges, fix winding/orientation |
| `segment.py` | dihedral sharp edges → components; single-primitive fast path; else greedy fit-driven region growing (plane/cylinder/cone/sphere candidates from seeds, largest coverage wins); boundary relaxation + refit |
| `fit.py` | least-squares plane, cylinder, sphere, cone (scipy LM) with rms/max/normal error |
| `snap.py` | direction clustering → orthogonal frame; parallel/perpendicular snaps; coplanar & coaxial merge; cylinder–plane tangency (fillets); rounding to 1.0/0.5 mm grids within `round_tol` (logged raw→snapped); any snap that worsens fit is reverted |
| `build.py` | cell decomposition: oversized fitted surfaces split an enlarged bbox (`BRepAlgoAPI_Splitter`), each cell classified by generalized winding number of the mesh, inside cells fused + `UnifySameDomain`, validated |
| `deviation.py` | mesh→solid and solid→mesh distances (max/mean/rms/p95, Hausdorff), per-region and per-vertex signed deviation, colour PLY |
| `pipeline.py` | orchestration + report.json (per-region params raw & snapped, fit residuals, per-region deviation, confidence/flags, named dimensions D1..Dn, measurement checks) |
| `synthetic.py` | known-dimension CAD parts → isotropic remesh (pymeshlab) → Gaussian normal noise |

## Report fields for the refinement loop
`regions[]`: id, kind, params (snapped), params_unsnapped, fit{rms,max,normal_err_deg}, deviation{max,mean,rms,p95},
confidence, flags, low_confidence. `dimensions[]`: name (D1..), type (diameter/distance/span/cone angle), regions,
value (snapped), fit_value (raw LS). `measurement_checks[]` from `--measure NAME=VALUE` (check only — not yet fed
back as constraints).

## Known limits (milestone 1)
* No torus / freeform surfaces → parts with toroidal blends (e.g. the six-lobed coupling) fail honestly.
* Cell decomposition uses untrimmed surfaces; cost grows with surface count (`--max-tools`, default 80).
* Synthetic tests use round truth dims; the non-round rotated test runs with rounding off.
