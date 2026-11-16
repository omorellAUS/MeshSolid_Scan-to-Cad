# MeshSolid 1.0.2

What's new since 1.0.1, and the known open issues. This is the engine and Fusion add-in package built from the `measure-v1` branch.

## What's new

- **Spec v1 / v2 Metrology and the honest verdict.** One-sided signed distance from scan to solid, 0.2 mm cap, `pass_pct_v1` counting capped points as failures (legacy counting is still logged). Spec v2 alignment is bounded point-to-plane ICP on at least 500k points, with a final rigid refinement and a warning above 0.01° or 0.01 mm. The target verdict is rms ≤ 0.03 mm, at least 90 % of points within ±0.05 mm (capped points count as failures), and no more than 5 % excluded. PASS is refused above 5 % excluded. Both pass percentages are labelled with the spec version.

- **Normalise.** The `--normalise report|apply` stage stores a 4×4 scan-to-normalised transform in the conversion report.

- **Recogniser fall-through.** A recogniser that fails no longer crashes the run; control falls through to the next one.

- **Splitter recognition.** Impeller splitters are detected from ring votes plus a per-sector leading-edge family test, and the build cuts them at the splitter leading edge. Cummins-style splitter vanes are the intended case.

- **Input-class detector.** Scan, CAD tessellation and decimated / low-resolution meshes are classified, with gates loosened for rough meshes (Cummins is treated as a low-resolution class). `--accuracy` defaults to unset so a low-res tessellation is not stuck on machined gates.

- **Cummins fixes.** Blade-trim and related impeller-build fixes so a Cummins 165 run produces a valid solid (fit accuracy is still short of the target; see open issues).

- **Turned-part gate.** A turned body with a minority of milled flats and cross-holes still passes the coaxial-area gate (cut features are credited). Cell-decomposition fallback uses a 2 % volume gate, so a tiny sliver is no longer accepted as the part.

- **Face classifier and balance-cut exclusion.** Front (nose) and back (balance-plane) faces of a recognised impeller are classified adaptively: irregular or non-periodic marks (balance cuts, drill marks, a one-off back-face slot) are not modelled and are listed as named Metrology exclusions with a 2 mm sector boundary; N-fold functional couplings (Hirth, drive dogs) are recognised and reported; concentric bores, spigots and registers stay with the hub. Uncertain features are excluded and flagged. `rim_slots=auto` no longer cuts on a recognised impeller.

- **2 mm auto-exclusion (any part).** Connected scan regions more than 2 mm from the solid become named whole-region exclusions in Metrology and the conversion report. Each is listed with its area, % of points and reason; its boundary is drawn on the deviation map; the total excluded % is stored and flagged when it is over 5 %. Those points are kept out of a cheap alignment refine and the conversion gate. `--no-auto-exclude-2mm` turns it off.

- **Solid-to-scan support gate.** Flags unsupported solid area above 0.5 % (material the solid has that the scan does not back). The result is merged into the honest verdict.

## Known open issues

- **Surface ripples** on stage 1 and stage 3 impeller skins.
- **Blade fit accuracy** is still short of the 90 % within ±0.05 mm target. STG1 is about 60 % within ±0.05 mm (0.063 mm rms, FAIL).
- **Turned-part profile collapse.** A small turned part with a milled flat and cross-holes is now recognised as turned, but the revolve profile can still collapse, and the run fails honestly instead of writing a sliver.
- **KK&K build time** is long; runs have hit the local-trim budget or been incomplete on this part.
- **Hirth modelling pending.** A Hirth coupling is recognised (tooth count, included angle, radii, height) and a cutter exists, but modelling on the real KK&K stage 1 part is not confirmed and is left pending when the profile test is inconclusive.
- **Root fillets.** Variable-radius G1 root fillets (and the impeller hub boolean) are not finished.

