# Measurement spec v3 (MeshSolid 2.0.1)

Spec v3 keeps spec v2 alignment. Distance, sign, population, the 0.2 mm cap and the pass targets are unchanged.
The hash of v3 is `spec_hash(SPEC_V3)` in `meshsolid/metrology_spec.py`. v1 and v2 are not edited.

## What changed

- **Exclusions.** One policy, `meshsolid.exclusions.apply_policy`, is used by the engine and by Metrology.
  Balance cuts have their own caps (depth 5 mm, total area 2500 mm², both configurable) and are reported on
  their own line. They are left out of the deviation and out of the general 5 % exclusion cap.
- **Volume.** Error % is `100 * (V_solid - (V_mesh - V_hand_entered)) / |V_mesh - V_hand_entered|`.
  A hand-entered design bore is subtracted from the mesh volume.
- **Support.** Solid samples on a hand-entered feature are ignored, so the bore does not fail solid-to-scan support.
- **valid_solid.** Exactly one closed solid that passes `BRepCheck_Analyzer`. The solid count is always logged.

## Like-for-like re-measure

```
python -m meshsolid.remeasure MESH.stl SOLID.step --out DIR --convert-report report.json
```

This measures the existing pair under v2 and under v3 and writes every metric that changed. It does not rebuild the solid.
