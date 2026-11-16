# Splitter-blade impellers (full + splitter alternating)

Status 2026-11-16 (commit after 2495edb): **recognition only**. `impeller.probe` now reports
`info["splitter"] = {n_full, n_splitter, splitter_le_ring_index, votes}` when the ring votes go N_full on the
inner ring(s) and 2 x N_full outside (Cummins 140 mm: 8/16/16/16). Such a part is still not built by the
impeller route; the run falls through with the reason "splitter impeller: 8 full + 8 splitter blades ...;
splitter build not supported yet". Synthetic test: tests/test_impeller_splitter.py.

Case notes
- cummins_140_8plus8: votes 8/16/16/16 -> splitter pattern (this is the 8+8 wheel).
- cummins_165_15blade: votes 15/15/15/15, 15 is odd -> cannot alternate full/splitter; all full blades.
  Its 9.33 mm rms was a different bug: the exact slab x envelope Common came back empty (blade_volume 0, a bare
  hub was built). Fixed in 2495edb (fuzzy retry 0.01/0.03 mm + raise on an empty blade); re-run pending.
- KK&K stage 1: votes 16/16/16/16. If it has splitters their LE lies inside 0.40 R or the splitters reach the
  eye ring; the ring test cannot see them. Needs the per-sector LE test below.

Build plan (next)
1. Families: in `analyse`, after the frame and N (= 2 x N_full), split the blade points by sector parity
   (blade index from theta / (2 pi / N) after phase alignment). Per family: LE position = min meridional p
   of the blade's points, blade length; accept 2 families if LE(meridional) differs by > 10 % of the
   meridional length consistently in every sector (alternating). Also run this when all ring votes are equal
   (KK&K), adding inner rings 0.25/0.32 R to the vote.
2. Fit each family separately: camber + side sheets (`_fit_sides`) from its own points, each in the full
   blade's mer-map but with its own LE line (splitter: p_le > 0) and own LE round estimate; hub/shroud
   curves shared (one passage). Phase offset of the splitter family fitted (not assumed pi/N_full).
3. Build: `build` gets a list of blade specs: full slab trimmed by the envelope as now; splitter slab trimmed
   by the envelope cut at its own LE surface (revolved profile through the splitter LE line). Pattern each
   family N_full times with its phase, fuse with the hub (staged fuse, fuzzy fallback `_common` style).
4. Report: `blade_parameters` per family (N, LE r/z, wrap, beta in/out, thickness), phase offset.
5. Gate: blade-to-blade scatter per family (fold within each family only).
Tests: synthetic splitter wheel (STEP of a known 8+8 wheel -> mesh with 0.013 noise -> families recovered,
rebuild rms < 0.02); robustness set: cummins_140_8plus8, KK&K stage 1, cummins_165 (must stay single family).
