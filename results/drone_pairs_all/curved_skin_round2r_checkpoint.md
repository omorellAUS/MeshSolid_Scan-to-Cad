# Curved-skin round 2r — tip LE densify + non-splitting pockets

**When:** 2026-11-09 ~18:40 AWST (Australia/Perth)  
**Branch:** `wip-local-trim`  
**Commit:** `c29e066` (+ probe refresh)  
**Tests:** accuracy + loft drone floors + smooth skin — **13 passed**.

## (a) Wingtip LE-local (printed 0.05)

Generic tip path: `M=160` + `_resample_te_le_protect` (absolute mid/TE sample floors, narrow LE band 0.44–0.56). Insert-only densify trial regressed overall to ~0.17 and was reverted.

| Part | self_check rms | p95 | printed ≤0.05/0.10 | floor ≤0.059 | mid_ok | TE_ok | LE rms (base 0.119) |
|---|---|---|---|---|---|---|---|
| WINGTIP L 2 | **0.04193** | 0.090 | **PASS** | PASS | PASS | **FAIL** (0.076 vs 0.043) | **0.073** ↑ |
| WINGTIP R 2 | **0.04000** | — | **PASS** | PASS | PASS | **FAIL** (0.086 vs 0.043) | **0.075** ↑ |

- LE band improved; mid held; **TE zone soft** vs pre-densify baseline (tradeoff for printed overall).
- Do not claim TE zone green. Next: TE-gated fallback (accept le_local only if TE≤1.08×base) or denser TE ends without moving mid.

## (b) Pockets — non-splitting tools (WING 2 L)

Plumbing: boundary-loop rim (inset), opposite-skin thickness clamp (`which="opposite"`), prism-first + AABB lateral clamp, retry ladder; multi-station rails available.

| | value |
|---|---|
| base rms | 0.573 |
| skin-only target | **0.188** |
| after cuts rms | **0.468** (Δ **+0.105**) |
| feat / seed rms | feat 0.90; seeds 0.99→0.74 / 0.96→0.66 |
| n_solids | **1** (both pockets accepted) |
| dV removed | ~99 |

- **Non-split gate met** (nsol=1). Opening centre = loft closest-points of high-residual verts (skin bridge) — prior mesh-lip opening cut volume without lowering seed rms.
- Still short of skin-only **0.188** (feat residual remains). Next: deepen/fit floor to mesh; multi-station rails on elongated pocket; then belly.

## Not done
- Pocket rms → ~0.188
- TE zone floor on wingtips
- Belly skin-only after pockets
- Matched-pair `--accuracy printed` end-to-end

## Artifacts
- `curved_skin_round2r_wingtip_le.json`
- `curved_skin_round2r_pocket_cut.json`
