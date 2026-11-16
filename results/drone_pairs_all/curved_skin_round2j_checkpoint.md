# Curved-skin round 2j checkpoint (2026-11-09 ~13:42 AWST)

## Ask
Auto slab breaks where spanwise corner count changes (loft each slab with consistent corners, fuse on shared planar section faces into one solid); apply to WING 2 L, then an aileron and a fuselage section. Also LE sampling density on wingtips (LE band 0.119 dominates). Gates unchanged. Checkpoint ~25 min.

## Landed in `meshsolid/loft.py` (tests 6/6 green)

1. **`_corner_count_breaks`** — median-filtered counts (w=5), persist≥5, Δ≥4, min_slab≥12% span; merges into cap levels.
2. **Per-slab corner loft** when `len(lv)>2` and median corners≥2; soft `_corners_like` (positional fallback if turn signs flip); fall back to smooth TE-seamed if corner follow / seed fails; never prismatic with 1 section.
3. **Fuse → one solid** — `_ensure_solid` (closed shell→MakeSolid after Unify/ShapeFix demotion); `_bop` fuzzy ladder `1e-6…5e-4 * ext`; per-slab section budget by span fraction.
4. **Open sections** — `_section` key_tol ladder (fixes aileron dangling endpoints).
5. **Tip apexes gated** — only when *both* end-section areas collapse vs mid (<15%); stops aileron AABB-gap false pointed tips.
6. **LE densify** — `dense_at=0.5` on TE-seamed closed runs kept; `dense_power=2.2` **regressed** WINGTIP L 2 LE band → reverted to 1.5; Ms cap 128 kept.

Gates unchanged. Pipeline wire **skipped** (no greens).

## Probe (`curved_skin_round2j_probe.json`)

| part | ok | nsol | vol ratio | rms / p95 | notes |
|---|---|---|---|---|---|
| **WING 2 L** | no | 0→fuse reaches solid | — | — | 3 slabs at breaks `[-395,-322.8,-304.4,-290]`; all fell back to **smooth**; fuse ladder OK then time-budget / OCC hang risk before unify+self_check |
| **AILERON 1R VTOL** | no | 1 | **1.008** | rms **0.81** / p95 1.58 | Open-section fixed; tip-apex false positive fixed (was vol×10⁴); still misses mesh verts (hinge/cutout geometry) |
| **FUS 1 L** | no | 0 | — | — | Cap over-split (6 levels); cornered short slabs; budget / OCC hang inside ThruSections |
| **WINGTIP L 2** | no | 1 | 1.002 | rms **0.104** / p95 0.222 | Zones: tip **0.030**, TE 0.157, **LE 0.158**, mid 0.102 — LE still dominates; denser LE power did not help |

### WINGTIP L 2 zones (dense_power=1.5, Ms≤128)

| zone | rms | p95 |
|---|---|---|
| ALL | 0.104 | 0.222 |
| tip | 0.030 | 0.051 |
| TE | 0.157 | 0.040 |
| **LE** | **0.158** | **0.372** |
| mid | 0.102 | 0.184 |

(Prior 2i LE rms was **0.119** — no improvement this round; s2m 0.030 under lim 0.042.)

## Blockers (honest)

- **Cornered slabs still don't stick** on WING 2 L — follow/seed fails → smooth fallback, so mid-skin corner miss remains.
- **Multi-slab fuse** can produce a solid but unify/self_check + OCC BOP/ThruSections can overrun or hang past `time_budget` (Python checks don't interrupt OCC).
- **Aileron** solidifies (vol OK) but vertex rms ≫ gate — needs corner/hinge-aware sections, not tip tricks.
- **FUS** over-split by planar caps; needs fewer levels or extruded short slabs before loft.
- **LE densify** alone does not close the wingtip LE band.

## Next
1. Make cornered `_loft_slab` succeed on WING 2 L (stable TE+spar corners per slab) without smooth fallback — then fuse+self_check under budget.
2. Cap-level merge / short-slab prism for FUS; hard OCC time box or skip-loft short caps.
3. Aileron: section corners at hinge; reject ballooned ThruSections earlier.
4. LE: try LE-local tip patch or more sections only in LE chord band (not global dense_power).
5. Wire loft only after matched-pair greens.

## Artifacts
- `results/drone_pairs_all/curved_skin_round2j_probe.json`
- `results/drone_pairs_all/curved_skin_round2j_checkpoint.md`
