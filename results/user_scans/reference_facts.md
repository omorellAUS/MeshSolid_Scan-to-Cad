# Reference facts for modelling (extracted from the user's PDFs, 25 Oct 2026)

Short factual extracts only; no document text is reproduced. Page numbers are PDF pages. OCR'd documents
are marked (OCR); OCR values should be double-checked against the original.
Files are in the user's attachments folder, identified by hash prefix.

## Tilt-pad / journal bearing geometry (bearing shells, pads, housings)
- **Preload definition** — Eisenmann, "Tilt-Pad Bearing Preload", Sulzer Technical Review 2/2004 (`ac377aa9`, p2-3):
  - Cbrg = diametral assembly (bearing) clearance = assembled bore diameter − journal diameter
    (= largest mandrel that fits − journal diameter);
  - Cpad = diametral pad clearance = machined pad bore diameter − journal diameter;
  - **Preload m = 1 − Cbrg/Cpad** (radial or diametral, but consistent). Pad curvature radius > shaft radius.
  - Worked example: 101.6 mm shaft, Cbrg 0.153 mm, Cpad 0.254 mm → m = 0.4. Industrial preloads usually 0.1–0.5.
  - Lift-check factors to get Cbrg from measured lift: 3 pads 0.667 (LOP and LBP); 4 pads 1.0 LOP / 0.707 LBP;
    5 pads 0.894; 6 pads 1.0 LOP / 0.866 LBP.
- **Glacier** — Jones & Martin, "Geometry Effects in Tilting-Pad Journal Bearings" (ASLE 1978) (`004188be`, OCR, p2-3):
  - Same definitions in radial form: c_p = R_pad − r (pad bore clearance), c'_b = R_brg − r (bearing clearance),
    preset/preload m = 1 − c'_b/c_p; pad clearance must exceed bearing clearance for positive preload.
  - Clearance-ratio charts span **Cd/d ≈ 0.001–0.004**.
  - Example bearing: 160 mm dia, 64 mm long (L/D 0.4), 5 pads × 60° arcs; diametral bearing clearance
    0.160–0.208 mm (≈ 0.0010–0.0013 × D).
- **Babbitt / white-metal liner thickness:** not given in any of the supplied documents.
  The NBS paper (`b2f364a0`, Herschman & Basil, RP512, 1932) gives only alloy compositions (tin-base ≈ 84–91% Sn
  with Sb/Cu; lead-base Pb-Sb-Sn) and their mechanical properties vs temperature.
- **Fits:** no ISO fit classes (H7/g6 etc.) are given in the supplied documents.
  TA2000 rotor/bearing-seal housings are located in their bores with woodruff keys (`5695d5e5` p1).
- **Turbine clearance sheet** (`f2394612`, OCR of a rotated German/English assembly-dimension form):
  - rotor displacement with all labyrinths installed: 0.52 mm desired / 0.58 mm actual (upwards) and
    0.18 / 0.22 mm (downwards);
  - emergency-trip bolt dimension 0.50–0.58 mm.
  - This is a steam turbine form, not the TA2000.

## Cooper / Cameron TA2000 (impellers, rotor)
Sources: "TA2000 rotor cartridge installation" technical training (`5695d5e5`, 28 p, text);
Cameron "Turbo Air 2000 Handbook" AAEDR-H-080 rev 03 (2008) (`3270685c`, 73 p, OCR at 150 dpi).
- 3-stage machine, two pinions:
  - low-speed rotor carries stage 1 and the 4th-stage bearing housing side;
  - high-speed rotor carries stages 2 and 3 (`5695d5e5` p1, p8-11).
- **Impeller axial tip clearance (TA2000):**
  - 1st stage 0.024" (0.610 mm); 2nd and 3rd stage 0.017" (0.432 mm) (`5695d5e5` p7 and p15);
  - the older-unit procedure B table lists 0.020" for stages 2 and 3 (p10-11, p13).
- **Impeller radial tip clearance** (inducer blade tip to inlet throat bore): 0.015" (0.381 mm) average on each stage;
  the 4 readings must be within 0.003"; the worksheet spec is 0.015–0.018" (`5695d5e5` p14, p15, p17).
- **Gearbox:** bullgear float 0–0.001"; pinion endfloat 0.004–0.008"; backlash 0.007–0.012" (TA2000) (p2, p15).
- **Diffuser vane counts (TA2000):** stage 1/2/3 = 11/9/7 (p15). (TA3000: 11/11/11; TA6000: 9/9/11.)
- **Performance:**
  - Standard ratings are tabulated at 100, 125 and 150 psig discharge, for 150–350 BHP
    (e.g. 250 BHP: 1315 / 1180 / 1110 ICFM) (`5695d5e5` p19).
  - The TA3000 axial tip clearances are design-pressure dependent (100/125/150 psig columns, p15).
  - **Nothing in the documents confirms which pressure the user's impellers are for**; 125 psig remains unconfirmed.
- **Weights:** 1st/2nd/3rd stage inlet 200/110/75 lb; bullgear 325 lb (`5695d5e5` p20; also handbook OCR).
- **Not found in the supplied documents:** impeller diameters, blade counts, bore/shaft fits, journal or thrust bearing
  clearances of the TA2000 pinion bearings, rotor balance tolerances.
- `87caa96c` (OCR, rotated 180°) is a Roteq "ITP split line compressor" inspection and test plan (Murrin Murrin, 2022).
  It lists the steps "check interstage seal clearance", "check final bearing clearance", "check impeller to diffuser
  line-up", but gives no values.
- `91122f71` ("List of problems and solutions for centrifugal air compressor") is generic troubleshooting with no
  dimensions.

## Cross-check of scanned parts against the documents
- 3rd-stage impeller (s07/s08/s09): **13 blades, no splitters, 82.6–83.8 mm OD**. The documents give no impeller OD
  or blade count, so no check is possible; the 3rd-stage diffuser has 7 vanes (13 and 7 are coprime, as is usual).
- 140 mm wheel (s01/s02/s03): **8 full + 8 splitter blades**. **Correction (user, 25 Oct 2026): this is the
  turbocharger compressor wheel of a Cummins KTA38 twin-turbo diesel generator, not a TA2000 impeller.** It is not
  cross-checked against the TA2000 documents (an earlier note here guessed "stage 1"; that guess is withdrawn).
- Only the 83 mm, 13-blade impeller (s07/s08/s09) is TA2000 (3rd stage).
- Radial tip clearance 0.381 mm and axial 0.432–0.610 mm concern the inlet shroud, which was not scanned.
  They bound the accuracy needed on the blade tip contour: scan deviation should be well below 0.4 mm there.
- Ring s04: no matching dimensions in the documents.

## User-supplied facts (25 Oct 2026, from the user, not from the PDFs)
- **Impeller-to-shaft fit:**
  - interference fit, measured at **0.025 mm** by micrometer at 25–28 °C ambient;
  - two impellers have different bore sizes, because brute-force pressing has delaminated the shaft/impeller interface.
- **Removal method:**
  - about 7 t hydraulic preload;
  - impeller heated to about 120 °C (foil around the tips);
  - shaft cooled with Loctite freeze spray through a threaded insert in the shaft face;
  - expected thermal growth/shrink 0.1–0.15 mm.
- **Modelling consequence:** impeller bores and shaft seats must be modelled to the **design fit**, not just to the
  scanned size. The report must carry both the measured (scan) value and the design value, and must not snap these
  diameters to round numbers.

## TA2000 low-speed pinion (user, 25 Oct 2026)
- The two pinion STEPs model **two physical spare pinions from different machines, made to the same drawing**:
  - A has the original seat size, Ø22.20 mm;
  - B (seat Ø22.10 mm) was reground after shaft damage from the factory impeller removal.
- **Gear:** imperial, US-made: **24 DP normal, left-hand helix about 20°**.
- Measured from scan and STEPs (not snapped): 33 teeth, LH, β 20.0° at pitch, mn 1.054–1.058 mm (24 DP = 1.0583 mm).
- **Rule for the gear recognizer:** snap to DP standards first (then metric module), and report the measured values next to them.
