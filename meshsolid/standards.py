"""Standards library (M9): nominal sizes the recognisers snap to.  Short numeric facts only.

  keys        parallel keys ISO/DIN 6885-1 (metric) and ANSI B17.1 square keys (inch): shaft range ->
              key width b, height h, shaft keyseat depth t1, hub depth t2
  splines     involute splines DIN 5480 (30 deg, module series, reference diameters) and ANSI B92.1
              (30 deg, fractional pitch P/Ps); straight-sided splines ISO 14 (light, medium series)
  bearings    deep-groove ball bearings 60xx, 62xx, 63xx and cylindrical roller NU2xx boundary
              dimensions (bore d, outside D, width B)
Every lookup returns the candidate and its misfit so callers can report how close a snap was and
refuse a snap that is not close."""
from __future__ import annotations

IN = 25.4

# ISO/DIN 6885-1: (d over, d up to, b, h, t1 shaft, t2 hub), mm
DIN6885 = [(6, 8, 2, 2, 1.2, 1.0), (8, 10, 3, 3, 1.8, 1.4), (10, 12, 4, 4, 2.5, 1.8), (12, 17, 5, 5, 3.0, 2.3),
           (17, 22, 6, 6, 3.5, 2.8), (22, 30, 8, 7, 4.0, 3.3), (30, 38, 10, 8, 5.0, 3.3), (38, 44, 12, 8, 5.0, 3.3),
           (44, 50, 14, 9, 5.5, 3.8), (50, 58, 16, 10, 6.0, 4.3), (58, 65, 18, 11, 7.0, 4.4), (65, 75, 20, 12, 7.5, 4.9),
           (75, 85, 22, 14, 9.0, 5.4), (85, 95, 25, 14, 9.0, 5.4), (95, 110, 28, 16, 10.0, 6.4), (110, 130, 32, 18, 11.0, 7.4),
           (130, 150, 36, 20, 12.0, 8.4), (150, 170, 40, 22, 13.0, 9.4), (170, 200, 45, 25, 15.0, 10.4),
           (200, 230, 50, 28, 17.0, 11.4)]

# ANSI B17.1 square keys: (d over, d up to, key width = height), inch; keyseat depth at the side = w / 2
B17_1 = [(5 / 16, 7 / 16, 3 / 32), (7 / 16, 9 / 16, 1 / 8), (9 / 16, 7 / 8, 3 / 16), (7 / 8, 1.25, 1 / 4),
         (1.25, 1.375, 5 / 16), (1.375, 1.75, 3 / 8), (1.75, 2.25, 1 / 2), (2.25, 2.75, 5 / 8), (2.75, 3.25, 3 / 4),
         (3.25, 3.75, 7 / 8), (3.75, 4.5, 1.0), (4.5, 5.5, 1.25), (5.5, 6.5, 1.5)]

DIN5480_MODULES = (0.5, 0.6, 0.75, 0.8, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0)
DIN5480_DB = (6, 7, 8, 9, 10, 12, 14, 15, 16, 17, 18, 20, 22, 25, 28, 30, 32, 35, 37, 38, 40, 42, 45, 47, 48, 50,
              55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210,
              220, 240, 250, 260, 280, 300, 320, 340, 360, 380, 400, 420, 440, 450, 460, 480, 500)
B92_1_PITCHES = ((2.5, 5), (3, 6), (4, 8), (5, 10), (6, 12), (8, 16), (10, 20), (12, 24), (16, 32), (20, 40),
                 (24, 48), (32, 64), (40, 80), (48, 96))

# ISO 14 straight-sided splines: (series, N, d minor, D major, B width), mm
ISO14 = [("light", 6, 23, 26, 6), ("light", 6, 26, 30, 6), ("light", 6, 28, 32, 7), ("light", 8, 32, 36, 6),
         ("light", 8, 36, 40, 7), ("light", 8, 42, 46, 8), ("light", 8, 46, 50, 9), ("light", 8, 52, 58, 10),
         ("light", 8, 56, 62, 10), ("light", 8, 62, 68, 12), ("light", 10, 72, 78, 12), ("light", 10, 82, 88, 12),
         ("light", 10, 92, 98, 14), ("light", 10, 102, 108, 16), ("light", 10, 112, 120, 18),
         ("medium", 6, 11, 14, 3), ("medium", 6, 13, 16, 3.5), ("medium", 6, 16, 20, 4), ("medium", 6, 18, 22, 5),
         ("medium", 6, 21, 25, 5), ("medium", 6, 23, 28, 6), ("medium", 6, 26, 32, 6), ("medium", 6, 28, 34, 7),
         ("medium", 8, 32, 38, 6), ("medium", 8, 36, 42, 7), ("medium", 8, 42, 48, 8), ("medium", 8, 46, 54, 9),
         ("medium", 8, 52, 60, 10), ("medium", 8, 56, 65, 10), ("medium", 8, 62, 72, 12), ("medium", 10, 72, 82, 12),
         ("medium", 10, 82, 92, 12), ("medium", 10, 92, 102, 14), ("medium", 10, 102, 112, 16),
         ("medium", 10, 112, 125, 18)]

# rolling bearings: designation -> (d, D, B) mm
BEARINGS = {}
for _n, _row in (("60", [(0, 10, 26, 8), (1, 12, 28, 8), (2, 15, 32, 9), (3, 17, 35, 10), (4, 20, 42, 12), (5, 25, 47, 12),
                         (6, 30, 55, 13), (7, 35, 62, 14), (8, 40, 68, 15), (9, 45, 75, 16), (10, 50, 80, 16)]),
                 ("62", [(0, 10, 30, 9), (1, 12, 32, 10), (2, 15, 35, 11), (3, 17, 40, 12), (4, 20, 47, 14), (5, 25, 52, 15),
                         (6, 30, 62, 16), (7, 35, 72, 17), (8, 40, 80, 18), (9, 45, 85, 19), (10, 50, 90, 20),
                         (11, 55, 100, 21), (12, 60, 110, 22)]),
                 ("63", [(0, 10, 35, 11), (1, 12, 37, 12), (2, 15, 42, 13), (3, 17, 47, 14), (4, 20, 52, 15), (5, 25, 62, 17),
                         (6, 30, 72, 19), (7, 35, 80, 21), (8, 40, 90, 23), (9, 45, 100, 25), (10, 50, 110, 27)]),
                 ("NU2", [(4, 20, 47, 14), (5, 25, 52, 15), (6, 30, 62, 16), (7, 35, 72, 17), (8, 40, 80, 18), (9, 45, 85, 19),
                          (10, 50, 90, 20)])):
    for _k, _d, _D, _B in _row:
        BEARINGS[f"{_n}{_k:02d}"] = (_d, _D, _B)


def key_for_shaft(d_mm, width_mm=None):
    """Matching parallel keys for a shaft diameter: the DIN 6885 row and the ANSI B17.1 row whose shaft
    range contains d (both ends inclusive by 0.5%), each with its width misfit when width_mm is given.
    Sorted by width misfit (or metric first without a width)."""
    out = []
    for lo, hi, b, h, t1, t2 in DIN6885:
        if lo * 0.995 < d_mm <= hi * 1.005:
            out.append({"standard": "DIN 6885-1", "designation": f"{b} x {h}", "b": b, "h": h, "t1": t1, "t2": t2,
                        "shaft_range_mm": [lo, hi]})
    din = DIN6885
    if width_mm is not None:                       # any DIN width whose b fits (keys can be sized off-range)
        for lo, hi, b, h, t1, t2 in din:
            if abs(b - width_mm) < 0.25 and not any(o["b"] == b and o["standard"] == "DIN 6885-1" for o in out):
                out.append({"standard": "DIN 6885-1", "designation": f"{b} x {h}", "b": b, "h": h, "t1": t1, "t2": t2,
                            "shaft_range_mm": [lo, hi], "note": "shaft diameter outside this key's range"})
    d_in = d_mm / IN
    for lo, hi, w in B17_1:
        if lo * 0.995 < d_in <= hi * 1.005 or (width_mm is not None and abs(w * IN - width_mm) < 0.25):
            out.append({"standard": "ANSI B17.1 square", "designation": _frac(w) + " in", "b": round(w * IN, 4),
                        "h": round(w * IN, 4), "t1": round(w * IN / 2, 4), "t2": None, "shaft_range_in": [lo, hi],
                        **({} if lo * 0.995 < d_in <= hi * 1.005 else {"note": "shaft diameter outside this key's range"})})
    for o in out:
        o["width_misfit_mm"] = None if width_mm is None else round(abs(o["b"] - width_mm), 4)
    return sorted(out, key=lambda o: (o["width_misfit_mm"] if width_mm is not None else 0.0,
                                      "note" in o, o["standard"] != "DIN 6885-1"))


def _frac(x):
    from fractions import Fraction
    f = Fraction(x).limit_denominator(64)
    w, r = divmod(f.numerator, f.denominator)
    return (f"{w}-" if w and r else (f"{w}" if w else "")) + (f"{r}/{f.denominator}" if r else "")


def bearing_match(d, D, B, kinds=("60", "62", "63", "NU2")):
    """Bearings sorted by boundary-dimension misfit (max abs mm over d, D, B)."""
    rows = []
    for name, (dd, DD, BB) in BEARINGS.items():
        if not any(name.startswith(k) and (k != "6" or True) for k in kinds):
            continue
        rows.append({"designation": name, "d": dd, "D": DD, "B": BB,
                     "misfit_mm": round(max(abs(dd - d), abs(DD - D), abs(BB - B)), 4)})
    return sorted(rows, key=lambda r: r["misfit_mm"])


def iso14_match(N, d, D, B=None):
    rows = []
    for series, n, dd, DD, BB in ISO14:
        if n != N:
            continue
        m = max(abs(dd - d), abs(DD - D), abs(BB - B) if B is not None else 0.0)
        rows.append({"standard": "ISO 14", "series": series, "designation": f"{n} x {dd} x {DD}", "N": n, "d": dd,
                     "D": DD, "B": BB, "misfit_mm": round(m, 4)})
    return sorted(rows, key=lambda r: r["misfit_mm"])
