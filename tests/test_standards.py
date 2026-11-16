from meshsolid import standards as S


def test_keys():
    k = S.key_for_shaft(40.0, 12.02)[0]
    assert k["standard"] == "DIN 6885-1" and k["designation"] == "12 x 8" and k["t1"] == 5.0
    a = S.key_for_shaft(25.4, 6.35)[0]
    assert a["standard"].startswith("ANSI") and a["designation"] == "1/4 in"


def test_bearings_and_iso14():
    assert S.bearing_match(25.02, 51.98, 15.0)[0]["designation"] in ("6205", "NU205")
    assert S.BEARINGS["6308"] == (40, 90, 23)
    assert S.iso14_match(8, 32.0, 36.0, 6.0)[0]["designation"] == "8 x 32 x 36"
