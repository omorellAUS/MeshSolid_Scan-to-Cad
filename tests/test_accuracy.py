"""Accuracy class: printed absolute gates; machined default unchanged."""
from meshsolid.accuracy import resolve_gates, normalize_accuracy, PRINTED_RMS_MM, PRINTED_P95_MM


def test_normalize():
    assert normalize_accuracy(None) == "machined"
    assert normalize_accuracy("printed") == "printed"
    assert normalize_accuracy("drone") == "printed"


def test_machined_matches_legacy_factors():
    tol = 0.02
    g = resolve_gates("machined", tol)
    assert g["class"] == "machined"
    assert abs(g["rms_limit"] - 1.5 * tol) < 1e-12
    assert abs(g["p95_limit"] - 3.0 * tol) < 1e-12


def test_printed_absolute_0p05():
    g = resolve_gates("printed", tol=0.01653)
    assert g["class"] == "printed"
    assert g["rms_limit"] == PRINTED_RMS_MM == 0.05
    assert g["p95_limit"] == PRINTED_P95_MM == 0.10
    # not scaled by tol
    assert g["rms_limit"] != 1.5 * 0.01653


def test_explicit_override_wins():
    g = resolve_gates("printed", 0.02, accept_rms=0.03, accept_p95=0.08)
    assert g["rms_limit"] == 0.03
    assert g["p95_limit"] == 0.08
