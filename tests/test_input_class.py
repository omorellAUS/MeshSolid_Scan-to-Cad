import trimesh
from meshsolid import input_class as IC


def test_coarse_cad_tessellation_is_lowres_and_gates_loosen():
    m = trimesh.creation.cylinder(10, 40, sections=24)
    fst = {"cad_tessellation": True, "facet_width_median": 2.6, "sag_p90": 0.05}
    ic = IC.classify(m, 0.4, False, fst)
    assert ic["class"] == "lowres_tessellation"
    ne, _ = IC.effective_noise(ic, 0.4)
    assert ne == 0.05                                     # sag, not the 0.4 mm quadric "noise"
    g = IC.lowres_gates(ic, {"rms_limit": 0.03, "p95_limit": 0.06, "class": "machined"}, 0.02)
    assert abs(g["rms_limit"] - 0.26) < 1e-9 and abs(g["p95_limit"] - 0.65) < 1e-9
    assert g["strict_limits"]["rms_limit"] == 0.03


def test_exact_but_not_faceted_is_not_lowres():
    m = trimesh.creation.box((10, 10, 10))
    assert IC.classify(m, 0.0, True, {"cad_tessellation": False})["class"] == "cad_tessellation"


def test_noisy_regular_mesh_is_scan():
    m = trimesh.creation.icosphere(5, radius=20)
    assert IC.classify(m, 0.005, False, {})["class"] == "scan"
