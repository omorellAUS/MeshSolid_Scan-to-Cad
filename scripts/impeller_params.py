"""Blade parameters of an analysed impeller spec (meshsolid.impeller.analyse, pickled).
Usage: python scripts/impeller_params.py spec.pkl [out.json]"""
import json, pickle, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

s = pickle.load(open(sys.argv[1], "rb")); mm = s.mermap()
P = np.linspace(0, 1, 201)


def line(v):
    R, Z = mm.rz(P, np.full_like(P, v)); th = s.camber(P, np.full_like(P, v))
    tp = s.sides[1](P, np.full_like(P, v)); tm = s.sides[-1](P, np.full_like(P, v))
    dm = np.r_[0, np.cumsum(np.hypot(np.diff(R), np.diff(Z)))]
    return R, Z, th, (tp - tm) * R, dm


out = {"N": s.N, "z_le_plane": round(s.z_le, 3), "r_te": round(s.r_te, 3), "le_round": s.info.get("le_round"),
       "eye": s.info.get("eye"), "spans": {}}
for name, v in (("hub_root_incl_fillet", 0.02), ("hub_above_fillet", 0.25), ("mid", 0.5), ("tip", 0.98)):
    R, Z, th, t, dm = line(v)
    # blade angle from the meridional direction: tan(beta) = r dtheta/dm (camber line); 90 deg = tangential
    b = np.degrees(np.arctan(R * np.gradient(th, dm)))
    out["spans"][name] = {
        "LE_rz": [round(float(R[0]), 3), round(float(Z[0]), 3)], "TE_rz": [round(float(R[-1]), 3), round(float(Z[-1]), 3)],
        "meridional_length": round(float(dm[-1]), 2), "wrap_deg": round(float(np.degrees(abs(th[-1] - th[0]))), 2),
        "beta_inlet_deg_from_meridional": round(float(np.median(b[2:8])), 1),
        "beta_exit_deg_from_meridional": round(float(np.median(b[-8:-2])), 1),
        "thickness_tangential_mm": {f"{q:.2f}": round(float(np.interp(q, P, t)), 3) for q in (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.97)},
        # normal thickness ~ tangential * cos(blade angle) (the camber surface's lean is not included)
        "thickness_normal_mm": {f"{q:.2f}": round(float(np.interp(q, P, t * np.cos(np.radians(b)))), 3) for q in (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.97)},
    }
Rs, Zs = s.shroud.at(np.linspace(0, 1, 11))
out["tip_line_rz"] = [[round(float(a), 3), round(float(b), 3)] for a, b in zip(Rs, Zs)]
Rh, Zh = s.hub.at(np.linspace(0, 1, 11))
out["hub_line_rz"] = [[round(float(a), 3), round(float(b), 3)] for a, b in zip(Rh, Zh)]
out["exit_blade_height"] = round(float(np.hypot(Rs[-1] - Rh[-1], Zs[-1] - Zh[-1])), 3)
out["side_fit"] = s.info.get("side_fit"); out["axis_corrections"] = len(s.info.get("axis_corrections", []))
js = json.dumps(out, indent=1, default=str); print(js)
if len(sys.argv) > 2:
    open(sys.argv[2], "w").write(js)
