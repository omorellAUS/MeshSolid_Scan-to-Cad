"""Write the synthetic screw rotor test meshes to out/screw/ (not committed)."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from meshsolid import synthetic_screw as S
os.makedirs("out/screw", exist_ok=True)
for n in (sys.argv[1:] or S.PARTS):
    t = time.time(); m, tr = S.part(n)
    tr["clean_mesh"].export(f"out/screw/{n}_clean.stl"); m.export(f"out/screw/{n}.stl")
    import numpy as np; np.save(f"out/screw/{n}_pose.npy", tr["pose"])
    print(n, len(m.faces), "clean watertight", tr["clean_mesh"].is_watertight, "scan watertight", m.is_watertight,
          "vol", round(tr["clean_mesh"].volume), f"{time.time()-t:.1f}s",
          {k: (round(v, 3) if isinstance(v, float) else v) for k, v in tr.items() if k in ("r_tip", "r_root", "lead_mm", "wrap_deg")}, flush=True)
