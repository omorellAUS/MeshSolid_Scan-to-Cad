import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from meshsolid import synthetic_bearings as B, occ_util as U
os.makedirs("out/brg", exist_ok=True)
for n in sys.argv[1:] or B.PARTS:
    m, tr = B.part(n)
    m.export(f"out/brg/{n}.stl"); np.save(f"out/brg/{n}_pose.npy", tr["pose"]); U.write_step(tr["shape"], f"out/brg/{n}_truth.step")
    print(n, len(m.faces), m.is_watertight, round(m.volume, 1), round(U.volume(tr["shape"]), 1), flush=True)
