import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from meshsolid import synthetic_keys as K
from meshsolid import occ_util as U
os.makedirs("out/keys", exist_ok=True)
for n in sys.argv[1:] or K.PARTS:
    m, tr = K.part(n)
    m.export(f"out/keys/{n}.stl"); np.save(f"out/keys/{n}_pose.npy", tr["pose"]); U.write_step(tr["shape"], f"out/keys/{n}_truth.step")
    print(n, len(m.faces), m.is_watertight, round(m.volume, 1), round(U.volume(tr["shape"]), 1))
