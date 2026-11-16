"""Write the synthetic spline test parts (out/spline/<name>.stl, *_clean.stl, *_pose.npy; not committed)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from meshsolid import synthetic_splines as S  # noqa: E402

out = os.path.join(os.path.dirname(__file__), "..", "out", "spline")
os.makedirs(out, exist_ok=True)
for name in (sys.argv[1:] or S.PARTS):
    m, tr = S.part(name)
    m.export(os.path.join(out, f"{name}.stl")); tr["clean_mesh"].export(os.path.join(out, f"{name}_clean.stl"))
    np.save(os.path.join(out, f"{name}_pose.npy"), tr["pose"])
    print(name, len(m.faces))
