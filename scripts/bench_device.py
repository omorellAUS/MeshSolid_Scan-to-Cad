"""Benchmark the optional GPU path on YOUR machine (the dev box has no GPU, so speedups there are not
measured).  For each mesh: deviation-style closest-point queries of the scan's vertices against the
mesh itself shifted by noise, timed on the CPU (libigl) and on CUDA (NVIDIA Warp), plus the largest
difference between the two (should be ~1e-12 mm).

  python scripts/bench_device.py path\\to\\scan1.stl [scan2.stl ...] [--points 2000000]

Run it from the engine's Python (the one the installer set up, after `install_windows.ps1 -Gpu`)."""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import trimesh

from meshsolid import device as D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("meshes", nargs="+")
    ap.add_argument("--points", type=int, default=1_000_000)
    ap.add_argument("--repeat", type=int, default=3)
    a = ap.parse_args()
    ok, desc = D.cuda_info()
    print("CUDA:", desc)
    import igl
    for path in a.meshes:
        m = trimesh.load(path, force="mesh")
        rng = np.random.default_rng(0)
        P = trimesh.sample.sample_surface(m, a.points, seed=1)[0] + rng.normal(0, 0.05, (a.points, 3))
        V, F = np.asarray(m.vertices, float), np.asarray(m.faces, np.int64)
        t = []
        for _ in range(a.repeat):
            t0 = time.perf_counter(); d2, _, _ = igl.point_mesh_squared_distance(P, V, F); t.append(time.perf_counter() - t0)
        cpu = min(t)
        line = f"{os.path.basename(path)}: {len(F)} triangles, {len(P)} points | CPU libigl {cpu:.2f} s"
        if ok:
            D.closest_points(V, F, P[:1000], "cuda")            # compile / warm up
            t = []
            for _ in range(a.repeat):
                t0 = time.perf_counter(); _, d, _ = D.closest_points(V, F, P, "cuda"); t.append(time.perf_counter() - t0)
            gpu = min(t)
            line += f" | CUDA {gpu:.2f} s | speedup x{cpu / gpu:.1f} | max diff {np.abs(np.sqrt(d2) - d).max():.1e} mm"
        else:
            line += " | CUDA not available"
        print(line, flush=True)


if __name__ == "__main__":
    main()
