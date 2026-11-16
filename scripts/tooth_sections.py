"""Transverse sections of the toothed zone: outer radius vs angle for scan and each STEP (dense polylines),
best clocking by cross-correlation, residual radial differences. Usage: python scripts/tooth_sections.py scan.stl pinion_dir out.json out.png"""
import json, sys
from pathlib import Path
import numpy as np
import trimesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
scan = trimesh.load(sys.argv[1], process=True); d0 = Path(sys.argv[2])
NB = 7200
def dense(sec, step=0.01):
    out = []
    for e in sec.discrete:
        seg = np.diff(e, axis=0); L = np.linalg.norm(seg, axis=1)
        for p, s, l in zip(e[:-1], seg, L):
            n = max(int(l / step), 1); out.append(p + np.outer(np.arange(n) / n, s))
    return np.vstack(out)
def rth(P):
    r = np.hypot(P[:, 1], P[:, 2]); th = np.arctan2(P[:, 2], P[:, 1])
    b = ((th + np.pi) / (2 * np.pi) * NB).astype(int) % NB; o = np.full(NB, np.nan); np.fmax.at(o, b, r)
    i = np.arange(NB); g = ~np.isnan(o); return np.interp(i, i[g], o[g], period=NB)
res = {}
xs = [125.0, 132.5, 140.0, 147.5, 155.0]
fig, axs = plt.subplots(1, 2, figsize=(14, 5))
for ia, k in enumerate("AB"):
    T = np.load(d0 / f"T_step{k}_to_scan.npy")
    s2 = scan.copy(); s2.apply_transform(np.linalg.inv(T))
    m = trimesh.load(d0 / f"step{k}.tess.stl", process=False)
    rows = []
    for x in xs:
        a = rth(dense(s2.section(plane_origin=[x, 0, 0], plane_normal=[1, 0, 0])))
        b = rth(dense(m.section(plane_origin=[x, 0, 0], plane_normal=[1, 0, 0])))
        cc = np.fft.irfft(np.fft.rfft(a - a.mean()) * np.conj(np.fft.rfft(b - b.mean())), n=NB)
        sh = int(np.argmax(cc)); sh = sh - NB if sh > NB / 2 else sh
        pitch = NB / 33.0
        shm = (sh + pitch / 2) % pitch - pitch / 2          # residual clocking within one tooth pitch
        bs = np.roll(b, sh)
        rows.append({"x": x, "clocking_residual_deg": round(shm * 360 / NB, 3),
                     "radial_rms_as_aligned": round(float(np.sqrt(np.mean((a - b) ** 2))), 4),
                     "radial_rms_best_clocking": round(float(np.sqrt(np.mean((a - bs) ** 2))), 4),
                     "radial_p95_best_clocking": round(float(np.percentile(np.abs(a - bs), 95)), 4),
                     "tip_r_scan_p95": round(float(np.percentile(a, 97)), 3), "tip_r_step_p95": round(float(np.percentile(b, 97)), 3),
                     "root_r_scan_p3": round(float(np.percentile(a, 3)), 3), "root_r_step_p3": round(float(np.percentile(b, 3)), 3)})
        if x == 140.0:
            t = np.arange(NB) * 360 / NB - 180
            axs[ia].plot(t, a, lw=.8, label="scan"); axs[ia].plot(t, b, lw=.8, label=f"STEP {k} as aligned")
            axs[ia].plot(t, bs, lw=.8, ls="--", label=f"STEP {k} best clocking")
            axs[ia].set_xlim(-25, 25); axs[ia].legend(fontsize=8); axs[ia].set_xlabel("deg"); axs[ia].set_ylabel("outer r (mm)")
            axs[ia].set_title(f"section x=140 (STEP frame): scan vs STEP {k}")
    res[k] = rows
    for r_ in rows: print(k, r_)
fig.tight_layout(); fig.savefig(sys.argv[4], dpi=110)
json.dump(res, open(sys.argv[3], "w"), indent=1)
