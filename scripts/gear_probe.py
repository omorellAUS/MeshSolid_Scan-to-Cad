"""Measure shaft/gear facts on a scan of a toothed shaft (pinion) - measurement only, no snapping.
Axis by PCA (right-handed frame), cylindrical coords, per-z-slice radial stats, tooth count by FFT of
the per-angle outer radius, helix lead from the slice-to-slice phase drift of the tooth harmonic.
Usage: python scripts/gear_probe.py mesh.stl out.json"""
import json, sys
import numpy as np
import trimesh

m = trimesh.load(sys.argv[1], process=True)
P = trimesh.sample.sample_surface(m, 1500000, seed=0)[0]
c = P.mean(0)
w, V = np.linalg.eigh(np.cov((P - c).T))
ax = V[:, 2]
# refine axis: centre the cross-sections (least squares line through slice centroids of circular parts)
def cyl(P, c, ax):
    e1 = np.cross(ax, [1, 0, 0]); e1 = e1 if np.linalg.norm(e1) > .1 else np.cross(ax, [0, 1, 0])
    e1 /= np.linalg.norm(e1); e2 = np.cross(ax, e1)          # (e1,e2,ax) right-handed
    Q = P - c
    z = Q @ ax; x = Q @ e1; y = Q @ e2
    return z, x, y
for _ in range(3):
    z, x, y = cyl(P, c, ax)
    zs = np.linspace(z.min() + 2, z.max() - 2, 60)
    cents = []
    for a, b in zip(zs[:-1], zs[1:]):
        k = (z >= a) & (z < b)
        if k.sum() < 500: continue
        # algebraic circle fit (Kasa)
        A = np.c_[2 * x[k], 2 * y[k], np.ones(k.sum())]
        sol = np.linalg.lstsq(A, x[k] ** 2 + y[k] ** 2, rcond=None)[0]
        cents.append([(a + b) / 2, sol[0], sol[1]])
    C = np.array(cents)
    e1 = np.cross(ax, [1, 0, 0]); e1 = e1 if np.linalg.norm(e1) > .1 else np.cross(ax, [0, 1, 0])
    e1 /= np.linalg.norm(e1); e2 = np.cross(ax, e1)
    pts = c + np.outer(C[:, 0], ax) + np.outer(C[:, 1], e1) + np.outer(C[:, 2], e2)
    c2 = np.median(pts, 0)
    _, _, Vt = np.linalg.svd(pts - pts.mean(0)); a2 = Vt[0] * np.sign(Vt[0] @ ax)
    c, ax = pts.mean(0), a2
z, x, y = cyl(P, c, ax)
r = np.hypot(x, y); th = np.arctan2(y, x)
L = z.max() - z.min()
dz = 0.5
zb = np.arange(z.min(), z.max() + dz, dz)
NB = 720
prof = []
for a in zb[:-1]:
    k = (z >= a) & (z < a + dz)
    if k.sum() < 200:
        prof.append(None); continue
    bi = ((th[k] + np.pi) / (2 * np.pi) * NB).astype(int) % NB
    rmax = np.full(NB, np.nan)
    np.fmax.at(rmax, bi, r[k])
    good = ~np.isnan(rmax)
    rr = np.where(good, rmax, np.nanmedian(rmax))
    F = np.fft.rfft(rr - rr.mean())
    amp = np.abs(F) / NB * 2
    h = int(np.argmax(amp[5:200]) + 5)
    prof.append({"z": float(a + dz / 2), "rmax": float(np.nanmax(rmax)), "rmin_outer": float(np.nanmin(rmax)),
                 "rmed": float(np.nanmedian(rmax)), "cov": float(good.mean()), "h": h, "amp": float(amp[h]),
                 "phase": float(np.angle(F[h])), "r_all_min": float(r[k].min())})
S = [p for p in prof if p]
Z = np.array([p["z"] for p in S]); amp = np.array([p["amp"] for p in S]); H = np.array([p["h"] for p in S])
toothed = amp > 0.5   # tooth harmonic amplitude > 0.5 mm => teeth present
N = int(np.bincount(H[toothed]).argmax()) if toothed.any() else None
# largest contiguous toothed run
best = (0, 0, 0); i = 0
while i < len(S):
    if toothed[i] and H[i] == N:
        j = i
        while j + 1 < len(S) and toothed[j + 1] and H[j + 1] == N: j += 1
        if j - i > best[0]: best = (j - i, i, j)
        i = j + 1
    else: i += 1
_, i0, i1 = best
G = S[i0:i1 + 1]
zg = np.array([p["z"] for p in G])
core = slice(len(G) // 6, len(G) - len(G) // 6)     # avoid tooth-end chamfers
r_tip = float(np.percentile([p["rmax"] for p in G][core], 50))
r_root = float(np.percentile([p["rmin_outer"] for p in G][core], 50))
# helix: unwrap phase of harmonic N along z; tooth angular position = -phase/N
ph = np.unwrap(np.array([p["phase"] for p in G]))
k = np.polyfit(zg[core], ph[core], 1)[0]          # rad (of harmonic) per mm
dtheta_dz = -k / N                                 # rad of rotation per mm along +ax
lead = 2 * np.pi / abs(dtheta_dz) if dtheta_dz != 0 else float("inf")
hand = "right" if dtheta_dz > 0 else "left"       # theta increasing with z in a right-handed frame = RH helix
beta_tip = float(np.degrees(np.arctan(r_tip * abs(dtheta_dz))))
# solve standard proportions: tip dia = mn (N/cos b + 2), b at pitch radius; tan b_p = r_p*|dth/dz|
mn = None
rp = r_tip * 0.9
for _ in range(50):
    bp = np.arctan(rp * abs(dtheta_dz))
    mn = 2 * r_tip / (N / np.cos(bp) + 2)
    rp = mn * N / np.cos(bp) / 2
bp_deg = float(np.degrees(np.arctan(rp * abs(dtheta_dz))))
# journals / collars: non-toothed stretches with stable radius
segs = []
cur = None
for p in S:
    if p["amp"] > 0.5 or p["cov"] < 0.6:
        if cur: segs.append(cur); cur = None
        continue
    if cur and abs(p["rmed"] - cur["r"][-1]) < 0.15:
        cur["r"].append(p["rmed"]); cur["z1"] = p["z"]
    else:
        if cur: segs.append(cur)
        cur = {"z0": p["z"], "z1": p["z"], "r": [p["rmed"]]}
if cur: segs.append(cur)
cyls = [{"z_from": round(s["z0"] - dz / 2, 2), "z_to": round(s["z1"] + dz / 2, 2), "length": round(s["z1"] - s["z0"] + dz, 2),
         "diameter_median": round(2 * float(np.median(s["r"])), 3), "diameter_spread": round(2 * float(np.ptp(s["r"])), 3)}
        for s in segs if s["z1"] - s["z0"] >= 3]
# diameter steps between consecutive cylinders = shoulder/collar faces (axial positions)
steps = []
for a, b in zip(cyls[:-1], cyls[1:]):
    steps.append({"z": round((a["z_to"] + b["z_from"]) / 2, 2), "from_d": a["diameter_median"], "to_d": b["diameter_median"]})
out = {"axis_length": round(float(L), 2), "z_min": round(float(z.min()), 2),
       "tooth_count": N, "toothed_z": [round(float(zg[0] - dz / 2), 2), round(float(zg[-1] + dz / 2), 2)],
       "face_width_incl_chamfers": round(float(zg[-1] - zg[0] + dz), 2),
       "tip_diameter": round(2 * r_tip, 3), "root_diameter_approx": round(2 * r_root, 3),
       "tooth_depth": round(r_tip - r_root, 3),
       "helix": {"hand": hand, "dtheta_dz_deg_per_mm": round(float(np.degrees(dtheta_dz)), 5), "lead_mm": round(float(lead), 1),
                 "beta_at_tip_deg": round(beta_tip, 2), "beta_at_est_pitch_deg": round(bp_deg, 2)},
       "est_standard": {"normal_module_mm": round(float(mn), 4), "transverse_module_mm": round(float(mn / np.cos(np.radians(bp_deg))), 4),
                        "normal_DP": round(25.4 / float(mn), 3), "pitch_diameter_mm": round(2 * float(rp), 3),
                        "whole_depth_ratio_h_over_mn": round((r_tip - r_root) / float(mn), 3),
                        "note": "assumes standard addendum = 1 mn; not snapped"},
       "cylinders": cyls, "shoulders": steps,
       "profile": [{k2: (round(v, 3) if isinstance(v, float) else v) for k2, v in p.items()} for p in S[::4]]}
json.dump(out, open(sys.argv[2], "w"), indent=1)
print(json.dumps({k2: v for k2, v in out.items() if k2 != "profile"}, indent=1))
