"""Step 8: the solid -> scan direction of the gate.

The scan -> solid deviation cannot see solid material that has no scan points at all (an invented boss, a
filled slot): every scan point can sit on the solid while the solid carries extra geometry.  This check
samples the solid surface and asks, for each sample, whether the scan supports it:

  supported    nearest scan point within support_mm (default 0.2 mm, the spec v1 outlier cap)
  uncaptured   not supported, but the nearest scan point lies on/near an OPEN BOUNDARY of the scan (a hole or
               an edge of the captured skin): the scanner did not see that area, so it is not penalised
  unsupported  not supported and the nearest scan point is interior scan skin: the solid has geometry the
               scan contradicts

FAIL when the unsupported area exceeds max_unsupported_pct of the solid area (default 0.5 %).  The largest
connected unsupported patch is reported with its centre so it can be found."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

DEFAULTS = {"support_mm": 0.2, "boundary_margin_mm": 1.5, "max_unsupported_pct": 0.5, "n_samples": 200000,
            "patch_link_mm": 2.0}


def scan_boundary_vertices(scan):
    """Vertex indices on open boundary edges (edges used by one face)."""
    import trimesh
    e = np.asarray(scan.edges_sorted)
    if len(e) == 0:
        return np.zeros(0, int)
    one = trimesh.grouping.group_rows(e, require_count=1)
    return np.unique(e[one].ravel()) if len(one) else np.zeros(0, int)


def solid_to_scan(scan, solid_tm, cfg=None, seed=7, excluded_vertices=None):
    """scan: trimesh of the (aligned) scan; solid_tm: trimesh of the solid.  excluded_vertices: scan vertices
    left out of the deviation (named exclusions); solid samples whose nearest scan point is excluded are left
    out here too.  Returns the gate dict."""
    import trimesh
    c = dict(DEFAULTS, **(cfg or {}))
    V = np.asarray(scan.vertices, float)
    Q, fi = trimesh.sample.sample_surface(solid_tm, int(c["n_samples"]), seed=seed)
    a_tot = float(solid_tm.area)
    w = a_tot / len(Q)
    tree = cKDTree(V)
    # distance to the scan SURFACE (closest point on triangles near the nearest vertex is approximated by the
    # nearest vertex corrected with the scan's vertex normal: |(q - v) . n| when the lateral offset is small)
    dvert, j = tree.query(Q)
    n = np.asarray(scan.vertex_normals)[j]
    off = Q - V[j]
    along = np.abs(np.einsum("ij,ij->i", off, n))
    lateral = np.sqrt(np.maximum(dvert ** 2 - along ** 2, 0.0))
    edge = float(np.median(scan.edges_unique_length)) if len(scan.edges_unique_length) else 1.0
    d = np.where(lateral <= 1.5 * edge, along, dvert)
    # exact closest point on the scan surface where the vertex estimate says 'not supported' (coarse or CAD-like
    # scans have large triangles whose interiors are far from every vertex)
    chk = np.where(d > c["support_mm"])[0]
    if len(chk):
        cp, dist, tri = trimesh.proximity.closest_point(scan, Q[chk])
        d[chk] = np.minimum(d[chk], dist)
        j[chk] = np.asarray(scan.faces)[tri, np.argmin(np.linalg.norm(V[np.asarray(scan.faces)[tri]] - cp[:, None, :], axis=2), axis=1)]
    bnd = scan_boundary_vertices(scan)
    near_b = np.zeros(len(V), bool)
    if len(bnd):
        bt = cKDTree(V[bnd])
        near_b = bt.query(V, distance_upper_bound=c["boundary_margin_mm"])[0] < np.inf
    sup = d <= c["support_mm"]
    exc = np.zeros(len(Q), bool)
    if excluded_vertices is not None and np.any(excluded_vertices):
        exc = (~sup) & np.asarray(excluded_vertices, bool)[j]
    unc = (~sup) & (~exc) & near_b[j]
    uns = (~sup) & (~exc) & (~near_b[j])
    pct = lambda m: round(100.0 * float(m.sum()) / len(Q), 4)  # noqa: E731
    out = {"rule": f"solid surface samples with no scan within {c['support_mm']} mm, excluding samples whose nearest scan "
                   f"point is within {c['boundary_margin_mm']} mm of an open scan boundary (uncaptured); FAIL above "
                   f"{c['max_unsupported_pct']} % of the solid area",
           "n_samples": int(len(Q)), "solid_area_mm2": round(a_tot, 2),
           "supported_pct": pct(sup), "uncaptured_pct": pct(unc), "excluded_pct": pct(exc), "unsupported_pct": pct(uns),
           "unsupported_area_mm2": round(w * float(uns.sum()), 2),
           "unsupported_max_mm": round(float(d[uns].max()), 4) if uns.any() else 0.0,
           "scan_boundary_vertices": int(len(bnd)), "config": c}
    if uns.any():
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        iu = np.where(uns)[0]
        pr = cKDTree(Q[iu]).query_pairs(c["patch_link_mm"], output_type="ndarray")
        g = coo_matrix((np.ones(len(pr)), (pr[:, 0], pr[:, 1])), shape=(len(iu), len(iu))) if len(pr) else coo_matrix((len(iu), len(iu)))
        _, lab = connected_components(g, directed=False)
        big = np.bincount(lab).argmax(); m = lab == big
        out["largest_unsupported_patch"] = {"area_mm2": round(w * float(m.sum()), 2),
                                            "centre": np.round(Q[iu][m].mean(0), 3).tolist(),
                                            "max_mm": round(float(d[iu][m].max()), 4)}
    out["result"] = "FAIL" if out["unsupported_pct"] > c["max_unsupported_pct"] else "PASS"
    out["label"] = f"{out['result']} · {out['unsupported_pct']:.2f} % of the solid unsupported by the scan"
    return out


def merge_into_verdict(vd, gate):
    """Put the solid -> scan gate into a verdict dict (engine_verdict / metrology verdict): listed as its own
    check, and a target PASS becomes FAIL when the gate fails (the other direction cannot see invented solid)."""
    if not isinstance(vd, dict) or not isinstance(gate, dict) or "result" not in gate:
        return vd
    vd["solid_to_scan"] = {k: gate[k] for k in ("result", "label", "unsupported_pct", "uncaptured_pct",
                                                 "unsupported_area_mm2", "unsupported_max_mm") if k in gate}
    t = vd.get("target")
    if isinstance(t, dict):
        t.setdefault("checks", []).append({"name": "solid_to_scan_unsupported_pct", "value": gate["unsupported_pct"],
                                           "limit": gate["config"]["max_unsupported_pct"], "op": "<=",
                                           "ok": gate["result"] == "PASS",
                                           "text": f"<= {gate['config']['max_unsupported_pct']} % of the solid area unsupported by the scan"})
        if gate["result"] == "FAIL":
            t.setdefault("failed", []).append("solid_to_scan_unsupported_pct")
            if t.get("result") == "PASS":
                t["result"] = "FAIL"
                t["label"] = "FAIL · solid has geometry the scan does not support"
    return vd
