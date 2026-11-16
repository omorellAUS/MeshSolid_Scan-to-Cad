"""Generic scan preparation (same steps for every part, no per-part tuning):
merge near-duplicate vertices, drop duplicate faces and small shells (<1% of faces), repair non-manifold
edges/vertices, close holes up to 1000 boundary edges, decimate (quadric, normals preserved) to a target
triangle count. Point clouds are meshed first by screened Poisson (depth 9) and trimmed to the input
bbox. Prints a JSON record of what was done.
Usage: python scripts/prep_scan.py in out.stl [target_faces]"""
import json, sys
import numpy as np
import pymeshlab
import trimesh


def close_small_holes(ms, maxholesize, span_frac=0.02):
    """Close holes, then re-open any fill patch that is not small: a patch whose mean width
    (2 x area / perimeter, = radius for a disc) exceeds span_frac x bbox diagonal is removed again.
    pymeshlab's limit is in boundary edges only, so on a fine mesh it happily caps a whole bore with a
    flat membrane. Returns (patches kept, patches removed)."""
    n0 = ms.current_mesh().face_number()
    ms.meshing_close_holes(maxholesize=maxholesize, newfaceselected=True)
    m = ms.current_mesh()
    V, F = m.vertex_matrix(), m.face_matrix()
    new = np.arange(n0, len(F))     # close_holes appends its faces
    ms.set_selection_none()
    if len(new) == 0:
        return 0, 0
    diag = float(np.linalg.norm(V.max(0) - V.min(0)))
    tm = trimesh.Trimesh(V, F[new], process=False)
    comps = trimesh.graph.connected_components(tm.face_adjacency, nodes=np.arange(len(new)), min_len=1)
    drop = []
    for c in comps:
        sub = tm.submesh([c], append=True, only_watertight=False)
        e = sub.edges_sorted
        u, cnt = np.unique(e, axis=0, return_counts=True)
        per = float(np.linalg.norm(sub.vertices[u[cnt == 1][:, 0]] - sub.vertices[u[cnt == 1][:, 1]], axis=1).sum())
        width = 2 * float(sub.area) / max(per, 1e-9)
        if width > span_frac * diag:
            drop.extend(new[c].tolist())
    if drop:
        keep = np.setdiff1d(np.arange(len(F)), drop)
        ms.add_mesh(pymeshlab.Mesh(V, F[keep]))
        ms.meshing_remove_unreferenced_vertices()
    return len(comps), len(drop)


src, dst = sys.argv[1], sys.argv[2]
target = int(sys.argv[3]) if len(sys.argv) > 3 else 60000
rec = {"input": src, "steps": []}
ms = pymeshlab.MeshSet()
ms.load_new_mesh(src)
m = ms.current_mesh()
rec["in_faces"], rec["in_vertices"] = int(m.face_number()), int(m.vertex_number())
if m.face_number() == 0:
    ms.generate_surface_reconstruction_screened_poisson(depth=9, preclean=True)
    rec["steps"].append("screened Poisson reconstruction (depth 9) from point cloud with normals")
    # remove the Poisson 'balloon' far from the points
    pts = ms.mesh(0).vertex_matrix()
    q = ms.current_mesh()
    from scipy.spatial import cKDTree
    d, _ = cKDTree(pts).query(q.vertex_matrix())
    spacing = float(np.median(cKDTree(pts).query(pts[:20000], 2)[0][:, 1]))
    bad = d > 4 * spacing
    ms.set_current_mesh(1)
    ms.compute_selection_by_condition_per_vertex(condselect=f"(q < 0)")  # no-op init
    vq = np.where(bad, -1.0, 1.0)
    mm = pymeshlab.Mesh(q.vertex_matrix(), q.face_matrix(), v_scalar_array=vq)
    ms.add_mesh(mm)
    ms.compute_selection_by_condition_per_vertex(condselect="q < 0")
    ms.meshing_remove_selected_vertices()
    rec["steps"].append(f"trimmed Poisson vertices > 4x point spacing ({spacing:.3f} mm) from the cloud: {int(bad.sum())}")
diag = float(np.linalg.norm(ms.current_mesh().bounding_box().diagonal()))
ms.meshing_merge_close_vertices(threshold=pymeshlab.PureValue(1e-3))
ms.meshing_remove_duplicate_faces()
ms.meshing_remove_duplicate_vertices()
ms.meshing_remove_connected_component_by_face_number(mincomponentsize=max(10, int(0.01 * ms.current_mesh().face_number())))
rec["steps"].append("merge vertices <1e-3 mm, remove duplicate faces, remove shells < 1% of faces")
ms.meshing_repair_non_manifold_edges()
ms.meshing_repair_non_manifold_vertices()
try:
    k_, d_ = close_small_holes(ms, 1000)
    rec["steps"].append(f"close holes <= 1000 edges and mean width <= 2% of bbox diagonal ({k_} patches, {d_} faces of oversized patches removed again)")
except Exception as e:  # noqa: BLE001
    rec["steps"].append(f"close holes failed: {e}")
ms.set_selection_none()   # close_holes leaves its new faces selected; decimation would act on those only
if ms.current_mesh().face_number() > target:
    ms.meshing_decimation_quadric_edge_collapse(targetfacenum=target, preservenormal=True)
    rec["steps"].append(f"quadric decimation to {target} faces")
ms.meshing_repair_non_manifold_edges()
k_, d_ = close_small_holes(ms, 1000)
rec["steps"].append(f"post-decimation close holes, same rule ({k_} patches, {d_} faces removed again)")
ms.save_current_mesh(dst)
t = trimesh.load(dst, process=True)
E = t.edges_sorted
_, cnt = np.unique(E, axis=0, return_counts=True)
cc = trimesh.graph.connected_components(t.face_adjacency, nodes=np.arange(len(t.faces)), min_len=1)
rec.update({"out_faces": int(len(t.faces)), "watertight": bool(t.is_watertight), "open_edges": int((cnt == 1).sum()),
            "nonmanifold_edges": int((cnt > 2).sum()), "shells": len(cc),
            "volume": float(t.volume) if t.is_watertight else None})
# decimation error (input -> output), sampled
try:
    a = trimesh.load(src, force="mesh", process=False)
    if len(getattr(a, "faces", [])):
        P = trimesh.sample.sample_surface(a, 20000, seed=0)[0]
        d = np.abs(trimesh.proximity.signed_distance(t, P)) if False else trimesh.proximity.closest_point(t, P)[1]
        rec["prep_error_mm"] = {"rms": float(np.sqrt(np.mean(d ** 2))), "p95": float(np.percentile(d, 95)), "max": float(d.max())}
except Exception as e:  # noqa: BLE001
    rec["prep_error_mm"] = str(e)
print(json.dumps(rec))
