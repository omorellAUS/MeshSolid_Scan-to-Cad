"""Mesh export for the engine: Fusion PolygonMesh arrays (cm) -> binary STL (mm).  No adsk import, so it
is unit-testable outside Fusion."""
import struct


def triangles_from_polygon_mesh(coords, tri_idx, quad_idx=(), poly_idx=(), poly_counts=()):
    """coords: flat [x0, y0, z0, x1, ...]; tri_idx flat triples; quad_idx flat quads (split 0-1-2 / 0-2-3);
    general polygons as flat indices with per-polygon counts (fanned).  Returns a list of index triples."""
    tris = [tuple(tri_idx[i:i + 3]) for i in range(0, len(tri_idx), 3)]
    for i in range(0, len(quad_idx), 4):
        a, b, c, d = quad_idx[i:i + 4]
        tris += [(a, b, c), (a, c, d)]
    k = 0
    for n in poly_counts:
        p = poly_idx[k:k + n]; k += n
        tris += [(p[0], p[j], p[j + 1]) for j in range(1, n - 1)]
    return tris


def write_binary_stl(path, coords, tris, scale=10.0):
    """coords flat in cm (Fusion internal units) -> STL in mm (scale 10)."""
    pts = [(coords[i] * scale, coords[i + 1] * scale, coords[i + 2] * scale) for i in range(0, len(coords), 3)]
    with open(path, "wb") as f:
        f.write(b"meshsolid fusion export".ljust(80, b" "))
        f.write(struct.pack("<I", len(tris)))
        for a, b, c in tris:
            p, q, r = pts[a], pts[b], pts[c]
            u = (q[0] - p[0], q[1] - p[1], q[2] - p[2]); v = (r[0] - p[0], r[1] - p[1], r[2] - p[2])
            n = (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])
            L = (n[0] ** 2 + n[1] ** 2 + n[2] ** 2) ** 0.5 or 1.0
            f.write(struct.pack("<12fH", n[0] / L, n[1] / L, n[2] / L, *p, *q, *r, 0))
    return len(tris)
