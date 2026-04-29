import numpy as np
import pyvista as pv
import cgal_repair

def _surface_to_arrays(surface):
    verts = np.asarray(surface.points, dtype=np.float64)
    faces_flat = surface.faces
    faces = faces_flat.reshape(-1, 4)[:, 1:4].astype(np.int32)
    return verts, faces

def _arrays_to_surface(verts, faces):
    n_faces = len(faces)
    pv_faces = np.column_stack([
        np.full(n_faces, 3, dtype=np.int32),
        faces
    ]).ravel()
    return pv.PolyData(verts, pv_faces)

def _repair_cgal(surface):
    verts, faces = _surface_to_arrays(surface)
    new_verts, new_faces = cgal_repair.repair_mesh(verts, faces)
    repaired = _arrays_to_surface(new_verts, new_faces)

    return repaired

def _simplify_cgal(surface, ratio):
    verts, faces = _surface_to_arrays(surface)
    new_verts, new_faces = cgal_repair.simplify_mesh(verts, faces, ratio)
    simplified = _arrays_to_surface(new_verts, new_faces)

    return simplified

def _repair_and_simplify_cgal(surface, ratio):
    verts, faces = _surface_to_arrays(surface)
    new_verts, new_faces = cgal_repair.repair_and_simplify(verts, faces, ratio)
    result = _arrays_to_surface(new_verts, new_faces)

    return result

def repair_surface(surface, decimate=0.0):
    needs_repair = not surface.is_manifold
    needs_simplify = decimate > 0
    ratio = 1.0 - decimate  # decimate=0.7 → ratio=0.3 (mantener 30%)

    if not needs_repair and not needs_simplify:
        return surface

    if needs_repair and needs_simplify:
        return _repair_and_simplify_cgal(surface, ratio)
    elif needs_repair:
        return _repair_cgal(surface)
    else:
        return _simplify_cgal(surface, ratio)