import numpy as np
import pyvista as pv
import cgal_repair

def _surface_to_arrays(surface):
    verts = np.asarray(surface.points, dtype=np.float64)
    faces = surface.faces.reshape(-1, 4)[:, 1:4].astype(np.int32)
    return verts, faces

def _arrays_to_surface(verts, faces):
    n = len(faces)
    pv_faces = np.column_stack([np.full(n, 3, dtype=np.int32), faces]).ravel()
    return pv.PolyData(verts, pv_faces)


def _repair_cgal(surface, verbose=True):
    verts, faces = _surface_to_arrays(surface)
    new_verts, new_faces = cgal_repair.repair_mesh(verts, faces)
    result = _arrays_to_surface(new_verts, new_faces)

    return result

def _simplify_cgal(surface, ratio, verbose=True):
    verts, faces = _surface_to_arrays(surface)
    new_verts, new_faces = cgal_repair.simplify_mesh(verts, faces, ratio)
    result = _arrays_to_surface(new_verts, new_faces)

    return result

def repair_surface(surface, decimate=0.0, verbose=True):
    needs_simplify = decimate > 0
    ratio = 1.0 - decimate

    surface = _repair_cgal(surface, verbose=verbose)

    if needs_simplify:
        surface = _simplify_cgal(surface, ratio, verbose=verbose)

        if not surface.is_manifold:
            surface = _repair_cgal(surface, verbose=verbose)

    return surface