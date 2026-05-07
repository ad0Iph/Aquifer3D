import numpy as np
import pyvista as pv
import cgal_repair


def _has_cgal():
    try:
        import cgal_repair
        return True
    except ImportError:
        return False


def _surface_to_arrays(surface):
    """Extrae vértices (N,3) y caras (M,3) de un PolyData triangulado."""
    verts = np.asarray(surface.points, dtype=np.float64)
    faces = surface.faces.reshape(-1, 4)[:, 1:4].astype(np.int32)
    return verts, faces


def _arrays_to_surface(verts, faces):
    """Reconstruye un PolyData desde vértices y caras."""
    n = len(faces)
    pv_faces = np.column_stack([np.full(n, 3, dtype=np.int32), faces]).ravel()
    return pv.PolyData(verts, pv_faces)


# ── CGAL ─────────────────────────────────────────────────────────────

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


def _repair_pyvista(surface, verbose=True):
    surface = surface.clean(tolerance=1e-6)

    if not surface.is_manifold:
        bounds = surface.bounds
        diagonal = np.sqrt(
            (bounds[1] - bounds[0])**2 +
            (bounds[3] - bounds[2])**2 +
            (bounds[5] - bounds[4])**2
        )
        surface = surface.fill_holes(diagonal * 0.1)
        surface = surface.triangulate()

    surface.compute_normals(cell_normals=True, point_normals=False,
                            auto_orient_normals=True, inplace=True)

    return surface


# ── Función principal ────────────────────────────────────────────────

def repair_surface(surface, decimate=0.0, verbose=True):
    needs_simplify = decimate > 0
    ratio = 1.0 - decimate

    if _has_cgal():
        surface = _repair_cgal(surface, verbose=verbose)

        if needs_simplify:
            surface = _simplify_cgal(surface, ratio, verbose=verbose)

            if not surface.is_manifold:
                surface = _repair_cgal(surface, verbose=verbose)

    else:
        surface = _repair_pyvista(surface, verbose=verbose)

        if needs_simplify:
            surface = surface.decimate(decimate)

            if not surface.is_manifold:
                surface = _repair_pyvista(surface, verbose=verbose)

    return surface