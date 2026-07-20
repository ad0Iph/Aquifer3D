"""
compare_repair.py — Compara la reparación de mallas con CGAL vs pymeshfix.

Carga un modelo MODFLOW 6, extrae la superficie de un grupo (o del modelo
completo), la repara con ambos métodos y muestra los resultados en
ventanas separadas (cerrar una ventana abre la siguiente).

Requiere:
    pip install pymeshfix

Uso:
    python compare_repair.py ..\\modelo --z-exag 2.5
    python compare_repair.py ..\\modelo --z-exag 2.5 --value 1e-06
"""

import argparse
import numpy as np
import pyvista as pv
from aquifer_grid_m6 import AquiferGridM6

try:
    import cgal_repair
    HAS_CGAL = True
except ImportError:
    HAS_CGAL = False

try:
    import pymeshfix
    HAS_PYMESHFIX = True
except ImportError:
    HAS_PYMESHFIX = False


# ── Conversión PyVista ↔ arrays ──────────────────────────────────────

def surface_to_arrays(surface):
    """Extrae vértices (N,3) float64 y caras (M,3) int32 de un PolyData."""
    verts = np.asarray(surface.points, dtype=np.float64)
    faces = surface.faces.reshape(-1, 4)[:, 1:4].astype(np.int32)
    return verts, faces


def arrays_to_surface(verts, faces):
    """Reconstruye un PolyData desde vértices y caras trianguladas."""
    n = len(faces)
    pv_faces = np.column_stack(
        [np.full(n, 3, dtype=np.int64), faces]).ravel()
    return pv.PolyData(verts, faces=pv_faces)


# ── Métodos de reparación ────────────────────────────────────────────

def repair_with_cgal(surface):
    """Reparación con el módulo propio (pipeline PMP de CGAL)."""
    verts, faces = surface_to_arrays(surface)
    new_verts, new_faces = cgal_repair.repair_mesh(verts, faces)
    return arrays_to_surface(new_verts, new_faces)


def repair_with_pymeshfix(surface):
    """
    Reparación con pymeshfix (wrapper de MeshFix, Attene 2010).

    MeshFix garantiza una malla watertight con una estrategia agresiva:
    por defecto elimina las componentes conexas pequeñas
    (remove_smallest_components=True) y rellena todos los agujeros.

    API pymeshfix >= 0.15: los resultados quedan en .points y .faces,
    y repair() acepta joincomp / remove_smallest_components.
    """
    verts, faces = surface_to_arrays(surface)
    fixer = pymeshfix.MeshFix(verts, faces)
    fixer.repair()
    return arrays_to_surface(fixer.points, fixer.faces.astype(np.int32))


# ── Estadísticas ─────────────────────────────────────────────────────

def mesh_stats(surface, label):
    """Imprime y retorna estadísticas de la malla."""
    open_edges = surface.extract_feature_edges(
        boundary_edges=True, feature_edges=False,
        manifold_edges=False, non_manifold_edges=False)
    n_open = open_edges.n_cells

    stats = {
        "caras": surface.n_cells,
        "vertices": surface.n_points,
        "manifold": surface.is_manifold,
        "aristas_abiertas": n_open,
    }
    print(f"\n  [{label}]")
    print(f"    Caras:            {stats['caras']}")
    print(f"    Vértices:         {stats['vertices']}")
    print(f"    Manifold:         {'✓' if stats['manifold'] else '✗'}")
    print(f"    Aristas abiertas: {stats['aristas_abiertas']}")
    return stats


# ── Visualización ────────────────────────────────────────────────────

def show_mesh(surface, title, stats, color="tan"):
    """Muestra una malla en su propia ventana con sus estadísticas."""
    p = pv.Plotter(window_size=[1000, 750])
    p.background_color = "white"
    p.add_mesh(surface, color=color, show_edges=True,
               edge_color="grey", line_width=0.5)
    p.add_text(title, position="upper_edge", font_size=12, color="black")
    p.add_text(
        f"Caras: {stats['caras']}   "
        f"Manifold: {'si' if stats['manifold'] else 'NO'}   "
        f"Aristas abiertas: {stats['aristas_abiertas']}",
        position=(10, 10), font_size=9, color="black")
    p.add_axes(xlabel="X (m)", ylabel="Y (m)", zlabel="Z (m)")
    p.show(title=title)


# ── Main ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="compareRepair",
        description="Compara reparación CGAL vs pymeshfix.")
    parser.add_argument("model_workspace")
    parser.add_argument("--prop",   default="k")
    parser.add_argument("--z-exag", type=float, default=1.0)
    parser.add_argument("--value",  type=float, default=None,
                        help="Valor del grupo a reparar (ej: 1e-06). "
                             "Si se omite, usa el grupo con más celdas.")

    args = parser.parse_args()

    if not HAS_CGAL:
        print("⚠ cgal_repair no disponible — se omite la vía CGAL")
    if not HAS_PYMESHFIX:
        print("⚠ pymeshfix no instalado — pip install pymeshfix")
    if not HAS_CGAL and not HAS_PYMESHFIX:
        raise SystemExit("No hay ningún método de reparación disponible.")

    # ── 1. Cargar modelo y extraer la superficie de un grupo ────────
    aquifer = AquiferGridM6(args.model_workspace)
    aquifer.load()
    grid = aquifer.build_grid(prop=args.prop, z_exag=args.z_exag)

    values = grid.cell_data[args.prop]

    if args.value is not None:
        target = args.value
    else:
        # Grupo con más celdas
        uniques, counts = np.unique(
            values[~np.isnan(values)], return_counts=True)
        target = uniques[np.argmax(counts)]

    indices = np.where(values == target)[0]
    print(f"Grupo {args.prop}={target:g}: {indices.size} celdas")

    sub = grid.extract_cells(indices)
    original = sub.extract_surface().triangulate()

    # ── 2. Reparar con ambos métodos ─────────────────────────────────
    stats_orig = mesh_stats(original, "Original (sin reparar)")

    results = []
    if HAS_CGAL:
        print("\nReparando con CGAL...")
        repaired_cgal = repair_with_cgal(original.copy())
        stats_cgal = mesh_stats(repaired_cgal, "CGAL (pipeline PMP)")
        results.append(("Reparación CGAL", repaired_cgal,
                        stats_cgal, "lightsteelblue"))

    if HAS_PYMESHFIX:
        print("\nReparando con pymeshfix...")
        repaired_pmf = repair_with_pymeshfix(original.copy())
        stats_pmf = mesh_stats(repaired_pmf, "pymeshfix (MeshFix)")
        results.append(("Reparación pymeshfix", repaired_pmf,
                        stats_pmf, "lightsalmon"))

    # ── 3. Mostrar en ventanas separadas ─────────────────────────────
    # (cerrar cada ventana abre la siguiente)
    print("\nMostrando resultados — cierra cada ventana para ver la siguiente.")
    show_mesh(original, "Original (sin reparar)", stats_orig, color="tan")
    for title, surface, stats, color in results:
        show_mesh(surface, title, stats, color=color)