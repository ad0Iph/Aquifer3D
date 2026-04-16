import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import argparse
from pathlib import Path
from aquifer_grid_m6 import AquiferGridM6


def build_norm(scalars, log_scale):
    if log_scale:
        vmin = np.nanmin(scalars[scalars > 0])
        vmax = np.nanmax(scalars)
        return mcolors.LogNorm(vmin=vmin, vmax=vmax)
    return mcolors.Normalize(vmin=np.nanmin(scalars), vmax=np.nanmax(scalars))


def apply_colormap(mesh, prop, cmap="viridis", log_scale=True, norm=None):
    """Aplica colormap a cell_data de un mesh (grid o superficie)."""
    scalars = mesh.cell_data[prop]
    cmap_fn = plt.get_cmap(cmap)

    if norm is None:
        norm = build_norm(scalars, log_scale)

    rgba   = cmap_fn(norm(scalars))
    rgb255 = (rgba[:, :3] * 255).astype(np.uint8)
    mesh.cell_data["RGB"] = rgb255
    return mesh


def finalize_colors_ply(surface):
    """
    Convierte cell_data RGB → point_data con canales separados (red, green, blue).
    MeshLab y la mayoría de visores PLY esperan este formato.
    """
    surface = surface.cell_data_to_point_data()
    rgb = np.clip(surface.point_data["RGB"], 0, 255).astype(np.uint8)

    surface.point_data["red"]   = rgb[:, 0]
    surface.point_data["green"] = rgb[:, 1]
    surface.point_data["blue"]  = rgb[:, 2]
    del surface.point_data["RGB"]

    return surface


# ── Exportación como superficie triangulada (PLY) ───────────────────

def export_surface(aquifer, prop, z_exag, log_scale, out_file):
    """Exporta el modelo como superficie triangulada con colores (.ply)."""
    grid    = aquifer.build_grid(prop=prop, z_exag=z_exag)
    surface = grid.extract_surface()
    surface = surface.triangulate()

    if not surface.is_manifold:
        surface = surface.clean(tolerance=1e-4)

    surface = apply_colormap(surface, prop, log_scale=log_scale)
    surface = finalize_colors_ply(surface)

    print(f"  {'✓ Watertight' if surface.is_manifold else '✗ No watertight'}")
    surface.save(out_file)
    print(f"  → {out_file}")


def export_surface_layers(aquifer, prop, z_exag, log_scale, out_dir):
    """Exporta una superficie PLY por capa geológica."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    grid      = aquifer.build_grid(prop=prop, z_exag=z_exag)
    scalars   = grid.cell_data[prop]
    layer_ids = grid.cell_data["layer"]
    norm      = build_norm(scalars, log_scale)

    for layer in np.unique(layer_ids):
        indices = np.where(layer_ids == layer)[0]
        sub     = grid.extract_cells(indices)

        surface = sub.extract_surface()
        surface = surface.triangulate()

        if not surface.is_manifold:
            surface = surface.clean(tolerance=1e-4)

        surface = apply_colormap(surface, prop, log_scale=log_scale, norm=norm)
        surface = finalize_colors_ply(surface)

        print(f"  Capa {int(layer):02d} — watertight: {surface.is_manifold}")

        out_path = f"{out_dir}/layer_{int(layer):02d}.ply"
        surface.save(out_path)
        print(f"    → {out_path}")



def export_cubes(aquifer, prop, z_exag, log_scale, out_file):
    """Exporta el grid completo de hexaedros con colores (.vtk)."""
    grid = aquifer.build_grid(prop=prop, z_exag=z_exag)
    grid = apply_colormap(grid, prop, log_scale=log_scale)

    grid.save(out_file)
    print(f"  → {out_file}  ({grid.n_cells} celdas)")


def export_cubes_layers(aquifer, prop, z_exag, log_scale, out_dir):
    """Exporta un vtk por capa geológica (hexaedros)."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    grid      = aquifer.build_grid(prop=prop, z_exag=z_exag)
    scalars   = grid.cell_data[prop]
    layer_ids = grid.cell_data["layer"]
    norm      = build_norm(scalars, log_scale)

    for layer in np.unique(layer_ids):
        indices = np.where(layer_ids == layer)[0]
        sub     = grid.extract_cells(indices)

        sub = apply_colormap(sub, prop, log_scale=log_scale, norm=norm)

        out_path = f"{out_dir}/layer_{int(layer):02d}.ply"
        sub.save(out_path)
        print(f"  Capa {int(layer):02d} → {out_path}  ({sub.n_cells} celdas)")


# ── Main ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="exportModelM6",
        description="Exporta un modelo MODFLOW 6 como malla 3D.",
    )
    parser.add_argument("model_workspace")
    parser.add_argument("--prop",    default="k")
    parser.add_argument("--z-exag",  type=float, default=1.0)
    parser.add_argument("--no-log",  action="store_true")

    # Modo de geometría
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--cubes",   action="store_true",
                      help="Exportar hexaedros completos (.vtk)")
    mode.add_argument("--surface", action="store_true", default=True,
                      help="Exportar superficie triangulada (.ply) [default]")

    parser.add_argument("--layers", action="store_true",
                        help="Exportar un archivo por capa geológica")
    parser.add_argument("--cuts",   type=int, default=None,
                        help="Subdividir en NxN bloques espaciales")

    parser.add_argument("--out",     default=None,
                        help="Archivo de salida (default: aquifer.ply o aquifer.vtk)")
    parser.add_argument("--out-dir", default="layers",
                        help="Directorio para capas/bloques")

    args = parser.parse_args()

    log_scale = not args.no_log
    use_cubes = args.cubes
    ext       = ".vtk" if use_cubes else ".ply"
    out_file  = args.out or f"aquifer{ext}"

    aquifer = AquiferGridM6(args.model_workspace)
    aquifer.load()

    # Determinar qué subgrids procesar
    if args.cuts:
        targets = aquifer.splitN(args.cuts)
    else:
        targets = {"model": aquifer}

    for name, sub in targets.items():
        print(f"[{name}]")
        out = out_file if name == "model" else f"{args.out_dir}/{name}{ext}"
        sub_dir = f"{args.out_dir}/{name}" if name != "model" else args.out_dir

        if args.layers:
            if use_cubes:
                export_cubes_layers(sub, args.prop, args.z_exag, log_scale, sub_dir)
            else:
                export_surface_layers(sub, args.prop, args.z_exag, log_scale, sub_dir)
        else:
            if use_cubes:
                export_cubes(sub, args.prop, args.z_exag, log_scale, out)
            else:
                export_surface(sub, args.prop, args.z_exag, log_scale, out)