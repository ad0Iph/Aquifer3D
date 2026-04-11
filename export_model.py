import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import argparse
from pathlib import Path
from aquifer_grid import AquiferGrid


def build_norm(scalars, log_scale):
    """
    Construye un normalizador de Matplotlib para una propiedad dada, considerando si se desea escala logarítmica o lineal.
    """
    if log_scale:
        vmin = np.nanmin(scalars[scalars > 0])
        vmax = np.nanmax(scalars)
        return mcolors.LogNorm(vmin=vmin, vmax=vmax)
    return mcolors.Normalize(vmin=np.nanmin(scalars), vmax=np.nanmax(scalars))


def apply_colormap(grid, prop, cmap="viridis", log_scale=True, norm=None):
    """
     Aplica un colormap a un grid de PyVista basado en los valores de una propiedad dada.
    """
    scalars = grid.cell_data[prop]
    cmap_fn = plt.get_cmap(cmap)
    if norm is None:
        if log_scale:
            vmin = np.nanmin(scalars[scalars > 0])
            vmax = np.nanmax(scalars)
            norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
        else:
            norm = mcolors.Normalize(vmin=np.nanmin(scalars), vmax=np.nanmax(scalars))
    rgba   = cmap_fn(norm(scalars))
    rgb255 = (rgba[:, :3] * 255).astype(np.uint8)
    grid.cell_data["RGB"] = rgb255
    return grid


def surface_from_grid(grid, decimate=0.0):
    """
    Extrae la superficie de un grid de PyVista y opcionalmente aplica decimación para reducir el número de triángulos.
    """
    surface = grid.extract_surface(algorithm='dataset_surface')
    surface = surface.triangulate()
    if decimate > 0:
        surface = surface.decimate(decimate)
    return surface

def export_single(aquifer, prop, z_exag, xy_scale, log_scale,
                  decimate, out_file):
    """
    Exporta una única superficie del modelo completo a un archivo PLY, aplicando colormap y opcionalmente decimación.
    """
    grid    = aquifer.build_grid(prop=prop, z_exag=z_exag, xy_scale=xy_scale)
    surface = surface_from_grid(grid, decimate)

    # Limpiar ANTES de colorear
    if not surface.is_manifold:
        surface = surface.clean(tolerance=1e-4)

    surface = apply_colormap(surface, prop, log_scale=log_scale)

    # Convertir a point data y forzar uint8
    surface = surface.cell_data_to_point_data()
    rgb = np.clip(surface.point_data["RGB"], 0, 255).astype(np.uint8)
    surface.point_data["RGB"] = rgb

    surface.save(out_file)
    print(f"Exportado: {out_file}")

def export_layers(aquifer, prop, z_exag, xy_scale, log_scale,
                  decimate, out_dir):
    """
    Exporta una superficie por cada capa del modelo a archivos PLY separados, aplicando colormap y opcionalmente decimación.
    """
    Path(out_dir).mkdir(exist_ok=True)

    grid      = aquifer.build_grid(prop=prop, z_exag=z_exag, xy_scale=xy_scale)
    scalars   = grid.cell_data[prop]
    layer_ids = grid.cell_data["layer"]
    norm      = build_norm(scalars, log_scale)

    for layer in np.unique(layer_ids):
        indices = np.where(layer_ids == layer)[0]
        sub     = grid.extract_cells(indices)

        surface = surface_from_grid(sub, decimate)

        # Limpiar ANTES de colorear
        if not surface.is_manifold:
            surface = surface.clean(tolerance=1e-4)

        surface = apply_colormap(surface, prop, log_scale=log_scale, norm=norm)

        # Convertir a point data y forzar uint8
        surface = surface.cell_data_to_point_data()
        rgb = np.clip(surface.point_data["RGB"], 0, 255).astype(np.uint8)
        surface.point_data["RGB"] = rgb

        print(f"Capa {int(layer):02d} — watertight: {surface.is_manifold}")

        out_path = f"{out_dir}/layer_{int(layer):02d}.ply"
        surface.save(out_path)
        print(f"  → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="exportModelM6")
    parser.add_argument("model_workspace")
    parser.add_argument("simulation_name")
    parser.add_argument("--prop",       default="k")
    parser.add_argument("--z-exag",     type=float, default=1.0)
    parser.add_argument("--xy-scale",   type=float, default=1.0)
    parser.add_argument("--no-log",     action="store_true")
    parser.add_argument("--decimate",   type=float, default=0.0)
    parser.add_argument("--layers",     action="store_true",
                        help="Exportar una superficie por capa")
    parser.add_argument("--cuts",       type=int,   default=None,
                        help="Subdividir en NxN bloques espaciales")
    parser.add_argument("--split-prop", default=None,
                        help="Subdividir por rangos de propiedad")
    parser.add_argument("--split-n",    type=int,   default=3)
    parser.add_argument("--out",        default="aquifer.ply")
    parser.add_argument("--out-dir",    default="layers")
    args = parser.parse_args()

    log_scale = not args.no_log
    kwargs    = dict(prop=args.prop, z_exag=args.z_exag,
                     xy_scale=args.xy_scale, log_scale=log_scale,
                     decimate=args.decimate)

    aquifer = AquiferGrid(args.model_workspace, args.simulation_name)
    aquifer.load()

    if args.cuts:
        targets = aquifer.splitN(args.cuts)
    elif args.split_prop:
        targets = aquifer.split_by_property(args.split_prop, args.split_n)
    else:
        targets = {"model": aquifer}

    for name, sub in targets.items():
        if args.layers:
            export_layers(sub, out_dir=f"{args.out_dir}/{name}", **kwargs)
        else:
            out = args.out if name == "model" else f"{args.out_dir}/{name}.ply"
            export_single(sub, out_file=out, **kwargs)