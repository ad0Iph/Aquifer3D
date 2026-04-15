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

def apply_colormap(surface, prop, cmap="viridis", log_scale=True, norm=None):
    """Aplica colormap a cell_data de una superficie."""
    scalars = surface.cell_data[prop]
    cmap_fn = plt.get_cmap(cmap)

    if norm is None:
        norm = build_norm(scalars, log_scale)

    rgba   = cmap_fn(norm(scalars))
    rgb255 = (rgba[:, :3] * 255).astype(np.uint8)
    surface.cell_data["RGB"] = rgb255
    return surface

def finalize_colors(surface):
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

def export_single(aquifer, prop, z_exag, log_scale,
                  decimate, out_file):
    grid    = aquifer.build_grid(prop=prop, z_exag=z_exag)
    surface = grid.extract_surface()
    surface = surface.triangulate()

    if not surface.is_manifold:
        surface = surface.clean(tolerance=1e-4)

    surface = apply_colormap(surface, prop, log_scale=log_scale)
    surface = finalize_colors(surface)

    print(f"{'✓ Watertight' if surface.is_manifold else '✗ No watertight'}")

    surface.save(out_file)
    print(f"Exportado: {out_file}")

def export_layers(aquifer, prop, z_exag, log_scale,
                  decimate, out_dir):
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
        surface = finalize_colors(surface)

        print(f"Capa {int(layer):02d} — watertight: {surface.is_manifold}")

        out_path = f"{out_dir}/layer_{int(layer):02d}.ply"
        surface.save(out_path)
        print(f"  → {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="exportModelM6")
    parser.add_argument("model_workspace")
    parser.add_argument("--prop",       default="k")
    parser.add_argument("--z-exag",     type=float, default=1.0)
    parser.add_argument("--no-log",     action="store_true")
    parser.add_argument("--layers",     action="store_true", help="Exportar una superficie por capa")
    parser.add_argument("--cuts",       type=int,   default=None, help="Subdividir en NxN bloques espaciales")
    parser.add_argument("--split-n",    type=int,   default=3)
    parser.add_argument("--out",        default="aquifer.ply")
    parser.add_argument("--out-dir",    default="layers")
    args = parser.parse_args()

    log_scale = not args.no_log
    kwargs    = dict(prop=args.prop, z_exag=args.z_exag, log_scale=log_scale, decimate=args.decimate)

    aquifer = AquiferGridM6(args.model_workspace)
    aquifer.load()

    if args.cuts:
        targets = aquifer.splitN(args.cuts)
    else:
        targets = {"model": aquifer}

    for name, sub in targets.items():
        if args.layers:
            export_layers(sub, out_dir=f"{args.out_dir}/{name}", **kwargs)
        else:
            out = args.out if name == "model" else f"{args.out_dir}/{name}.ply"
            export_single(sub, out_file=out, **kwargs)