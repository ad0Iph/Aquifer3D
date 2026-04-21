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
    scalars = mesh.cell_data[prop]
    cmap_fn = plt.get_cmap(cmap)

    if norm is None:
        norm = build_norm(scalars, log_scale)

    rgba   = cmap_fn(norm(scalars))
    rgb255 = (rgba[:, :3] * 255).astype(np.uint8)
    mesh.cell_data["RGB"] = rgb255
    return mesh


def finalize_colors_ply(surface):
    surface = surface.cell_data_to_point_data()
    rgb = np.clip(surface.point_data["RGB"], 0, 255).astype(np.uint8)

    surface.point_data["red"]   = rgb[:, 0]
    surface.point_data["green"] = rgb[:, 1]
    surface.point_data["blue"]  = rgb[:, 2]
    del surface.point_data["RGB"]

    return surface


def format_value(v):
    if v == 0:
        return "0"
    elif abs(v) < 1e-3 or abs(v) >= 1e4:
        return f"{v:.2e}"
    else:
        return f"{v:.4g}"

def export_surface(grid, prop, log_scale, out_file, norm=None,
                   clean=False, min_ratio=0.01):
    surface = grid.extract_surface()
    surface = surface.triangulate()

    if not surface.is_manifold:
        surface = surface.clean(tolerance=1e-4)

    if clean:
        surface = AquiferGridM6.clean_surface(
            surface, min_ratio=min_ratio, remove_enclosed=True)
        surface = surface.triangulate()

    surface = apply_colormap(surface, prop, log_scale=log_scale, norm=norm)
    surface = finalize_colors_ply(surface)

    surface.save(out_file)
    print(f"  → {out_file}")


def export_surface_layers(grid, prop, log_scale, out_dir, norm=None,
                          clean=False, min_ratio=0.01):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    layer_ids = grid.cell_data["layer"]
    if norm is None:
        norm = build_norm(grid.cell_data[prop], log_scale)

    for layer in np.unique(layer_ids):
        sub = grid.extract_cells(np.where(layer_ids == layer)[0])

        surface = sub.extract_surface()
        surface = surface.triangulate()

        if not surface.is_manifold:
            surface = surface.clean(tolerance=1e-4)

        if clean:
            surface = AquiferGridM6.clean_surface(
                surface, min_ratio=min_ratio, remove_enclosed=True)
            surface = surface.triangulate()

        surface = apply_colormap(surface, prop, log_scale=log_scale, norm=norm)
        surface = finalize_colors_ply(surface)

        out_path = f"{out_dir}/layer_{int(layer):02d}.ply"
        surface.save(out_path)
        print(f"    → {out_path}")

def export_cubes(grid, prop, log_scale, out_file, norm=None):
    grid = apply_colormap(grid, prop, log_scale=log_scale, norm=norm)
    grid.save(out_file)
    print(f"  → {out_file}  ({grid.n_cells} celdas)")


def export_cubes_layers(grid, prop, log_scale, out_dir, norm=None):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    layer_ids = grid.cell_data["layer"]
    if norm is None:
        norm = build_norm(grid.cell_data[prop], log_scale)

    for layer in np.unique(layer_ids):
        sub = grid.extract_cells(np.where(layer_ids == layer)[0])
        sub = apply_colormap(sub, prop, log_scale=log_scale, norm=norm)

        out_path = f"{out_dir}/layer_{int(layer):02d}.vtk"
        sub.save(out_path)
        print(f"    → {out_path}  ({sub.n_cells} celdas)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="exportModelM6",
        description="Exporta un modelo MODFLOW 6 como malla 3D.",
    )
    parser.add_argument("model_workspace")
    parser.add_argument("--prop",         default="k")
    parser.add_argument("--z-exag",       type=float, default=1.0)
    parser.add_argument("--no-log",       action="store_true")

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--cubes",   action="store_true",
                      help="Exportar hexaedros (.vtk)")
    mode.add_argument("--surface", action="store_true", default=True,
                      help="Exportar superficie triangulada (.ply) [default]")

    parser.add_argument("--layers",       action="store_true",
                        help="Un archivo por capa geológica")
    parser.add_argument("--cuts",         type=int, default=None,
                        help="Subdividir en NxN bloques espaciales")
    parser.add_argument("--split-unique", action="store_true",
                        help="Un archivo por cada valor distinto de la propiedad")
    parser.add_argument("--clean",        action="store_true",
                        help="Eliminar componentes pequeños y encerrados (para impresión 3D)")
    parser.add_argument("--min-ratio",    type=float, default=0.01,
                        help="Fracción mínima respecto al mayor componente (default: 0.01 = 1%%)")

    parser.add_argument("--out",     default=None)
    parser.add_argument("--out-dir", default="export")

    args = parser.parse_args()

    log_scale = not args.no_log
    use_cubes = args.cubes
    ext       = ".vtk" if use_cubes else ".ply"
    out_file  = args.out or f"aquifer{ext}"
    clean_kw  = dict(clean=args.clean, min_ratio=args.min_ratio)

    aquifer = AquiferGridM6(args.model_workspace)
    aquifer.load()

    if args.cuts:
        targets = aquifer.splitN(args.cuts)
    else:
        targets = {"model": aquifer}

    for name, sub in targets.items():
        print(f"\n[{name}]")

        grid = sub.build_grid(prop=args.prop, z_exag=args.z_exag)
        norm = build_norm(grid.cell_data[args.prop], log_scale)

        if args.split_unique:
            segments = AquiferGridM6.split_grid_by_unique(grid, prop=args.prop)
            seg_dir = f"{args.out_dir}/{name}" if name != "model" else args.out_dir
            Path(seg_dir).mkdir(parents=True, exist_ok=True)

            print(f"  {len(segments)} valores distintos de '{args.prop}'")

            for val, seg_grid in segments.items():
                fname = f"{args.prop}_{format_value(val)}"
                print(f"\n  [{fname}]  ({seg_grid.n_cells} celdas)")

                if args.layers:
                    layer_dir = f"{seg_dir}/{fname}"
                    if use_cubes:
                        export_cubes_layers(seg_grid, args.prop, log_scale,
                                            layer_dir, norm=norm)
                    else:
                        export_surface_layers(seg_grid, args.prop, log_scale,
                                              layer_dir, norm=norm, **clean_kw)
                else:
                    seg_out = f"{seg_dir}/{fname}{ext}"
                    if use_cubes:
                        export_cubes(seg_grid, args.prop, log_scale,
                                     seg_out, norm=norm)
                    else:
                        export_surface(seg_grid, args.prop, log_scale,
                                       seg_out, norm=norm, **clean_kw)

        else:
            out = out_file if name == "model" else f"{args.out_dir}/{name}{ext}"

            if args.layers:
                sub_dir = f"{args.out_dir}/{name}" if name != "model" else args.out_dir
                if use_cubes:
                    export_cubes_layers(grid, args.prop, log_scale, sub_dir, norm=norm)
                else:
                    export_surface_layers(grid, args.prop, log_scale, sub_dir,
                                          norm=norm, **clean_kw)
            else:
                if use_cubes:
                    export_cubes(grid, args.prop, log_scale, out, norm=norm)
                else:
                    export_surface(grid, args.prop, log_scale, out,
                                   norm=norm, **clean_kw)