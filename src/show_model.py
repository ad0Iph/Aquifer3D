import pyvista as pv
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from aquifer_grid_m6 import AquiferGridM6


def apply_colormap(surface, prop, cmap="viridis", log_scale=True, norm=None):
    scalars = surface.cell_data[prop]
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
    surface.cell_data["RGB"] = rgb255
    return surface


def prepare_colored_surface(grid, prop, log_scale=True, cmap="viridis",
                            norm=None, decimate=0.0):
    surface = grid.extract_surface()
    surface = surface.triangulate()

    if decimate > 0:
        surface = surface.decimate(decimate)

    if not surface.is_manifold:
        surface = surface.clean(tolerance=1e-4)

    surface = apply_colormap(surface, prop, cmap=cmap, log_scale=log_scale, norm=norm)

    surface = surface.cell_data_to_point_data()
    rgb = np.clip(surface.point_data["RGB"], 0, 255).astype(np.uint8)

    surface.point_data["red"]   = rgb[:, 0]
    surface.point_data["green"] = rgb[:, 1]
    surface.point_data["blue"]  = rgb[:, 2]
    del surface.point_data["RGB"]

    return surface


def finalize_colors_ply(surface):
    """Convierte cell_data RGB → point_data con canales separados."""
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


def visualizeModflow(model_ws, prop="k", showGrid=False,
                     z_exag=1.0, log_scale=True, export=None,
                     clip=False, cuts=None, split_unique=False,
                     clean=False, min_ratio=0.01):
    aquifer = AquiferGridM6(model_ws)
    aquifer.load()

    mesh_kwargs = dict(
        scalars=prop, cmap="viridis",
        show_edges=showGrid, log_scale=log_scale,
        scalar_bar_args={"title": f"{'log10(' + prop + ')' if log_scale else prop}"},
    )

    if split_unique:
        grid = aquifer.build_grid(prop=prop, z_exag=z_exag)
        segments = AquiferGridM6.split_grid_by_unique(grid, prop=prop)

        print(f"Detectados {len(segments)} valores distintos de '{prop}':")

        surfaces = {}
        for val, seg_grid in segments.items():
            surface = seg_grid.extract_surface().triangulate()
            if not surface.is_manifold:
                surface = surface.clean(tolerance=1e-4)
            if clean:
                surface = AquiferGridM6.clean_surface(
                    surface, min_ratio=min_ratio, remove_enclosed=True)
                surface = surface.triangulate()
            surfaces[val] = surface
            print(f"  {format_value(val):>12s}  →  {surface.n_cells} caras")

        p = pv.Plotter()
        p.background_color = "white"

        n = len(surfaces)
        cmap_cb = plt.get_cmap("tab10") if n <= 10 else plt.get_cmap("tab20")
        colors = [cmap_cb(i / max(n - 1, 1))[:3] for i in range(n)]

        checkbox_size = 25
        y_offset = 10

        for idx, (val, surface) in enumerate(surfaces.items()):
            color = colors[idx]

            actor = p.add_mesh(
                surface,
                color=color,
                show_edges=showGrid,
                opacity=1.0,
            )

            text = f"{prop} = {format_value(val)}  ({surface.n_cells})"

            def make_callback(a):
                def callback(state):
                    a.SetVisibility(state)
                return callback

            y_pos = y_offset + idx * (checkbox_size + 10)
            p.add_checkbox_button_widget(
                make_callback(actor),
                value=True,
                position=(10, y_pos),
                size=checkbox_size,
                color_on=color,
                color_off="grey",
            )
            p.add_text(
                text,
                position=(10 + checkbox_size + 8, y_pos + 2),
                font_size=8,
                color="black",
            )

        p.add_axes(xlabel="X (m)", ylabel="Y (m)", zlabel="Z (m)")

        if export:
            from pathlib import Path
            out_dir = export.rsplit(".", 1)[0] if "." in export else export
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            for val, surface in surfaces.items():
                out_path = f"{out_dir}/{prop}_{format_value(val)}.ply"
                colored = apply_colormap(surface, prop, log_scale=log_scale)
                colored = finalize_colors_ply(colored)
                colored.save(out_path)
                print(f"Exportado: {out_path}")

        p.show(title=f"{prop}: {n} valores únicos")
        return

    if cuts:
        subgrids = aquifer.splitN(cuts)

        all_vals = np.concatenate([
            sub.properties[prop].ravel() for sub in subgrids.values()
        ])
        all_vals = all_vals[~np.isnan(all_vals)]
        if log_scale:
            norm = mcolors.LogNorm(vmin=all_vals[all_vals > 0].min(),
                                   vmax=all_vals.max())
        else:
            norm = mcolors.Normalize(vmin=all_vals.min(), vmax=all_vals.max())

        for name, sub in subgrids.items():
            p = pv.Plotter()
            p.background_color = "white"

            grid = sub.build_grid(prop=prop, z_exag=z_exag)

            if clip:
                p.add_mesh_clip_plane(grid, **mesh_kwargs)
            else:
                p.add_mesh(grid, **mesh_kwargs)

            p.add_axes(xlabel="X (m)", ylabel="Y (m)", zlabel="Z (m)")

            if export:
                out_path = f"{export}_{name}.ply"
                surface = prepare_colored_surface(grid, prop,
                                                  log_scale=log_scale, norm=norm)
                surface.save(out_path)
                print(f"Exportado: {out_path}")

            p.show(title=f"Bloque: {name}")

    else:
        p = pv.Plotter()
        p.background_color = "white"

        grid = aquifer.build_grid(prop=prop, z_exag=z_exag)

        if export:
            out_path = export if export.endswith(".ply") else \
                       export.rsplit(".", 1)[0] + ".ply"
            surface = prepare_colored_surface(grid, prop, log_scale=log_scale)
            surface.save(out_path)
            print(f"Exportado: {out_path}")

        if clip:
            p.add_mesh_clip_plane(grid, **mesh_kwargs)
        else:
            p.add_mesh(grid, **mesh_kwargs)

        p.add_axes(xlabel="X (m)", ylabel="Y (m)", zlabel="Z (m)")
        p.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="showModelM6")
    parser.add_argument("model_workspace")
    parser.add_argument("--prop",         default="k")
    parser.add_argument("--show-grid",    action="store_true")
    parser.add_argument("--z-exag",       type=float, default=1.0)
    parser.add_argument("--no-log",       action="store_true")
    parser.add_argument("--clip",         action="store_true",
                        help="Plano de corte interactivo")
    parser.add_argument("--cuts",         type=int, default=None,
                        help="Visualizar modelo dividido en NxN bloques")
    parser.add_argument("--split-unique", action="store_true",
                        help="Un checkbox por cada valor distinto de la propiedad")
    parser.add_argument("--clean",        action="store_true",
                        help="Eliminar componentes pequeños y encerrados (para impresión 3D)")
    parser.add_argument("--min-ratio",    type=float, default=0.01,
                        help="Fracción mínima respecto al mayor componente (default: 0.01 = 1%%)")
    parser.add_argument("--export",       default=None)

    args = parser.parse_args()

    visualizeModflow(
        args.model_workspace,
        prop=args.prop,
        showGrid=args.show_grid,
        z_exag=args.z_exag,
        log_scale=not args.no_log,
        export=args.export,
        clip=args.clip,
        cuts=args.cuts,
        split_unique=args.split_unique,
        clean=args.clean,
        min_ratio=args.min_ratio,
    )