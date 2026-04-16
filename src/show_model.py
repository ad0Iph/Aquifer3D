import pyvista as pv
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from aquifer_grid_m6 import AquiferGridM6


def apply_colormap(surface, prop, cmap="viridis", log_scale=True, norm=None):
    """
    Aplica un colormap a una superficie de PyVista y deja los colores
    como point_data con canales separados (red, green, blue) para
    compatibilidad con MeshLab y otros visores.
    """
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
    """
    Pipeline completo: grid → superficie triangulada → coloreada →
    point_data con canales r/g/b separados, lista para guardar como PLY.
    """
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

def visualizeModflow(model_ws, sim_name, prop="k", showGrid=False,
                     z_exag=1.0, log_scale=True, export=None,
                     clip=False, cuts=None):
    """
    Visualiza un modelo de MODFLOW usando PyVista.
    """
    aquifer = AquiferGridM6(model_ws)
    aquifer.load()

    mesh_kwargs = dict(
        scalars=prop, cmap="viridis",
        show_edges=showGrid, log_scale=log_scale,
        scalar_bar_args={"title": f"{'log10(' + prop + ')' if log_scale else prop}"},
    )

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
                surface = prepare_colored_surface(grid, prop, log_scale=log_scale, norm=norm)
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
    parser.add_argument("--prop",      default="k")
    parser.add_argument("--show-grid", action="store_true")
    parser.add_argument("--z-exag",    type=float, default=1.0)
    parser.add_argument("--no-log",    action="store_true")
    parser.add_argument("--clip",      action="store_true", help="Plano de corte interactivo")
    parser.add_argument("--cuts",      type=int, default=None, help="Visualizar modelo dividido en NxN bloques")
    parser.add_argument("--export",    default=None)

    args = parser.parse_args()

    visualizeModflow(
        args.model_workspace, args.prop,
        showGrid=args.show_grid,
        z_exag=args.z_exag,
        log_scale=not args.no_log,
        export=args.export,
        clip=args.clip,
        cuts=args.cuts,
    )