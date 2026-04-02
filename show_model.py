import pyvista as pv
import argparse
from aquifer_grid import AquiferGrid

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

def apply_colormap(grid, prop, cmap="viridis", log_scale=True):
    """Aplica el mismo colormap que PyVista y guarda RGB en el mesh."""
    scalars = grid.cell_data[prop]
    cmap_fn = plt.get_cmap(cmap)

    if log_scale:
        vmin = np.nanmin(scalars[scalars > 0])
        vmax = np.nanmax(scalars)
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = mcolors.Normalize(vmin=np.nanmin(scalars), vmax=np.nanmax(scalars))

    rgba   = cmap_fn(norm(scalars))               # (n_cells, 4) float 0–1
    rgb255 = (rgba[:, :3] * 255).astype(np.uint8) # (n_cells, 3) uint8
    grid.cell_data["RGB"] = rgb255

    return grid

def visualizeModflow(model_ws, sim_name, prop="k", showGrid=False, z_exag=1.0, log_scale=True,  export=None):    
    aquifer = AquiferGrid(model_ws, sim_name)
    aquifer.load()
    grid = aquifer.build_grid(prop="k", z_exag=z_exag, xy_scale=1.0)

    if export:
        grid_colored = apply_colormap(grid, prop, cmap="viridis", log_scale=log_scale)
        grid_colored.save(export)
        print(f"Malla exportada con colores: {export}")

    p = pv.Plotter()
    p.background_color = "white"
    p.add_mesh(
        grid, scalars=prop, cmap="viridis",
        show_edges=showGrid, log_scale=log_scale,
        scalar_bar_args={"title": f"{'log10(' + prop + ')' if log_scale else prop}"}
    )
    p.add_axes(xlabel="X (m)", ylabel="Y (m)", zlabel="Z (m)")
    p.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="showModelM6")
    parser.add_argument("model_workspace")
    parser.add_argument("simulation_name")
    parser.add_argument("--prop",      default="k")
    parser.add_argument("--show-grid", action="store_true")
    parser.add_argument("--z-exag",    type=float, default=1.0)
    parser.add_argument("--no-log",    action="store_true")
    parser.add_argument("--export",    default=None,
                    help="Exportar malla: aquifer.ply / .vtk / .stl")
    args = parser.parse_args()

    visualizeModflow(
        args.model_workspace, args.simulation_name, args.prop,
        showGrid=args.show_grid,
        z_exag=args.z_exag,
        log_scale=not args.no_log,
        export=args.export
    )