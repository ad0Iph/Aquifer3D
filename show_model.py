import pyvista as pv
import argparse
from aquifer_grid import AquiferGrid

def visualizeModflow(model_ws, sim_name, prop="k", showGrid=False, z_exag=1.0, log_scale=True):
    
    aquifer = AquiferGrid(model_ws, sim_name)
    aquifer.load()
    grid = aquifer.build_grid(prop="k", z_exag=3.0)

    p = pv.Plotter()
    p.background_color = "white"
    p.add_mesh(
        grid, scalars=prop, cmap="viridis",
        show_edges=showGrid, log_scale=log_scale,
        scalar_bar_args={"title": f"{'log10(' + prop + ')' if log_scale else prop}"}
    )
    p.add_axes(xlabel="X (m)", ylabel="Y (m)", zlabel="Z (m)")
    p.show()
    p.save_mesh(grid, "aquifer_pyvista.ply")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="showModelM6")
    parser.add_argument("model_workspace")
    parser.add_argument("simulation_name")
    parser.add_argument("--prop",      default="k")
    parser.add_argument("--show-grid", action="store_true")
    parser.add_argument("--z-exag",    type=float, default=1.0)
    parser.add_argument("--no-log",    action="store_true")
    args = parser.parse_args()

    visualizeModflow(
        args.model_workspace, args.simulation_name, args.prop,
        showGrid=args.show_grid,
        z_exag=args.z_exag,
        log_scale=not args.no_log
    )