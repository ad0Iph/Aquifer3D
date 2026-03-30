import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import argparse
from aquifer_grid import AquiferGrid

def export_model(model_ws, sim_name, prop="k", z_exag=1.0, log_scale=True,
                 decimate=0.0, out_file="aquifer.ply"):
    """
    Exporta el modelo como superficie PLY coloreada, igual a lo que
    muestra show_model.py (mismo grid, mismo colormap, misma escala).
    """
    aquifer = AquiferGrid(model_ws, sim_name)
    aquifer.load()
    grid = aquifer.build_grid(prop="k", z_exag=3.0)

    # --- 1. Extraer superficie ---
    surface = grid.extract_surface()
    surface = surface.triangulate()
    surface = surface.fill_holes(50)
    surface = surface.clean()
    if decimate > 0:
        surface = surface.decimate(decimate)

    print(f"¿Watertight? {surface.is_manifold}")



    # --- 3. Exportar ---
    surface.save(out_file)
    print(f"Exportado: {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="exportModelM6")
    parser.add_argument("model_workspace")
    parser.add_argument("simulation_name")
    parser.add_argument("--prop",     default="k")
    parser.add_argument("--z-exag",   type=float, default=1.0)
    parser.add_argument("--no-log",   action="store_true")
    parser.add_argument("--decimate", type=float, default=0.0)
    parser.add_argument("--out",      default="aquifer.ply")
    args = parser.parse_args()

    export_model(
        args.model_workspace, args.simulation_name, args.prop,
        z_exag=args.z_exag,
        log_scale=not args.no_log,
        decimate=args.decimate,
        out_file=args.out
    )