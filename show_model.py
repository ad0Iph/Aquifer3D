import pyvista as pv
import argparse
from aquifer_grid import AquiferGrid
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

def apply_colormap(grid, prop, cmap="viridis", log_scale=True, norm=None):
    """
    Aplica un colormap a un grid de PyVista basado en los valores de una propiedad dada.
    Si log_scale es True, se aplicará una escala logarítmica a los valores
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

def visualizeModflow(model_ws, sim_name, prop="k", showGrid=False,
                     z_exag=1.0, log_scale=True, export=None,
                     clip=False, cuts=None):
    """
    Visualiza un modelo de MODFLOW usando PyVista. Permite mostrar el grid, aplicar escala logarítmica, exportar a VTU, y dividir el modelo en bloques.
    """
    aquifer = AquiferGrid(model_ws, sim_name)
    aquifer.load()
    
    mesh_kwargs = dict(
        scalars=prop, cmap="viridis",
        show_edges=showGrid, log_scale=log_scale,
        scalar_bar_args={"title": f"{'log10(' + prop + ')' if log_scale else prop}"}
    )
    
    if cuts:
        subgrids = aquifer.splitN(cuts)
        # Calcular norm global para que todos los bloques compartan la misma escala
        all_vals = np.concatenate([
            sub.properties[prop].ravel() for sub in subgrids.values()
        ])
        all_vals = all_vals[~np.isnan(all_vals)]
        if log_scale:
            norm = mcolors.LogNorm(vmin=all_vals[all_vals > 0].min(), vmax=all_vals.max())
        else:
            norm = mcolors.Normalize(vmin=all_vals.min(), vmax=all_vals.max())
        
        # Crear un plotter para cada subgrid
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
                out = f"{export}_{name}.vtk"
                grid.save(out)
                print(f"Exportado: {out}")
            
            # Mostrar cada ventana
            p.show(title=f"Bloque: {name}")
    else:
        p = pv.Plotter()
        p.background_color = "white"
        
        grid = aquifer.build_grid(prop=prop, z_exag=z_exag)
        if export:
            grid_colored = apply_colormap(grid, prop, log_scale=log_scale)
            grid_colored.extract_surface().save(export)
            print(f"Exportado: {export}")
        if clip:
            p.add_mesh_clip_plane(grid, **mesh_kwargs)
        else:
            p.add_mesh(grid, **mesh_kwargs)
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
    parser.add_argument("--clip",      action="store_true",
                        help="Plano de corte interactivo")
    parser.add_argument("--cuts",      type=int, default=None,
                        help="Visualizar modelo dividido en NxN bloques")
    parser.add_argument("--export",    default=None)
    args = parser.parse_args()
    
    visualizeModflow(
        args.model_workspace, args.simulation_name, args.prop,
        showGrid=args.show_grid,
        z_exag=args.z_exag,
        log_scale=not args.no_log,
        export=args.export,
        clip=args.clip,
        cuts=args.cuts
    )