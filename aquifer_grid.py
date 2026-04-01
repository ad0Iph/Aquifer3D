import flopy
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import argparse


class AquiferGrid:
    def __init__(self, model_ws, sim_name):
        self.model_ws = Path(model_ws)
        self.sim_name = sim_name

        self.nlay = None
        self.nrow = None
        self.ncol = None

        self.x_edges = None
        self.y_edges = None
        self.z_edges = None

        self.properties = {}

    def _check_loaded(self):
        if self.nlay is None:
            raise RuntimeError("Debes llamar load() antes de usar esta función.")

    # ------------------------------------------------------------------ #
    #  Carga                                                               #
    # ------------------------------------------------------------------ #

    def load(self):
        sim = flopy.mf6.MFSimulation.load(
            sim_name=self.sim_name,
            sim_ws=self.model_ws,
            verbosity_level=0
        )
        model = sim.get_model()
        dis   = model.get_package("dis")
        npf   = model.get_package("npf")

        self.nlay = dis.nlay.get_data()
        self.nrow = dis.nrow.get_data()
        self.ncol = dis.ncol.get_data()

        delr = dis.delr.get_data()
        delc = dis.delc.get_data()

        self.x_edges = np.concatenate([[0], np.cumsum(delr)])
        self.y_edges = np.concatenate([[0], np.cumsum(delc)])

        top  = dis.top.get_data()
        botm = dis.botm.get_data()

        self.z_edges = np.zeros((self.nlay + 1, self.nrow, self.ncol))
        self.z_edges[0]  = top
        self.z_edges[1:] = botm

        self.properties["k"] = npf.k.get_data().copy()

        try:
            k33_array = npf.k33.get_data()
            if k33_array is not None:
                self.properties["k33"] = k33_array.copy()
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    #  Construcción del grid PyVista                                       #
    # ------------------------------------------------------------------ #

    def build_grid(self, prop="k", z_exag=1.0, xy_scale=1.0):
        self._check_loaded()

        values = self.properties[prop]

        # Escalar edges antes de construir
        x_edges = self.x_edges * xy_scale
        y_edges = self.y_edges * xy_scale

        points_list = []
        cells_list  = []
        cell_values = []
        layer_ids   = []
        point_cache = {}

        def get_pid(x, y, z):
            key = (x, y, z)
            if key not in point_cache:
                point_cache[key] = len(points_list)
                points_list.append([x, y, z])
            return point_cache[key]

        hex_type = pv.CellType.HEXAHEDRON

        for k in range(self.nlay):
            for i in range(self.nrow):
                for j in range(self.ncol):
                    val = values[k, i, j]
                    if np.isnan(val):
                        continue

                    x0, x1 = x_edges[j],     x_edges[j + 1]   # ← escalado
                    y0, y1 = y_edges[i],     y_edges[i + 1]   # ← escalado
                    zt     = self.z_edges[k,     i, j]
                    zb     = self.z_edges[k + 1, i, j]

                    verts = [
                        (x0, y0, zb), (x1, y0, zb),
                        (x1, y1, zb), (x0, y1, zb),
                        (x0, y0, zt), (x1, y0, zt),
                        (x1, y1, zt), (x0, y1, zt),
                    ]
                    ids = [get_pid(*v) for v in verts]
                    cells_list.append([8] + ids)
                    cell_values.append(val)
                    layer_ids.append(k)

        cells      = np.array(cells_list, dtype=np.int64).ravel()
        cell_types = np.full(len(cell_values), hex_type, dtype=np.uint8)
        pts        = np.array(points_list)

        grid = pv.UnstructuredGrid(cells, cell_types, pts)
        grid.cell_data[prop]    = np.array(cell_values)
        grid.cell_data["layer"] = np.array(layer_ids, dtype=np.int32)
        grid.points[:, 2]      *= z_exag

        return grid

    # ------------------------------------------------------------------ #
    #  Exportación para impresión 3D                                      #
    # ------------------------------------------------------------------ #

    def export_layers(self, prop="k", z_exag=1.0, cmap="viridis",
                      log_scale=True, out_dir="layers"):
        """
        Exporta una superficie watertight por capa como .ply independiente.
        Al ensamblarlos en el slicer forman el volumen completo y al cortar
        transversalmente se ven los colores de cada capa.
        """
        self._check_loaded()
        Path(out_dir).mkdir(exist_ok=True)

        grid      = self.build_grid(prop=prop, z_exag=z_exag, xy_scale=2.0)
        scalars   = grid.cell_data[prop]
        layer_ids = grid.cell_data["layer"]
        cmap_fn   = plt.get_cmap(cmap)

        # Misma normalización que show_model para colores idénticos
        if log_scale:
            vmin = np.nanmin(scalars[scalars > 0])
            vmax = np.nanmax(scalars)
            norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
        else:
            norm = mcolors.Normalize(vmin=np.nanmin(scalars),
                                     vmax=np.nanmax(scalars))

        for layer in np.unique(layer_ids):
            mask    = layer_ids == layer
            indices = np.where(mask)[0]
            sub     = grid.extract_cells(indices)

            # Superficie cerrada de esta capa
            surface = sub.extract_surface()
            surface = surface.triangulate()
            surface = surface.fill_holes(100)
            surface = surface.clean()

            is_ok = surface.is_manifold
            print(f"Capa {int(layer):02d} — watertight: {is_ok}")

            # Color por celda usando la mediana de k en la capa
            # (misma lógica que el colormap de show_model)
            median_val = np.nanmedian(sub.cell_data[prop])
            rgba   = cmap_fn(norm(median_val))
            rgb255 = (np.array(rgba[:3]) * 255).astype(np.uint8)
            surface.cell_data["RGB"] = np.tile(rgb255, (surface.n_cells, 1))

            out_path = f"{out_dir}/layer_{int(layer):02d}.ply"
            surface.save(out_path)
            print(f"  → {out_path}")

    # ------------------------------------------------------------------ #
    #  División                                                            #
    # ------------------------------------------------------------------ #

    def splitN(self, N):
        self._check_loaded()
        row_splits = np.linspace(0, self.nrow, N + 1, dtype=int)
        col_splits = np.linspace(0, self.ncol, N + 1, dtype=int)

        subdomains = {}
        for i in range(N):
            for j in range(N):
                r0, r1 = row_splits[i], row_splits[i + 1]
                c0, c1 = col_splits[j], col_splits[j + 1]
                key = f"block_{i}_{j}"
                subdomains[key] = {
                    "x_edges": self.x_edges[c0:c1 + 1],
                    "y_edges": self.y_edges[r0:r1 + 1],
                    "z_edges": self.z_edges[:, r0:r1, c0:c1],
                    "k":       self.properties["k"][:, r0:r1, c0:c1],
                }
                if "k33" in self.properties:
                    subdomains[key]["k33"] = self.properties["k33"][:, r0:r1, c0:c1]

        return subdomains

    def split_by_property(self, prop="k", N=3):
        self._check_loaded()
        values = self.properties[prop]
        vmin   = np.nanmin(values)
        vmax   = np.nanmax(values)
        bins   = np.linspace(vmin, vmax, N + 1)

        groups = {}
        for i in range(N):
            if i == N - 1:
                mask = (values >= bins[i]) & (values <= bins[i + 1])
            else:
                mask = (values >= bins[i]) & (values <  bins[i + 1])

            group_name = f"{prop}_{i}"
            groups[group_name] = {
                "x_edges": self.x_edges,
                "y_edges": self.y_edges,
                "z_edges": self.z_edges,
                prop:      np.where(mask, values, np.nan),
            }
            if "k33" in self.properties:
                groups[group_name]["k33"] = np.where(
                    mask, self.properties["k33"], np.nan
                )

        return groups

    # ------------------------------------------------------------------ #
    #  Exportación .npz                                                    #
    # ------------------------------------------------------------------ #

    def export(self, filename="aquifer_grid_m6.npz"):
        self._check_loaded()
        np.savez(filename, x_edges=self.x_edges, y_edges=self.y_edges,
                 z_edges=self.z_edges, **self.properties)

    def export_splitN(self, N):
        for name, b in self.splitN(N).items():
            np.savez(f"aquifer_{name}.npz", **b)

    def export_by_property(self, prop="k", N=3):
        for name, g in self.split_by_property(prop, N).items():
            np.savez(f"aquifer_{name}.npz", **g)


# ---------------------------------------------------------------------- #
#  CLI                                                                    #
# ---------------------------------------------------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="aquiferGridM6")
    parser.add_argument("model_workspace")
    parser.add_argument("simulation_name")
    parser.add_argument("--cuts",       type=int,   default=None)
    parser.add_argument("--split-prop", default =None)
    parser.add_argument("--export-layers", action="store_true",
                        help="Exportar capas individuales para impresión 3D")
    parser.add_argument("--prop",       default="k")
    parser.add_argument("--z-exag",     type=float, default=1.0)
    parser.add_argument("--no-log",     action="store_true")
    parser.add_argument("--out",        default="aquifer_grid_m6.npz")
    parser.add_argument("--out-dir",    default="layers")
    args = parser.parse_args()

    aquifer = AquiferGrid(args.model_workspace, args.simulation_name)
    aquifer.load()

    if args.export_layers:
        aquifer.export_layers(
            prop=args.prop,
            z_exag=args.z_exag,
            log_scale=not args.no_log,
            out_dir=args.out_dir
        )
    elif args.cuts:
        aquifer.export_splitN(args.cuts)
    elif args.split_prop:
        aquifer.export_by_property(args.split_prop)
    else:
        aquifer.export(args.out)