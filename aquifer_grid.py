import flopy
import numpy as np
import pyvista as pv
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

    def build_grid(self, prop="k", z_exag=1.0):
        """
        Construye un UnstructuredGrid hexaédrico con los datos de la clase.
        Fuente única de verdad para show_model y export_model.
        Las celdas con NaN se omiten (útil tras split_by_property).
        """
        self._check_loaded()

        values = self.properties[prop]

        points_list = []
        cells_list  = []
        cell_values = []
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

                    x0, x1 = self.x_edges[j],     self.x_edges[j + 1]
                    y0, y1 = self.y_edges[i],     self.y_edges[i + 1]
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

        cells      = np.array(cells_list, dtype=np.int64).ravel()
        cell_types = np.full(len(cell_values), hex_type, dtype=np.uint8)
        pts        = np.array(points_list)

        grid = pv.UnstructuredGrid(cells, cell_types, pts)
        grid.cell_data[prop] = np.array(cell_values)
        grid.points[:, 2]   *= z_exag

        return grid

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
                groups[group_name]["k33"] = np.where(mask, self.properties["k33"], np.nan)

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
    parser.add_argument("--cuts", type=int, default=None,
                        help="Número de cortes espaciales (opcional)")
    parser.add_argument("--split-prop", default=None,
                        help="Dividir por propiedad (ej: k)")
    parser.add_argument("--out", default="aquifer_grid_m6.npz")
    args = parser.parse_args()

    aquifer = AquiferGrid(args.model_workspace, args.simulation_name)
    aquifer.load()

    if args.cuts:
        aquifer.export_splitN(args.cuts)
    elif args.split_prop:
        aquifer.export_by_property(args.split_prop)
    else:
        aquifer.export(args.out)