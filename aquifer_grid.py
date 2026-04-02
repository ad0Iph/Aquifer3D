import flopy
import numpy as np
import pyvista as pv
from pathlib import Path

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

        self.x_edges = np.concatenate([[0], np.cumsum(dis.delr.get_data())])
        self.y_edges = np.concatenate([[0], np.cumsum(dis.delc.get_data())])

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

    def build_grid(self, prop="k", z_exag=1.0, xy_scale=1.0):
        self._check_loaded()

        values  = self.properties[prop]
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

                    x0, x1 = x_edges[j],         x_edges[j + 1]
                    y0, y1 = y_edges[i],         y_edges[i + 1]
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

    def splitN(self, N):
        self._check_loaded()
        row_splits = np.linspace(0, self.nrow, N + 1, dtype=int)
        col_splits = np.linspace(0, self.ncol, N + 1, dtype=int)

        subgrids = {}
        for i in range(N):
            for j in range(N):
                r0, r1 = row_splits[i], row_splits[i + 1]
                c0, c1 = col_splits[j], col_splits[j + 1]

                sub = AquiferGrid.__new__(AquiferGrid)
                sub.nlay       = self.nlay
                sub.nrow       = r1 - r0
                sub.ncol       = c1 - c0
                sub.x_edges    = self.x_edges[c0:c1 + 1]
                sub.y_edges    = self.y_edges[r0:r1 + 1]
                sub.z_edges    = self.z_edges[:, r0:r1, c0:c1]
                sub.properties = {k: v[:, r0:r1, c0:c1]
                                  for k, v in self.properties.items()}
                subgrids[f"block_{i}_{j}"] = sub

        return subgrids

    def split_by_property(self, prop="k", N=3):
        self._check_loaded()
        values = self.properties[prop]
        bins   = np.linspace(np.nanmin(values), np.nanmax(values), N + 1)

        subgrids = {}
        for i in range(N):
            mask = ((values >= bins[i]) & (values <= bins[i + 1])
                    if i == N - 1 else
                    (values >= bins[i]) & (values <  bins[i + 1]))

            sub = AquiferGrid.__new__(AquiferGrid)
            sub.nlay       = self.nlay
            sub.nrow       = self.nrow
            sub.ncol       = self.ncol
            sub.x_edges    = self.x_edges
            sub.y_edges    = self.y_edges
            sub.z_edges    = self.z_edges
            sub.properties = {k: np.where(mask, v, np.nan)
                              for k, v in self.properties.items()}
            subgrids[f"{prop}_{i}"] = sub

        return subgrids