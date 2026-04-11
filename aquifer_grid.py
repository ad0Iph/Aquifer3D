import flopy
import numpy as np
import pyvista as pv
from pathlib import Path

class AquiferGrid:
    def __init__(self, model_ws, sim_name):
        """
        Clase para cargar un modelo MODFLOW y construir un UnstructuredGrid de PyVista.
        """
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
        """
        Verifica que el modelo haya sido cargado antes de construir el grid.
        """
        if self.nlay is None:
            raise RuntimeError("Debes llamar load() antes de usar esta función.")

    def load(self):
        """
        Carga el modelo MODFLOW y extrae la información necesaria para construir el grid.
        """
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
        """
        Construye un UnstructuredGrid de hexaedros a partir del modelo MODFLOW.
        Versión vectorizada — ~100x más rápido que la versión con bucles.
        """
        self._check_loaded()

        values  = self.properties[prop]
        x_edges = self.x_edges * xy_scale
        y_edges = self.y_edges * xy_scale
        nlay, nrow, ncol = self.nlay, self.nrow, self.ncol

        K, I, J = np.mgrid[0:nlay+1, 0:nrow+1, 0:ncol+1]

        ic = np.clip(I, 0, nrow - 1)
        jc = np.clip(J, 0, ncol - 1)

        pts = np.stack([
            x_edges[J],
            y_edges[I],
            self.z_edges[K, ic, jc] * z_exag,
        ], axis=-1).reshape(-1, 3)

        mask = ~np.isnan(values)                       
        k, i, j = np.nonzero(mask)                     
        n_cells = k.size

        stride_k = (nrow + 1) * (ncol + 1)
        stride_i = ncol + 1

        def pidx(kk, ii, jj):
            return kk * stride_k + ii * stride_i + jj
        v0 = pidx(k+1, i,   j  )
        v1 = pidx(k+1, i,   j+1)
        v2 = pidx(k+1, i+1, j+1)
        v3 = pidx(k+1, i+1, j  )
        v4 = pidx(k,   i,   j  )
        v5 = pidx(k,   i,   j+1)
        v6 = pidx(k,   i+1, j+1)
        v7 = pidx(k,   i+1, j  )

        cells = np.column_stack([
            np.full(n_cells, 8, dtype=np.int64),
            v0, v1, v2, v3, v4, v5, v6, v7,
        ]).ravel()

        cell_types = np.full(n_cells, pv.CellType.HEXAHEDRON, dtype=np.uint8)

        grid = pv.UnstructuredGrid(cells, cell_types, pts)
        grid.cell_data[prop]    = values[mask]
        grid.cell_data["layer"] = k.astype(np.int32)

        return grid

    def splitN(self, N):
        """
        Divide el grid en N x N bloques iguales.
        """
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
        """
        Divide el grid en N bloques basados en los valores de una propiedad dada.
        """
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