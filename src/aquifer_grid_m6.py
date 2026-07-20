import flopy
import numpy as np
import pyvista as pv
from pathlib import Path


class AquiferGridM6:
    def __init__(self, model_ws):
        self.model_ws = Path(model_ws)

        self.nlay = None
        self.nrow = None
        self.ncol = None

        self.x_edges = None
        self.y_edges = None
        self.z_edges = None

        self.properties = {}

    def check_loaded(self):
        """Checkea que la clase este inicializada con un modelo MODFLOW cargado, lanza error si no lo esta"""
        if self.nlay is None:
            raise RuntimeError("Debes llamar load() antes de usar esta función.")

    def load(self):
        """Carga los archivos dis y npf del modelo MODFLOW 6 desde la carpeta model_ws y extrae la geometría y propiedades relevantes"""
        sim = flopy.mf6.MFSimulation.load(
            sim_ws=self.model_ws,
            verbosity_level=0,
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

    def build_grid(self, prop="k", z_exag=1.0):
        """Construye un grid de PyVista a partir de la geometría y propiedades cargadas del modelo MODFLOW 6"""
        self.check_loaded()

        values  = self.properties[prop]
        x_edges = self.x_edges
        y_edges = self.y_edges
        nlay, nrow, ncol = self.nlay, self.nrow, self.ncol

        K, I, J = np.mgrid[0:nlay + 1, 0:nrow + 1, 0:ncol + 1]

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

        #Se definen los vértices de cada celda hexaédrica en el orden correcto para PyVista
        v0 = (k + 1) * stride_k + i       * stride_i + j
        v1 = (k + 1) * stride_k + i       * stride_i + (j + 1)
        v2 = (k + 1) * stride_k + (i + 1) * stride_i + (j + 1)
        v3 = (k + 1) * stride_k + (i + 1) * stride_i + j
        v4 = k       * stride_k + i       * stride_i + j
        v5 = k       * stride_k + i       * stride_i + (j + 1)
        v6 = k       * stride_k + (i + 1) * stride_i + (j + 1)
        v7 = k       * stride_k + (i + 1) * stride_i + j

        # Se construye el arreglo de celdas y tipos de celda para PyVista
        cells = np.column_stack([
            np.full(n_cells, 8, dtype=np.int64),
            v0, v1, v2, v3, v4, v5, v6, v7,
        ]).ravel()

        cell_types = np.full(n_cells, pv.CellType.HEXAHEDRON, dtype=np.uint8)

        grid = pv.UnstructuredGrid(cells, cell_types, pts)
        grid.cell_data[prop]    = values[mask]
        grid.cell_data["layer"] = k.astype(np.int32)

        return grid