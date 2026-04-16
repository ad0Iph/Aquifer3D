import flopy
import numpy as np
import pyvista as pv
from pathlib import Path


class AquiferGridM6:
    def __init__(self, model_ws):
        """
        Clase para cargar un modelo MODFLOW 6 y construir un UnstructuredGrid de PyVista.
        """
        self.model_ws = Path(model_ws)

        self.nlay = None
        self.nrow = None
        self.ncol = None

        self.x_edges = None
        self.y_edges = None
        self.z_edges = None

        self.properties = {}

    # ── Constructor alternativo para subgrids ────────────────────────

    @classmethod
    def from_subset(cls, nlay, nrow, ncol, x_edges, y_edges, z_edges, properties):
        """Crea un AquiferGridM6 a partir de arrays ya recortados, sin cargar MODFLOW."""
        obj = cls.__new__(cls)
        obj.model_ws   = None
        obj.nlay       = nlay
        obj.nrow       = nrow
        obj.ncol       = ncol
        obj.x_edges    = x_edges
        obj.y_edges    = y_edges
        obj.z_edges    = z_edges
        obj.properties = properties
        return obj

    # ── Utilidades internas ──────────────────────────────────────────

    def _check_loaded(self):
        if self.nlay is None:
            raise RuntimeError("Debes llamar load() antes de usar esta función.")

    # ── Carga del modelo MODFLOW ─────────────────────────────────────

    def load(self):
        """Carga el modelo MODFLOW 6 y extrae la información del grid."""
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

    # ── Construcción del grid 3D (vectorizado) ───────────────────────

    def build_grid(self, prop="k", z_exag=1.0, xy_scale=1.0):
        """
        Construye un UnstructuredGrid de hexaedros a partir del modelo MODFLOW.
        Versión vectorizada.
        """
        self._check_loaded()

        values  = self.properties[prop]
        x_edges = self.x_edges * xy_scale
        y_edges = self.y_edges * xy_scale
        nlay, nrow, ncol = self.nlay, self.nrow, self.ncol

        # ── 1. Puntos del grid ──────────────────────────────────────
        K, I, J = np.mgrid[0:nlay + 1, 0:nrow + 1, 0:ncol + 1]

        ic = np.clip(I, 0, nrow - 1)
        jc = np.clip(J, 0, ncol - 1)

        pts = np.stack([
            x_edges[J],
            y_edges[I],
            self.z_edges[K, ic, jc] * z_exag,
        ], axis=-1).reshape(-1, 3)

        # ── 2. Máscara de celdas activas ────────────────────────────
        mask = ~np.isnan(values)
        k, i, j = np.nonzero(mask)
        n_cells = k.size

        # ── 3. Índice lineal de punto ───────────────────────────────
        stride_k = (nrow + 1) * (ncol + 1)
        stride_i = ncol + 1

        v0 = (k + 1) * stride_k + i       * stride_i + j
        v1 = (k + 1) * stride_k + i       * stride_i + (j + 1)
        v2 = (k + 1) * stride_k + (i + 1) * stride_i + (j + 1)
        v3 = (k + 1) * stride_k + (i + 1) * stride_i + j
        v4 = k       * stride_k + i       * stride_i + j
        v5 = k       * stride_k + i       * stride_i + (j + 1)
        v6 = k       * stride_k + (i + 1) * stride_i + (j + 1)
        v7 = k       * stride_k + (i + 1) * stride_i + j

        cells = np.column_stack([
            np.full(n_cells, 8, dtype=np.int64),
            v0, v1, v2, v3, v4, v5, v6, v7,
        ]).ravel()

        cell_types = np.full(n_cells, pv.CellType.HEXAHEDRON, dtype=np.uint8)

        # ── 4. Construir grid ───────────────────────────────────────
        grid = pv.UnstructuredGrid(cells, cell_types, pts)
        grid.cell_data[prop]    = values[mask]
        grid.cell_data["layer"] = k.astype(np.int32)

        return grid

    # ── Subdivisión espacial NxN ─────────────────────────────────────

    def splitN(self, N):
        """Subdivide el grid en NxN bloques espaciales."""
        self._check_loaded()
        row_splits = np.linspace(0, self.nrow, N + 1, dtype=int)
        col_splits = np.linspace(0, self.ncol, N + 1, dtype=int)

        subgrids = {}
        for i in range(N):
            for j in range(N):
                r0, r1 = row_splits[i], row_splits[i + 1]
                c0, c1 = col_splits[j], col_splits[j + 1]

                subgrids[f"block_{i}_{j}"] = AquiferGridM6.from_subset(
                    nlay       = self.nlay,
                    nrow       = r1 - r0,
                    ncol       = c1 - c0,
                    x_edges    = self.x_edges[c0:c1 + 1],
                    y_edges    = self.y_edges[r0:r1 + 1],
                    z_edges    = self.z_edges[:, r0:r1, c0:c1],
                    properties = {k: v[:, r0:r1, c0:c1]
                                  for k, v in self.properties.items()},
                )
        return subgrids