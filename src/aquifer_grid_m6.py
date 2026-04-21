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

    @classmethod
    def from_subset(cls, nlay, nrow, ncol, x_edges, y_edges, z_edges, properties):
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

    def _check_loaded(self):
        if self.nlay is None:
            raise RuntimeError("Debes llamar load() antes de usar esta función.")

    def load(self):
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

    def build_grid(self, prop="k", z_exag=1.0, xy_scale=1.0):
        self._check_loaded()

        values  = self.properties[prop]
        x_edges = self.x_edges * xy_scale
        y_edges = self.y_edges * xy_scale
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

        grid = pv.UnstructuredGrid(cells, cell_types, pts)
        grid.cell_data[prop]    = values[mask]
        grid.cell_data["layer"] = k.astype(np.int32)

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

    @staticmethod
    def split_grid_by_unique(grid, prop="k"):
        """
        Detecta todos los valores distintos de una propiedad y retorna
        un subgrid por cada valor único, ordenado de menor a mayor.
        """
        scalars = grid.cell_data[prop]
        unique_vals = np.unique(scalars[~np.isnan(scalars)])

        segments = {}
        for val in np.sort(unique_vals):
            indices = np.where(scalars == val)[0]
            if indices.size == 0:
                continue
            segments[val] = grid.extract_cells(indices)

        return segments

    @staticmethod
    def clean_surface(surface, min_ratio=0.01, remove_enclosed=True, verbose=True):
        """
        Limpia una superficie eliminando componentes pequeños y/o
        contenidos dentro de otros.

        Parámetros
        ----------
        surface : pv.PolyData
            Superficie triangulada.
        min_ratio : float
            Fracción mínima de celdas respecto al componente más grande.
            Componentes con menos de (max_cells * min_ratio) celdas se eliminan.
            Ej: 0.01 = eliminar todo lo que tenga menos del 1% del mayor.
        remove_enclosed : bool
            Si True, elimina componentes cuyo centro está dentro de otro.
        verbose : bool
            Imprimir información de los componentes eliminados.

        Retorna
        -------
        pv.PolyData
            Superficie limpia.
        """
        conn = surface.connectivity(largest=False)
        region_ids = conn.cell_data["RegionId"]
        unique_regions = np.unique(region_ids)

        if len(unique_regions) <= 1:
            if verbose:
                print(f"    1 solo componente, nada que limpiar")
            return surface

        components = []
        for rid in unique_regions:
            indices = np.where(region_ids == rid)[0]
            comp = conn.extract_cells(indices)
            components.append({
                "id":      rid,
                "mesh":    comp,
                "n_cells": comp.n_cells,
                "bounds":  comp.bounds,       # (xmin, xmax, ymin, ymax, zmin, zmax)
                "center":  comp.center,
            })

        components.sort(key=lambda c: c["n_cells"], reverse=True)
        max_cells = components[0]["n_cells"]
        threshold = int(max_cells * min_ratio)

        if verbose:
            print(f"    {len(components)} componentes detectados "
                  f"(mayor: {max_cells}, umbral: {threshold} celdas)")

        keep = []
        removed_small = 0
        for comp in components:
            if comp["n_cells"] < threshold:
                removed_small += 1
                if verbose:
                    print(f"    ✗ Componente {comp['id']}: {comp['n_cells']} celdas "
                          f"(< {threshold}) → eliminado por tamaño")
            else:
                keep.append(comp)

        if remove_enclosed and len(keep) > 1:
            def bbox_contains_point(bounds, point):
                """Verifica si un punto está dentro de un bounding box."""
                return (bounds[0] <= point[0] <= bounds[1] and
                        bounds[2] <= point[1] <= bounds[3] and
                        bounds[4] <= point[2] <= bounds[5])

            final = []
            removed_enclosed = 0
            for i, comp in enumerate(keep):
                enclosed = False
                for j, other in enumerate(keep):
                    if i == j:
                        continue
                    # Si el otro es más grande y contiene el centro de este
                    if (other["n_cells"] > comp["n_cells"] and
                            bbox_contains_point(other["bounds"], comp["center"])):
                        enclosed = True
                        removed_enclosed += 1
                        if verbose:
                            print(f"    ✗ Componente {comp['id']}: {comp['n_cells']} celdas "
                                  f"→ eliminado (contenido en componente {other['id']})")
                        break
                if not enclosed:
                    final.append(comp)
        else:
            final = keep

        if verbose:
            total_removed = len(components) - len(final)
            print(f"    Resultado: {len(final)} componentes "
                  f"({total_removed} eliminados)")

        if len(final) == 0:
            if verbose:
                print(f"    ⚠ Todos eliminados, manteniendo el mayor")
            return components[0]["mesh"].extract_surface()

        if len(final) == 1:
            return final[0]["mesh"].extract_surface()

        combined = final[0]["mesh"]
        for comp in final[1:]:
            combined = combined.merge(comp["mesh"])

        return combined.extract_surface()