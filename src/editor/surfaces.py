import numpy as np

try:
    from mesh_repair import repair_surface
    HAS_REPAIR = True
except ImportError:
    HAS_REPAIR = False

class Surfaces:
    def rebuild_all_surfaces(self):
        """Rebuild the surfaces of all groups from the original mesh and group IDs"""
        unique_groups = np.unique(self.group_ids[~np.isnan(self.group_ids)])
        self.surfaces.clear()

        for val in np.sort(unique_groups):
            indices = np.where(self.group_ids == val)[0]
            if indices.size == 0:
                continue
            sub = self.grid.extract_cells(indices)
            surface = sub.extract_surface().triangulate()
            if HAS_REPAIR:
                surface = repair_surface(surface, decimate=self.decimate,
                                         verbose=False)
            self.surfaces[val] = surface

    def compute_components(self, group_key):
        """Compute the connected components of a group and return a list of dictionaries with their surfaces and cells"""
        group_mask = self.group_ids == group_key
        group_cell_indices = np.where(group_mask)[0]
        if group_cell_indices.size == 0:
            return []

        sub_grid = self.grid.extract_cells(group_cell_indices)
        connected = sub_grid.connectivity(extraction_mode='all')
        region_ids = connected.cell_data["RegionId"]

        components = []
        for rid in np.sort(np.unique(region_ids)):
            region_mask = region_ids == rid
            grid_cells = group_cell_indices[region_mask]
            comp_grid = connected.extract_cells(np.where(region_mask)[0])
            comp_surface = comp_grid.extract_surface().triangulate()
            components.append({
                "grid_indices": grid_cells,
                "surface": comp_surface,
                "n_cells": len(grid_cells),
            })
        components.sort(key=lambda c: c["n_cells"], reverse=True)
        return components