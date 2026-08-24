import numpy as np

try:
    from mesh_repair import repair_surface
    HAS_REPAIR = True
except ImportError:
    HAS_REPAIR = False

class Surfaces:
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

    def compute_exterior_set(self):
        """compute the set of exterior points of the full mesh, used for cut operations"""
        active = np.where(~np.isnan(self.group_ids))[0]
        full = self.grid.extract_cells(active).extract_surface()
        return set(map(tuple, np.round(np.asarray(full.points), 6)))

    def displace_boundary(self, surface, full_set, tol):
        """Desplaza vértices de interfaz según la normal de las caras de interfaz."""
        pts = np.asarray(surface.points)
        keys = np.round(pts, 6)
        is_boundary = np.fromiter(
            (tuple(p) not in full_set for p in keys), dtype=bool, count=len(pts))
        if not is_boundary.any():
            return

        faces = surface.faces.reshape(-1, 4)[:, 1:]
        face_iface = is_boundary[faces].all(axis=1)   
        if not face_iface.any():
            return

        surface.compute_normals(cell_normals=True, point_normals=False,
                                auto_orient_normals=True, inplace=True)
        fn = np.asarray(surface.cell_data["Normals"])

        vn = np.zeros_like(pts)
        np.add.at(vn, faces[face_iface].ravel(),
                np.repeat(fn[face_iface], 3, axis=0))
        norms = np.linalg.norm(vn, axis=1, keepdims=True)
        move = norms.ravel() > 1e-12
        vn[move] /= norms[move]
        pts[move] -= vn[move] * tol                   
        surface.points = pts

    def rebuild_all_surfaces(self):
        """Rebuild the surfaces of all groups from the original mesh and group IDs"""
        unique_groups = np.unique(self.group_ids[~np.isnan(self.group_ids)])
        self.surfaces.clear()
        full_set = self.compute_exterior_set() if self.prepared else None

        for val in np.sort(unique_groups):
            indices = np.where(self.group_ids == val)[0]
            if indices.size == 0:
                continue
            sub = self.grid.extract_cells(indices)
            surface = sub.extract_surface().triangulate()
            if self.prepared:
                self.displace_boundary(surface, full_set, self.tolerance)
            if HAS_REPAIR:
                surface = repair_surface(surface, decimate=self.decimate,
                                        verbose=False)
            self.surfaces[val] = surface