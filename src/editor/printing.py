import numpy as np


class Printing:
    def prepare_for_printing(self):
        """Aplica la tolerancia a superficies de frontera para impresión 3D"""
        if self.prepared:
            self.prepared = False
            self.refresh_all_actors()
            self.update_status("Tolerancia revertida — superficies originales")
            return

        try:
            tol = self.tolerance
            self.update_status(f"Aplicando tolerancia de {tol} mm...")
            self.plotter.render()

            full_surface = self.full_surface.triangulate()
            full_pts = np.round(np.asarray(full_surface.points), decimals=6)
            full_set = set(map(tuple, full_pts))

            for key in list(self.surfaces.keys()):
                surface = self.surfaces[key]
                if surface is None or surface.n_cells == 0:
                    continue

                pts = np.asarray(surface.points).copy()
                n_pts = len(pts)

                # Vértices que estan en la frontera
                is_boundary = np.zeros(n_pts, dtype=bool)
                for i in range(n_pts):
                    pt_rounded = tuple(np.round(pts[i], decimals=6))
                    if pt_rounded not in full_set:
                        is_boundary[i] = True

                if not np.any(is_boundary):
                    continue

                surface.compute_normals(
                    cell_normals=True, point_normals=False,
                    auto_orient_normals=True, inplace=True)
                face_normals = np.asarray(surface.cell_data["Normals"])

                vert_normals = np.zeros((n_pts, 3))
                vert_count = np.zeros(n_pts)

                faces = surface.faces.reshape(-1, 4)
                for fi in range(surface.n_cells):
                    v0, v1, v2 = faces[fi, 1], faces[fi, 2], faces[fi, 3]
                    fn = face_normals[fi]
                    for vi in [v0, v1, v2]:
                        if is_boundary[vi]:
                            vert_normals[vi] += fn
                            vert_count[vi] += 1

                mask = vert_count > 0
                vert_normals[mask, 2] = 0.0  # Sin offset en Z
                norms = np.linalg.norm(vert_normals[mask], axis=1, keepdims=True)
                valid = norms.ravel() > 1e-12
                vert_normals[mask] = np.where(
                    valid[:, None],
                    vert_normals[mask] / (norms + 1e-12),
                    0.0)

                pts[mask] -= vert_normals[mask] * tol
                surface.points = pts
                self.add_mesh_to_plotter(key)

            self.prepared = True
            self.plotter.render()

            n_groups = len(self.surfaces)
            self.update_status(
                f"✓ Tolerancia de {tol} mm aplicada a {n_groups} grupos — "
                f"[L] revertir — [E] exportar")

        except Exception as e:
            self.update_status(f"Error: {e}")
            import traceback
            traceback.print_exc()