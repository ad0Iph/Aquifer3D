import numpy as np
import pyvista as pv
from .utils import CutPlaneStyle, format_value

class CutPlane:
    def get_vtk_interactor(self):
        iren = self.plotter.iren
        if hasattr(iren, 'interactor') and iren.interactor is not None:
            return iren.interactor
        elif hasattr(iren, '_iren'):
            return iren._iren
        return iren

    def get_cut_normal(self):
        """Devuelve el vector normal del plano de corte en coordenadas globales"""
        return self._cut_rotation[:, 2].copy()

    def rotate_cut_plane(self, axis, angle_deg):
        """Rota el plano de corte alrededor del eje especificado (0=X, 1=Y, 2=Z)"""
        angle = np.radians(angle_deg)
        c, s = np.cos(angle), np.sin(angle)
        if axis == 0:
            R = np.array([[1,0,0],[0,c,-s],[0,s,c]])
        elif axis == 1:
            R = np.array([[c,0,s],[0,1,0],[-s,0,c]])
        else:
            R = np.array([[c,-s,0],[s,c,0],[0,0,1]])
        self._cut_rotation = R @ self._cut_rotation

    def build_cut_plane_mesh(self):
        """Construye un mesh de PyVista representando el plano de corte"""
        bounds = self.grid.bounds
        dx = bounds[1] - bounds[0]
        dy = bounds[3] - bounds[2]
        half = max(dx, dy) * 0.4 * self._cut_scale

        axis_i = self._cut_rotation[:, 0]
        axis_j = self._cut_rotation[:, 1]
        normal = self._cut_rotation[:, 2]

        c = self._cut_origin
        p0 = c - half * axis_i - half * axis_j
        p1 = c + half * axis_i - half * axis_j
        p2 = c - half * axis_i + half * axis_j
        p3 = c + half * axis_i + half * axis_j

        pts = np.array([p0, p1, p2, p3])
        faces = np.array([4, 0, 1, 3, 2])
        plane = pv.PolyData(pts, faces=faces).triangulate()

        return plane, normal

    def update_cut_plane_visual(self):
        """Actualiza la visualización del plano de corte en la escena"""
        if not self._cut_active:
            return
        if self._cut_actor is not None:
            self.plotter.remove_actor(self._cut_actor)
        if self._cut_border_actor is not None:
            self.plotter.remove_actor(self._cut_border_actor)

        plane, _ = self.build_cut_plane_mesh()
        self._cut_actor = self.plotter.add_mesh(
            plane, color="red", opacity=0.15,
            name="cut_plane", pickable=False,
            reset_camera=False)
        edges = plane.extract_feature_edges(
            boundary_edges=True, feature_edges=False,
            manifold_edges=False, non_manifold_edges=False)
        self._cut_border_actor = self.plotter.add_mesh(
            edges, color="red", line_width=2.0,
            name="cut_border", pickable=False,
            reset_camera=False)
        self.plotter.render()

    def track_key_press(self, obj, event):
        """Registra la tecla presionada (para rotar manteniendo Z/X/Y)."""
        vtk_iren = self.get_vtk_interactor()
        key = vtk_iren.GetKeySym()
        if key:
            self._keys_down.add(key.lower())

    def track_key_release(self, obj, event):
        """Elimina la tecla del registro al soltarla."""
        vtk_iren = self.get_vtk_interactor()
        key = vtk_iren.GetKeySym()
        if key:
            self._keys_down.discard(key.lower())

    def toggle_cut_plane(self):
        """Activa o desactiva el modo de plano de corte libre."""
        vtk_iren = self.get_vtk_interactor()

        if self._cut_active:
            if self._cut_actor is not None:
                self.plotter.remove_actor(self._cut_actor)
                self._cut_actor = None
            if self._cut_border_actor is not None:
                self.plotter.remove_actor(self._cut_border_actor)
                self._cut_border_actor = None
            self._cut_active = False
            if self._original_style is not None:
                vtk_iren.SetInteractorStyle(self._original_style)
                self._original_style = None
            # Remover observers de teclado
            for obs_id in self._key_observers:
                vtk_iren.RemoveObserver(obs_id)
            self._key_observers = []
            self._keys_down.clear()
            self.update_status("Plano de corte desactivado")
        else:
            bounds = self.grid.bounds
            self._cut_origin = np.array([
                (bounds[0]+bounds[1])/2, (bounds[2]+bounds[3])/2,
                (bounds[4]+bounds[5])/2])
            self._cut_rotation = np.eye(3)
            self._cut_scale = 1.0
            diag = np.sqrt((bounds[1]-bounds[0])**2 +
                           (bounds[3]-bounds[2])**2 +
                           (bounds[5]-bounds[4])**2)
            self._move_step = diag * 0.02
            self._cut_active = True
            self._original_style = vtk_iren.GetInteractorStyle()
            cut_style = CutPlaneStyle(self)
            cut_style.SetDefaultRenderer(self.plotter.renderer)
            vtk_iren.SetInteractorStyle(cut_style)

            # Observers para rastrear teclas mantenidas (Z/X/Y + rueda)
            self._keys_down = set()
            self._key_observers = [
                vtk_iren.AddObserver("KeyPressEvent",
                                     self.track_key_press, 1.0),
                vtk_iren.AddObserver("KeyReleaseEvent",
                                     self.track_key_release, 1.0),
            ]

            self.update_cut_plane_visual()
            self.update_status(
                "PLANO — Mantener Z/X/Y + rueda: rotar — "
                "Ctrl/Alt/Shift+rueda: mover Z/X/Y — "
                "Ctrl+Shift+rueda: escalar — [F] cortar — [T] cerrar")

    def on_wheel(self, direction):
        """Mueve o rota el plano de corte según la tecla mantenida y la rueda del mouse."""
        vtk_iren = self.get_vtk_interactor()

        ctrl = vtk_iren.GetControlKey()
        shift = vtk_iren.GetShiftKey()
        alt = vtk_iren.GetAltKey()

        if ctrl and shift:
            self._cut_scale *= (1.1 if direction > 0 else 0.9)
        elif ctrl:
            self._cut_origin[2] += direction * self._move_step
        elif alt:
            self._cut_origin[0] += direction * self._move_step
        elif shift:
            self._cut_origin[1] += direction * self._move_step
        elif 'z' in self._keys_down:
            self.rotate_cut_plane(2, direction * self._rot_step)
        elif 'x' in self._keys_down:
            self.rotate_cut_plane(0, direction * self._rot_step)
        elif 'y' in self._keys_down:
            self.rotate_cut_plane(1, direction * self._rot_step)
        else:
            # Sin tecla mantenida: la rueda no hace nada
            return
        self.update_cut_plane_visual()

    def execute_cut(self):
        """Ejecuta el corte de las celdas que intersectan el plano de corte."""
        if not self._cut_active:
            self.update_status("Primero activa el plano con [T]")
            return

        try:
            normal = self.get_cut_normal()
            origin = self._cut_origin.copy()
            plane_mesh, _ = self.build_cut_plane_mesh()

            # Determinar los ejes locales del plano de corte
            plane_pts = np.asarray(plane_mesh.points)
            v_i = plane_pts[1] - plane_pts[0]
            v_j = plane_pts[2] - plane_pts[0]
            len_i = np.linalg.norm(v_i)
            len_j = np.linalg.norm(v_j)
            if len_i == 0 or len_j == 0:
                self.update_status("Plano degenerado")
                return
            u_i = v_i / len_i
            u_j = v_j / len_j

            # Determinar qué celdas intersectan el plano de corte
            points = np.asarray(self.grid.points)
            point_dist = np.dot(points - origin, normal)
            cells_array = self.grid.cells
            n_cells = self.grid.n_cells
            stride = cells_array[0] + 1
            vert_ids = cells_array.reshape(n_cells, stride)[:, 1:]
            vert_dists = point_dist[vert_ids]
            straddles = (vert_dists.min(axis=1) < 0) & (vert_dists.max(axis=1) > 0)

            # Determinar qué celdas están dentro del rectángulo definido por el plano de corte
            grid_centers = self.grid.cell_centers().points
            v = grid_centers - plane_pts[0]
            in_rect = ((np.dot(v, u_i) >= 0) & (np.dot(v, u_i) <= len_i) &
                       (np.dot(v, u_j) >= 0) & (np.dot(v, u_j) <= len_j))

            # Crear máscara final de celdas a cortar
            cut_mask = straddles & in_rect

            cut_any = False
            self._cut_counter += 1

            # Guardar estado de la cámara para restaurarlo después del corte
            cam_pos = tuple(self.plotter.camera.position)
            cam_focal = tuple(self.plotter.camera.focal_point)
            cam_up = tuple(self.plotter.camera.up)
            cam_clip = tuple(self.plotter.camera.clipping_range)

            for key in list(self._group_keys_ordered):
                if not self.visible.get(key, True):
                    continue
                
                group_mask = self.group_ids == key
                to_separate = group_mask & cut_mask

                n_sep = int(np.sum(to_separate))
                if n_sep == 0 or n_sep == int(np.sum(group_mask)):
                    continue

                cut_any = True
                self.group_ids[to_separate] = np.nan

                print(f"  Cortado {format_value(key)}: "
                      f"{n_sep} celdas eliminadas")

            if not cut_any:
                self.update_status("El plano no intersecta ninguna celda")
                return

            self.refresh_all_actors()

            margin = max(len_i, len_j) * 0.01
            for key in list(self.surfaces.keys()):
                if not self.visible.get(key, True):
                    continue
                surface = self.surfaces.get(key)
                if surface is None or surface.n_cells == 0:
                    continue
                try:
                    sliced = surface.slice(normal=normal, origin=origin)
                    if sliced is None or sliced.n_points == 0:
                        continue
                    sv = np.asarray(sliced.points) - plane_pts[0]
                    keep = ((np.dot(sv, u_i) >= -margin) &
                            (np.dot(sv, u_i) <= len_i + margin) &
                            (np.dot(sv, u_j) >= -margin) &
                            (np.dot(sv, u_j) <= len_j + margin))
                    if not np.any(keep):
                        continue
                    clipped = sliced.extract_points(
                        np.where(keep)[0], adjacent_cells=True)
                    if clipped.n_cells > 0:
                        actor = self.plotter.add_mesh(
                            clipped, color="red", line_width=3.0,
                            name=f"cut_{self._cut_counter}_{id(surface)}",
                            pickable=False, reset_camera=False)
                        self._cut_line_actors.append(actor)
                except Exception:
                    pass

            self.plotter.camera.position = cam_pos
            self.plotter.camera.focal_point = cam_focal
            self.plotter.camera.up = cam_up
            self.plotter.camera.clipping_range = cam_clip
            self.plotter.render()

            n_groups = len(np.unique(
                self.group_ids[~np.isnan(self.group_ids)]))
            self.update_status(
                f"✓ Corte #{self._cut_counter} — {n_groups} grupos — "
                f"[S] navegar componentes — [F] cortar de nuevo")

        except Exception as e:
            self.update_status(f"Error al cortar: {e}")
            import traceback
            traceback.print_exc()

    def clear_cut_lines(self):
        """Elimina las líneas de corte de la escena."""
        for actor in self._cut_line_actors:
            self.plotter.remove_actor(actor)
        self._cut_line_actors.clear()