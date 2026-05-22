import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from pathlib import Path

try:
    import vtk
except ImportError:
    import vtkmodules.all as vtk

try:
    from mesh_repair import repair_surface
    HAS_REPAIR = True
except ImportError:
    HAS_REPAIR = False


def format_value(v):
    if v == 0:
        return "0"
    elif abs(v) < 1e-3 or abs(v) >= 1e4:
        return f"{v:.2e}"
    else:
        return f"{v:.4g}"

try:
    import vtkmodules.vtkInteractionStyle as _vtk_style
    _BaseStyle = _vtk_style.vtkInteractorStyleTrackballCamera
except (ImportError, AttributeError):
    import vtk
    _BaseStyle = vtk.vtkInteractorStyleTrackballCamera


class _CutPlaneStyle(_BaseStyle):
    def __init__(self, editor):
        super().__init__()
        self._editor = editor
        self.AddObserver("MouseWheelForwardEvent", self._wheel_forward)
        self.AddObserver("MouseWheelBackwardEvent", self._wheel_backward)

    def _wheel_forward(self, obj, event):
        self._editor._on_wheel(1)
        # NO llamar a super().OnMouseWheelForward() → sin zoom

    def _wheel_backward(self, obj, event):
        self._editor._on_wheel(-1)
        # NO llamar a super().OnMouseWheelBackward() → sin zoom


class AquiferEditor:

    WINDOW_W = 1400
    WINDOW_H = 900
    DIM_OPACITY = 0.3

    MODE_VIEW = "view"
    MODE_SELECT = "select"
    MODE_BROWSE = "browse"
    MODE_DESTINATION = "dest"

    def __init__(self, aquifer, prop="k", z_exag=1.0, decimate=0.0):
        self.prop = prop
        self.z_exag = z_exag
        self.decimate = decimate

        # ── 1. Grid completo ────────────────────────────────────────
        self.grid = aquifer.build_grid(prop=prop, z_exag=z_exag)

        # ── 2. Array de grupos ──────────────────────────────────────
        self.group_ids = self.grid.cell_data[prop].copy()

        # ── 3. Superficies ──────────────────────────────────────────
        self.surfaces = {}
        self.original_surfaces = {}
        self._rebuild_all_surfaces()

        # ── 4. Estado visual ────────────────────────────────────────
        self.actors = {}
        self.visible = {}
        self.clip_enabled = {}
        self.colors = {}

        group_keys = sorted(self.surfaces.keys())
        n = len(group_keys)
        cmap = plt.get_cmap("tab10") if n <= 10 else plt.get_cmap("tab20")

        for idx, key in enumerate(group_keys):
            self.colors[key] = cmap(idx / max(n - 1, 1))[:3]
            self.visible[key] = True
            self.clip_enabled[key] = False

        # ── 5. Plano de corte por propiedad (checkboxes) ───────────
        self.clip_normal = None
        self.clip_origin = None
        self._plane_widget_id = None

        # ── 6. Selección de componentes ─────────────────────────────
        self.mode = self.MODE_VIEW
        self.source_group = None
        self.components = []
        self.component_idx = 0
        self.target_group = None
        self._highlight_actor = None

        # ── 7. Plano de corte libre (tecla T) ───────────────────────
        self._cut_active = False
        self._cut_origin = None       # Centro del plano
        self._cut_euler = np.zeros(3)  # Rotación [rx, ry, rz] en grados
        self._cut_scale = 1.0          # Escala del plano visual
        self._cut_rot_axis = 2         # 0=X, 1=Y, 2=Z (default Z)
        self._cut_actor = None         # Actor del plano visual
        self._cut_border_actor = None  # Bordes del plano
        self._cut_line_actors = []     # Líneas rojas de cortes realizados
        self._cut_counter = 0          # Contador para IDs únicos de cortes

        # Pasos de movimiento (se calculan al activar según tamaño del modelo)
        self._move_step = 1.0
        self._rot_step = 5.0  # Grados por tick de rueda

        # ── 8. Plotter ──────────────────────────────────────────────
        self.plotter = None
        self._group_keys_ordered = group_keys
        self._status_actor = None
        self._original_style = None  # Interactor style original (se guarda al activar corte)

    # ═══════════════════════════════════════════════════════════════════
    # Superficies
    # ═══════════════════════════════════════════════════════════════════

    def _rebuild_all_surfaces(self):
        unique_groups = np.unique(self.group_ids[~np.isnan(self.group_ids)])

        self.surfaces.clear()
        self.original_surfaces.clear()

        for val in np.sort(unique_groups):
            indices = np.where(self.group_ids == val)[0]
            if indices.size == 0:
                continue

            sub = self.grid.extract_cells(indices)
            surface = sub.extract_surface(algorithm=None).triangulate()

            if self.decimate > 0 and HAS_REPAIR:
                surface = repair_surface(surface, decimate=self.decimate,
                                         verbose=False)

            self.surfaces[val] = surface
            self.original_surfaces[val] = surface.copy()

    def _compute_components(self, group_key):
        group_mask = self.group_ids == group_key
        group_cell_indices = np.where(group_mask)[0]
        if group_cell_indices.size == 0:
            return []

        sub_grid = self.grid.extract_cells(group_cell_indices)
        connected = sub_grid.connectivity(extraction_mode='all')
        region_ids = connected.cell_data["RegionId"]
        unique_regions = np.unique(region_ids)

        components = []
        for rid in np.sort(unique_regions):
            region_mask = region_ids == rid
            grid_cells = group_cell_indices[region_mask]
            comp_grid = connected.extract_cells(np.where(region_mask)[0])
            comp_surface = comp_grid.extract_surface(algorithm=None).triangulate()
            components.append({
                "grid_indices": grid_cells,
                "surface": comp_surface,
                "n_cells": len(grid_cells),
            })
        components.sort(key=lambda c: c["n_cells"], reverse=True)
        return components

    # ═══════════════════════════════════════════════════════════════════
    # Renderizado
    # ═══════════════════════════════════════════════════════════════════

    def _add_mesh_to_plotter(self, key):
        if key in self.actors:
            self.plotter.remove_actor(self.actors[key])
        surface = self.surfaces.get(key)
        if surface is None or surface.n_cells == 0:
            self.actors.pop(key, None)
            return
        actor = self.plotter.add_mesh(
            surface,
            color=self.colors.get(key, (0.5, 0.5, 0.5)),
            show_edges=False, opacity=1.0,
            name=f"group_{format_value(key)}",
        )
        actor.SetVisibility(self.visible.get(key, True))
        self.actors[key] = actor

    def _refresh_all_actors(self):
        self._rebuild_all_surfaces()
        for key in list(self.actors.keys()):
            if key not in self.surfaces:
                self.plotter.remove_actor(self.actors.pop(key))
        for key in self.surfaces:
            if key not in self.colors:
                n = max(len(self._group_keys_ordered), 1)
                cmap = plt.get_cmap("tab10") if n <= 10 else plt.get_cmap("tab20")
                self.colors[key] = cmap(len(self.colors) / max(n - 1, 1))[:3]
                self.visible[key] = True
                self.clip_enabled[key] = False
            self._add_mesh_to_plotter(key)
            if self.clip_enabled.get(key, False) and self.clip_normal is not None:
                self._apply_clip_to_group(key)

    def _remove_highlight(self):
        if self._highlight_actor is not None:
            self.plotter.remove_actor(self._highlight_actor)
            self._highlight_actor = None

    def _highlight_component(self):
        self._remove_highlight()
        if not self.components or self.component_idx >= len(self.components):
            return
        comp = self.components[self.component_idx]
        self._highlight_actor = self.plotter.add_mesh(
            comp["surface"], color="white", style="wireframe",
            line_width=1.5, opacity=1.0, name="highlight", pickable=False)
        for gkey, actor in self.actors.items():
            p = actor.GetProperty()
            p.SetOpacity(0.7 if gkey == self.source_group else self.DIM_OPACITY)
        self.plotter.render()

    def _restore_opacities(self):
        for key, actor in self.actors.items():
            actor.GetProperty().SetOpacity(1.0)
        self.plotter.render()

    # ═══════════════════════════════════════════════════════════════════
    # Visibilidad
    # ═══════════════════════════════════════════════════════════════════

    def _toggle_visibility(self, key, state):
        self.visible[key] = state
        if key in self.actors:
            self.actors[key].SetVisibility(state)

    # ═══════════════════════════════════════════════════════════════════
    # Plano de corte por propiedad (checkboxes)
    # ═══════════════════════════════════════════════════════════════════

    def _toggle_clip(self, key, state):
        self.clip_enabled[key] = state
        if state and self.clip_normal is not None:
            self._apply_clip_to_group(key)
        elif not state:
            self.surfaces[key] = self.original_surfaces[key].copy()
            self._add_mesh_to_plotter(key)
        any_clipped = any(self.clip_enabled.values())
        if any_clipped and self._plane_widget_id is None:
            self._show_plane_widget()
        elif not any_clipped and self._plane_widget_id is not None:
            self._hide_plane_widget()

    def _apply_clip_to_group(self, key):
        original = self.original_surfaces.get(key)
        if original is None or self.clip_normal is None:
            return
        try:
            clipped = original.clip(normal=self.clip_normal,
                                    origin=self.clip_origin, invert=False)
            self.surfaces[key] = clipped if clipped.n_cells > 0 else original.copy()
        except Exception:
            self.surfaces[key] = original.copy()
        self._add_mesh_to_plotter(key)

    def _on_plane_moved(self, normal, origin):
        self.clip_normal = np.array(normal)
        self.clip_origin = np.array(origin)
        for key in self.clip_enabled:
            if self.clip_enabled[key]:
                self._apply_clip_to_group(key)

    def _show_plane_widget(self):
        self._plane_widget_id = self.plotter.add_plane_widget(
            self._on_plane_moved, bounds=self.grid.bounds,
            factor=1.1, normal="z", color="red",
            tubing=True, interaction_event="always")

    def _hide_plane_widget(self):
        if self._plane_widget_id is not None:
            self.plotter.clear_plane_widgets()
            self._plane_widget_id = None
            self.clip_normal = None
            self.clip_origin = None

    # ═══════════════════════════════════════════════════════════════════
    # Selección y reasignación de componentes
    # ═══════════════════════════════════════════════════════════════════

    def _enter_select_mode(self):
        if self.mode != self.MODE_VIEW:
            self._cancel()
            return
        self.mode = self.MODE_SELECT
        group_list = " | ".join(
            f"[{i+1}] {format_value(k)}"
            for i, k in enumerate(self._group_keys_ordered[:9])
        )
        self._update_status(f"SELECCIÓN — Elige grupo fuente: {group_list}")

    def _on_number_key(self, index):
        if self.mode == self.MODE_SELECT:
            self._select_source_group(index)
        elif self.mode == self.MODE_DESTINATION:
            self._select_destination(index)

    def _select_source_group(self, index):
        if index >= len(self._group_keys_ordered):
            return
        self.source_group = self._group_keys_ordered[index]
        self._update_status(f"Calculando componentes...")
        self.plotter.render()
        try:
            self.components = self._compute_components(self.source_group)
        except Exception as e:
            self._update_status(f"Error: {e}")
            self.mode = self.MODE_VIEW
            return
        if not self.components:
            self._update_status("Grupo sin celdas")
            self.mode = self.MODE_VIEW
            return
        self.component_idx = 0
        self.mode = self.MODE_BROWSE
        self._highlight_component()
        self._show_browse_status()

    def _next_component(self):
        if self.mode != self.MODE_BROWSE or not self.components:
            return
        self.component_idx = (self.component_idx + 1) % len(self.components)
        self._highlight_component()
        self._show_browse_status()

    def _prev_component(self):
        if self.mode != self.MODE_BROWSE or not self.components:
            return
        self.component_idx = (self.component_idx - 1) % len(self.components)
        self._highlight_component()
        self._show_browse_status()

    def _show_browse_status(self):
        comp = self.components[self.component_idx]
        total = len(self.components)
        self._update_status(
            f"{self.prop}={format_value(self.source_group)} — "
            f"Componente {self.component_idx+1}/{total} ({comp['n_cells']} celdas) — "
            f"[N/P] navegar [D] destino [Esc] cancelar"
        )

    def _enter_destination_mode(self):
        if self.mode != self.MODE_BROWSE:
            return
        self.mode = self.MODE_DESTINATION
        group_list = " | ".join(
            f"[{i+1}] {format_value(k)}"
            for i, k in enumerate(self._group_keys_ordered[:9])
            if k != self.source_group
        )
        self._update_status(f"DESTINO — Elige grupo: {group_list}")

    def _select_destination(self, index):
        if index >= len(self._group_keys_ordered):
            return
        target = self._group_keys_ordered[index]
        if target == self.source_group:
            self._update_status("No puedes elegir el mismo grupo")
            return
        self.target_group = target
        comp = self.components[self.component_idx]
        self._update_status(
            f"{format_value(self.source_group)} → {format_value(self.target_group)} "
            f"({comp['n_cells']} celdas) — [G] confirmar"
        )
        self.mode = self.MODE_BROWSE

    def _confirm_reassign(self):
        if self.mode != self.MODE_BROWSE or self.target_group is None:
            self._update_status("Elige destino con [D] + número")
            return
        if not self.components:
            return

        comp = self.components[self.component_idx]
        source = self.source_group
        target = self.target_group
        n = len(comp["grid_indices"])

        self.group_ids[comp["grid_indices"]] = target

        if np.sum(self.group_ids == source) == 0:
            for d in [self.visible, self.clip_enabled, self.colors]:
                d.pop(source, None)
            if source in self._group_keys_ordered:
                self._group_keys_ordered.remove(source)

        self._remove_highlight()
        self._restore_opacities()
        self.source_group = self.target_group = None
        self.components = []
        self.mode = self.MODE_VIEW
        self._refresh_all_actors()
        self._update_status(
            f"✓ {n} celdas: {format_value(source)} → {format_value(target)}"
        )

    def _cancel(self):
        self._remove_highlight()
        self._restore_opacities()
        self.mode = self.MODE_VIEW
        self.source_group = self.target_group = None
        self.components = []
        self._update_status(f"{len(self._group_keys_ordered)} grupos — [S] seleccionar")

    # ═══════════════════════════════════════════════════════════════════
    # Plano de corte libre (tecla T)
    # ═══════════════════════════════════════════════════════════════════

    def _normal_from_euler(self):
        """Calcula la normal del plano a partir de los ángulos de Euler."""
        rx, ry, rz = np.radians(self._cut_euler)
        # Empezamos con normal = [0, 0, 1] y rotamos
        # Rotación X
        cx, sx = np.cos(rx), np.sin(rx)
        # Rotación Y
        cy, sy = np.cos(ry), np.sin(ry)
        # Rotación Z
        cz, sz = np.cos(rz), np.sin(rz)

        # Matriz de rotación ZYX
        n = np.array([
            sy,
            -cy * sx,
            cx * cy,
        ])
        return n / np.linalg.norm(n)

    def _build_cut_plane_mesh(self):
        """
        Crea la geometría del plano de corte como un rectángulo.
        El tamaño base es proporcional al modelo.
        """
        bounds = self.grid.bounds
        dx = bounds[1] - bounds[0]
        dy = bounds[3] - bounds[2]
        size = max(dx, dy) * 0.8 * self._cut_scale

        normal = self._normal_from_euler()

        plane = pv.Plane(
            center=self._cut_origin,
            direction=normal,
            i_size=size,
            j_size=size,
            i_resolution=1,
            j_resolution=1,
        )
        return plane, normal

    def _update_cut_plane_visual(self):
        """Actualiza la posición/rotación/escala del plano en el visor."""
        if not self._cut_active:
            return

        # Eliminar anteriores
        if self._cut_actor is not None:
            self.plotter.remove_actor(self._cut_actor)
        if self._cut_border_actor is not None:
            self.plotter.remove_actor(self._cut_border_actor)

        plane, normal = self._build_cut_plane_mesh()

        # Plano semitransparente
        self._cut_actor = self.plotter.add_mesh(
            plane, color="red", opacity=0.15,
            name="cut_plane", pickable=False,
        )

        # Bordes del plano
        edges = plane.extract_feature_edges(
            boundary_edges=True, feature_edges=False,
            manifold_edges=False, non_manifold_edges=False,
        )
        self._cut_border_actor = self.plotter.add_mesh(
            edges, color="red", line_width=2.0,
            name="cut_border", pickable=False,
        )

        self.plotter.render()

    def _toggle_cut_plane(self):
        # Acceso al interactor VTK
        iren = self.plotter.iren
        if hasattr(iren, 'interactor') and iren.interactor is not None:
            vtk_iren = iren.interactor
        elif hasattr(iren, '_iren'):
            vtk_iren = iren._iren
        else:
            vtk_iren = iren

        if self._cut_active:
            # ── Desactivar ──────────────────────────────────────────
            if self._cut_actor is not None:
                self.plotter.remove_actor(self._cut_actor)
                self._cut_actor = None
            if self._cut_border_actor is not None:
                self.plotter.remove_actor(self._cut_border_actor)
                self._cut_border_actor = None
            self._cut_active = False

            # Restaurar interactor style original → zoom con rueda vuelve
            if self._original_style is not None:
                vtk_iren.SetInteractorStyle(self._original_style)
                self._original_style = None

            self._update_status("Plano de corte desactivado")
        else:
            # ── Activar ─────────────────────────────────────────────
            bounds = self.grid.bounds
            self._cut_origin = np.array([
                (bounds[0] + bounds[1]) / 2,
                (bounds[2] + bounds[3]) / 2,
                (bounds[4] + bounds[5]) / 2,
            ])
            self._cut_euler = np.zeros(3)
            self._cut_scale = 1.0
            self._cut_rot_axis = 2  # Z

            diag = np.sqrt(
                (bounds[1]-bounds[0])**2 +
                (bounds[3]-bounds[2])**2 +
                (bounds[5]-bounds[4])**2
            )
            self._move_step = diag * 0.02

            # Guardar interactor style original y cambiar a nuestro custom
            # que redirige la rueda al plano (sin zoom)
            self._original_style = vtk_iren.GetInteractorStyle()
            vtk_iren.SetInteractorStyle(_CutPlaneStyle(self))

            self._cut_active = True
            self._update_cut_plane_visual()

            axis_names = {0: "X", 1: "Y", 2: "Z"}
            self._update_status(
                f"PLANO DE CORTE — Eje: {axis_names[self._cut_rot_axis]} — "
                f"[Z/X/Y] eje rot. — Rueda: rotar — "
                f"Ctrl/Alt/Shift+rueda: mover — Ctrl+Shift+rueda: escalar — "
                f"[F] cortar — [T] cerrar"
            )

    def _set_cut_rot_axis(self, axis):
        if not self._cut_active:
            return
        self._cut_rot_axis = axis
        axis_names = {0: "X", 1: "Y", 2: "Z"}
        self._update_status(
            f"Eje de rotación: {axis_names[axis]} — Rueda para rotar"
        )

    def _on_wheel(self, direction):
        # Obtener estado de modificadores
        iren = self.plotter.iren
        if hasattr(iren, 'interactor') and iren.interactor is not None:
            vtk_iren = iren.interactor
        elif hasattr(iren, '_iren'):
            vtk_iren = iren._iren
        else:
            vtk_iren = iren

        ctrl = vtk_iren.GetControlKey()
        shift = vtk_iren.GetShiftKey()
        alt = vtk_iren.GetAltKey()

        if ctrl and shift:
            # Escalar
            factor = 1.1 if direction > 0 else 0.9
            self._cut_scale *= factor
        elif ctrl:
            # Mover en Z
            self._cut_origin[2] += direction * self._move_step
        elif alt:
            # Mover en X
            self._cut_origin[0] += direction * self._move_step
        elif shift:
            # Mover en Y
            self._cut_origin[1] += direction * self._move_step
        else:
            # Rotar
            self._cut_euler[self._cut_rot_axis] += direction * self._rot_step

        self._update_cut_plane_visual()

    def _execute_cut(self):
        if not self._cut_active:
            self._update_status("Primero activa el plano con [T]")
            return

        try:
            normal = self._normal_from_euler()
            origin = self._cut_origin.copy()
            plane_mesh, _ = self._build_cut_plane_mesh()

            # ── 1. Sistema de coordenadas del rectángulo ────────────
            plane_pts = np.asarray(plane_mesh.points)
            v_i = plane_pts[1] - plane_pts[0]
            v_j = plane_pts[2] - plane_pts[0]
            len_i = np.linalg.norm(v_i)
            len_j = np.linalg.norm(v_j)
            if len_i == 0 or len_j == 0:
                self._update_status("Plano degenerado")
                return
            u_i = v_i / len_i
            u_j = v_j / len_j

            # ── 2. ¿El plano atraviesa la celda? ───────────────────
            points = np.asarray(self.grid.points)
            point_dist = np.dot(points - origin, normal)

            cells_array = self.grid.cells
            n_cells = self.grid.n_cells
            n_verts_per_cell = cells_array[0]
            stride = n_verts_per_cell + 1
            vert_ids = cells_array.reshape(n_cells, stride)[:, 1:]
            vert_dists = point_dist[vert_ids]
            straddles_plane = (vert_dists.min(axis=1) < 0) & (vert_dists.max(axis=1) > 0)

            # ── 3. ¿Centro dentro del rectángulo? ──────────────────
            grid_centers = self.grid.cell_centers().points
            v = grid_centers - plane_pts[0]
            proj_i = np.dot(v, u_i)
            proj_j = np.dot(v, u_j)
            signed_dist = np.dot(grid_centers - origin, normal)

            in_rect = (
                (proj_i >= 0) & (proj_i <= len_i) &
                (proj_j >= 0) & (proj_j <= len_j)
            )

            # ── 4. Máscara ESTRICTA: todas las celdas atravesadas ────
            # Para separar el grupo en dos componentes conexos,
            # hay que remover TODAS las celdas que el plano atraviesa
            # (ambos lados), creando una brecha completa.
            cut_mask = straddles_plane & in_rect

            # ── 5. Cortar grupos visibles ───────────────────────────
            cut_any = False
            self._cut_counter += 1

            cam = self.plotter.camera
            cam_state = (cam.position, cam.focal_point,
                         cam.up, cam.clipping_range)

            for key in list(self._group_keys_ordered):
                if not self.visible.get(key, True):
                    continue

                group_mask = self.group_ids == key
                negative_mask = group_mask & cut_mask

                n_neg = int(np.sum(negative_mask))
                if n_neg == 0:
                    continue

                n_remaining = int(np.sum(group_mask)) - n_neg
                if n_remaining == 0:
                    continue

                cut_any = True

                # Nuevo ID negativo, MISMO color que el grupo original
                new_key = -(self._cut_counter * 100
                            + len(self._group_keys_ordered))
                self.group_ids[negative_mask] = float(new_key)

                self._group_keys_ordered.append(new_key)
                self.colors[new_key] = self.colors[key]
                self.visible[new_key] = True
                self.clip_enabled[new_key] = False

                print(f"  Cortado {format_value(key)}: "
                      f"{n_neg} celdas separadas → grupo {new_key}")

            if not cut_any:
                self._update_status(
                    "El plano no intersecta ninguna celda visible")
                return

            # ── 6. Reconstruir ──────────────────────────────────────
            self._refresh_all_actors()
            cam.position = cam_state[0]
            cam.focal_point = cam_state[1]
            cam.up = cam_state[2]
            cam.clipping_range = cam_state[3]

            # ── 7. Línea roja recortada al rectángulo ───────────────
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
                    # Filtrar puntos fuera del rectángulo
                    sv = np.asarray(sliced.points) - plane_pts[0]
                    si = np.dot(sv, u_i)
                    sj = np.dot(sv, u_j)
                    keep = ((si >= -margin) & (si <= len_i + margin) &
                            (sj >= -margin) & (sj <= len_j + margin))
                    if not np.any(keep):
                        continue
                    clipped = sliced.extract_points(
                        np.where(keep)[0], adjacent_cells=True)
                    if clipped.n_cells > 0:
                        actor = self.plotter.add_mesh(
                            clipped, color="red", line_width=3.0,
                            name=f"cut_{self._cut_counter}_{id(surface)}",
                            pickable=False)
                        self._cut_line_actors.append(actor)
                except Exception:
                    pass

            self.plotter.render()

            n_groups = len(np.unique(
                self.group_ids[~np.isnan(self.group_ids)]))
            self._update_status(
                f"✓ Corte #{self._cut_counter} — {n_groups} grupos — "
                f"[S] navegar componentes — [F] cortar de nuevo"
            )

        except Exception as e:
            self._update_status(f"Error al cortar: {e}")
            import traceback
            traceback.print_exc()

    def _clear_cut_lines(self):
        """Elimina todas las líneas de corte rojas."""
        for actor in self._cut_line_actors:
            self.plotter.remove_actor(actor)
        self._cut_line_actors.clear()

    # ═══════════════════════════════════════════════════════════════════
    # Exportación
    # ═══════════════════════════════════════════════════════════════════

    def _export_visible(self):
        out_dir = Path("export/editor")
        out_dir.mkdir(parents=True, exist_ok=True)

        exported = 0
        for key in self._group_keys_ordered:
            if not self.visible.get(key, False):
                continue
            surface = self.surfaces.get(key)
            if surface is None or surface.n_cells == 0:
                continue
            fname = f"{self.prop}_{format_value(key)}.ply"
            out_path = out_dir / fname
            surface.save(str(out_path))
            print(f"  → {out_path}  ({surface.n_cells} caras)")
            exported += 1

        self._update_status(f"Exportados {exported} archivos en {out_dir}/")

    # ═══════════════════════════════════════════════════════════════════
    # UI
    # ═══════════════════════════════════════════════════════════════════

    def _update_status(self, text):
        if self._status_actor is not None:
            self.plotter.remove_actor(self._status_actor)
        self._status_actor = self.plotter.add_text(
            text, position=(10, 8), font_size=9,
            color="black", name="status_text",
        )

    def _setup_ui(self):
        checkbox_size = 22
        row_height = checkbox_size + 10
        y_start = self.WINDOW_H - 50

        for idx, key in enumerate(self._group_keys_ordered):
            color = self.colors[key]
            y_pos = y_start - idx * row_height

            def make_vis_cb(k):
                def cb(state): self._toggle_visibility(k, state)
                return cb

            self.plotter.add_checkbox_button_widget(
                make_vis_cb(key), value=True,
                position=(10, y_pos), size=checkbox_size,
                color_on=color, color_off="grey")

            def make_clip_cb(k):
                def cb(state): self._toggle_clip(k, state)
                return cb

            self.plotter.add_checkbox_button_widget(
                make_clip_cb(key), value=False,
                position=(40, y_pos), size=checkbox_size - 4,
                color_on="red", color_off="darkgrey")

            n_cells = self.surfaces[key].n_cells if key in self.surfaces else 0
            self.plotter.add_text(
                f"{idx+1}: {self.prop}={format_value(key)}  ({n_cells})",
                position=(68, y_pos + 2), font_size=8, color="black")

        # Instrucciones
        self.plotter.add_text(
            "[S] Seleccionar  [N/P] Navegar  [D+num] Destino  [G] Confirmar  "
            "[T] Plano de corte  [F] Cortar  [Esc] Cancelar  "
            "[C] Reset  [E] Exportar",
            position=(10, 30), font_size=7, color="grey")

        # Atajos: selección
        self.plotter.add_key_event("s", self._enter_select_mode)
        self.plotter.add_key_event("n", self._next_component)
        self.plotter.add_key_event("p", self._prev_component)
        self.plotter.add_key_event("d", self._enter_destination_mode)
        self.plotter.add_key_event("g", self._confirm_reassign)

        # Atajos: plano de corte
        self.plotter.add_key_event("t", self._toggle_cut_plane)
        self.plotter.add_key_event("f", self._execute_cut)
        self.plotter.add_key_event("z", lambda: self._set_cut_rot_axis(2))
        self.plotter.add_key_event("x", lambda: self._set_cut_rot_axis(0))
        self.plotter.add_key_event("y", lambda: self._set_cut_rot_axis(1))

        # Atajos: otros
        self.plotter.add_key_event("e", self._export_visible)
        self.plotter.add_key_event("Escape", self._cancel)

        def reset_all():
            # Reset clips
            for k in list(self.clip_enabled.keys()):
                self.clip_enabled[k] = False
                if k in self.original_surfaces:
                    self.surfaces[k] = self.original_surfaces[k].copy()
                    self._add_mesh_to_plotter(k)
            self._hide_plane_widget()
            # Reset cut lines
            self._clear_cut_lines()
            self._update_status("Reset completado")
        self.plotter.add_key_event("c", reset_all)

        # Números 1-9
        for i in range(min(9, len(self._group_keys_ordered))):
            def make_key_cb(idx):
                def cb(): self._on_number_key(idx)
                return cb
            self.plotter.add_key_event(str(i + 1), make_key_cb(i))

        self._update_status(
            f"{len(self._group_keys_ordered)} grupos — "
            f"[S] seleccionar  [T] plano de corte"
        )

    # ═══════════════════════════════════════════════════════════════════
    # Main
    # ═══════════════════════════════════════════════════════════════════

    def show(self):
        self.plotter = pv.Plotter(window_size=[self.WINDOW_W, self.WINDOW_H])
        self.plotter.background_color = "white"

        for key in self._group_keys_ordered:
            self._add_mesh_to_plotter(key)

        self._setup_ui()
        self.plotter.add_axes(xlabel="X (m)", ylabel="Y (m)", zlabel="Z (m)")

        n = len(self._group_keys_ordered)
        self.plotter.show(title=f"Aquifer Editor — {self.prop}: {n} grupos")