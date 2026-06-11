"""
aquifer_editor.py — Editor interactivo de mallas de acuíferos.

Funcionalidades:
  1. Checkboxes de visibilidad por propiedad
  2. Navegación por componentes conexos (S → 1-9 → N/P → D+num → G)
  3. Plano de corte libre (T): rotar, mover, escalar, cortar (F)
  4. Exportación (E)

Uso:
    python edit_model.py ../modelo --z-exag 10
"""

import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from pathlib import Path

try:
    from mesh_repair import repair_surface
    HAS_REPAIR = True
except ImportError:
    HAS_REPAIR = False

try:
    import vtkmodules.vtkInteractionStyle as _vtk_style
    _BaseStyle = _vtk_style.vtkInteractorStyleTrackballCamera
except (ImportError, AttributeError):
    import vtk
    _BaseStyle = vtk.vtkInteractorStyleTrackballCamera


def format_value(v):
    if v == 0:
        return "0"
    elif abs(v) < 1e-3 or abs(v) >= 1e4:
        return f"{v:.2e}"
    else:
        return f"{v:.4g}"


class _CutPlaneStyle(_BaseStyle):
    """Redirige la rueda del mouse al plano de corte en vez de zoom."""
    def __init__(self, editor):
        super().__init__()
        self._editor = editor
        self.AddObserver("MouseWheelForwardEvent", self._fwd)
        self.AddObserver("MouseWheelBackwardEvent", self._bwd)

    def _fwd(self, obj, event):
        self._editor._on_wheel(1)

    def _bwd(self, obj, event):
        self._editor._on_wheel(-1)


class AquiferEditor:

    WINDOW_W = 1400
    WINDOW_H = 900
    DIM_OPACITY = 0.3

    MODE_VIEW = "view"
    MODE_SELECT = "select"
    MODE_BROWSE = "browse"
    MODE_DESTINATION = "dest"

    def __init__(self, aquifer, prop="k", z_exag=1.0, decimate=0.0, tolerance=0.15):
        self.prop = prop
        self.z_exag = z_exag
        self.decimate = decimate
        self.tolerance = tolerance  # mm de holgura por lado

        # ── 1. Grid completo ────────────────────────────────────────
        self.grid = aquifer.build_grid(prop=prop, z_exag=z_exag)

        # ── 2. Array de grupos ──────────────────────────────────────
        self.group_ids = self.grid.cell_data[prop].copy()

        # ── 3. Superficies ──────────────────────────────────────────
        self.surfaces = {}
        self._rebuild_all_surfaces()

        # ── 4. Estado visual ────────────────────────────────────────
        self.actors = {}
        self.visible = {}
        self.colors = {}

        group_keys = sorted(self.surfaces.keys())
        n = len(group_keys)
        cmap = plt.get_cmap("tab10") if n <= 10 else plt.get_cmap("tab20")

        for idx, key in enumerate(group_keys):
            self.colors[key] = cmap(idx / max(n - 1, 1))[:3]
            self.visible[key] = True

        # ── 5. Selección de componentes ─────────────────────────────
        self.mode = self.MODE_VIEW
        self.source_group = None
        self.components = []
        self.component_idx = 0
        self.target_group = None
        self._highlight_actor = None

        # ── 6. Plano de corte libre (tecla T) ───────────────────────
        self._cut_active = False
        self._cut_origin = None
        self._cut_rotation = np.eye(3)
        self._cut_scale = 1.0
        self._cut_rot_axis = 2
        self._cut_actor = None
        self._cut_border_actor = None
        self._cut_line_actors = []
        self._cut_counter = 0
        self._move_step = 1.0
        self._rot_step = 5.0
        self._original_style = None

        # ── 7. Tolerancia para impresión (tecla L) ──────────────────
        self._prepared = False

        # ── 7. Preparación para impresión (tecla P) ─────────────────
        self._prepared = False
        # Superficie del grid completo: solo caras exteriores.
        # Se usa para distinguir caras de frontera (entre grupos)
        # de caras exteriores (borde del modelo).
        self._full_surface = self.grid.extract_surface(algorithm=None)

        # ── 8. Plotter ──────────────────────────────────────────────
        self.plotter = None
        self._group_keys_ordered = group_keys
        self._status_actor = None

    # ═══════════════════════════════════════════════════════════════════
    # Superficies
    # ═══════════════════════════════════════════════════════════════════

    def _rebuild_all_surfaces(self):
        unique_groups = np.unique(self.group_ids[~np.isnan(self.group_ids)])
        self.surfaces.clear()

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

    def _compute_components(self, group_key):
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
            surface, color=self.colors.get(key, (0.5, 0.5, 0.5)),
            show_edges=False, opacity=1.0,
            name=f"group_{format_value(key)}")
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
            self._add_mesh_to_plotter(key)

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
    # Selección y reasignación de componentes
    # ═══════════════════════════════════════════════════════════════════

    def _enter_select_mode(self):
        if self.mode != self.MODE_VIEW:
            self._cancel()
            return
        self.mode = self.MODE_SELECT
        group_list = " | ".join(
            f"[{i+1}] {format_value(k)}"
            for i, k in enumerate(self._group_keys_ordered[:9]))
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
        self._update_status("Calculando componentes...")
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
            f"[N/P] navegar [D] destino [Esc] cancelar")

    def _enter_destination_mode(self):
        if self.mode != self.MODE_BROWSE:
            return
        self.mode = self.MODE_DESTINATION
        group_list = " | ".join(
            f"[{i+1}] {format_value(k)}"
            for i, k in enumerate(self._group_keys_ordered[:9])
            if k != self.source_group)
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
            f"({comp['n_cells']} celdas) — [G] confirmar")
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
            for d in [self.visible, self.colors]:
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
            f"✓ {n} celdas: {format_value(source)} → {format_value(target)}")

    def _cancel(self):
        self._remove_highlight()
        self._restore_opacities()
        self.mode = self.MODE_VIEW
        self.source_group = self.target_group = None
        self.components = []
        self._update_status(
            f"{len(self._group_keys_ordered)} grupos — [S] seleccionar")

    # ═══════════════════════════════════════════════════════════════════
    # Plano de corte libre (tecla T)
    # ═══════════════════════════════════════════════════════════════════

    def _get_cut_normal(self):
        return self._cut_rotation[:, 2].copy()

    def _rotate_cut_plane(self, axis, angle_deg):
        angle = np.radians(angle_deg)
        c, s = np.cos(angle), np.sin(angle)
        if axis == 0:
            R = np.array([[1,0,0],[0,c,-s],[0,s,c]])
        elif axis == 1:
            R = np.array([[c,0,s],[0,1,0],[-s,0,c]])
        else:
            R = np.array([[c,-s,0],[s,c,0],[0,0,1]])
        self._cut_rotation = R @ self._cut_rotation

    def _build_cut_plane_mesh(self):
        bounds = self.grid.bounds
        dx = bounds[1] - bounds[0]
        dy = bounds[3] - bounds[2]
        size = max(dx, dy) * 0.8 * self._cut_scale
        normal = self._get_cut_normal()
        plane = pv.Plane(center=self._cut_origin, direction=normal,
                         i_size=size, j_size=size,
                         i_resolution=1, j_resolution=1)
        return plane, normal

    def _update_cut_plane_visual(self):
        if not self._cut_active:
            return
        if self._cut_actor is not None:
            self.plotter.remove_actor(self._cut_actor)
        if self._cut_border_actor is not None:
            self.plotter.remove_actor(self._cut_border_actor)

        plane, _ = self._build_cut_plane_mesh()
        self._cut_actor = self.plotter.add_mesh(
            plane, color="red", opacity=0.15,
            name="cut_plane", pickable=False)
        edges = plane.extract_feature_edges(
            boundary_edges=True, feature_edges=False,
            manifold_edges=False, non_manifold_edges=False)
        self._cut_border_actor = self.plotter.add_mesh(
            edges, color="red", line_width=2.0,
            name="cut_border", pickable=False)
        self.plotter.render()

    def _toggle_cut_plane(self):
        """Tecla T: activa/desactiva el plano de corte libre."""
        # Obtener interactor VTK
        iren = self.plotter.iren
        if hasattr(iren, 'interactor') and iren.interactor is not None:
            vtk_iren = iren.interactor
        elif hasattr(iren, '_iren'):
            vtk_iren = iren._iren
        else:
            vtk_iren = iren

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
            self._update_status("Plano de corte desactivado")
        else:
            bounds = self.grid.bounds
            self._cut_origin = np.array([
                (bounds[0]+bounds[1])/2, (bounds[2]+bounds[3])/2,
                (bounds[4]+bounds[5])/2])
            self._cut_rotation = np.eye(3)
            self._cut_scale = 1.0
            self._cut_rot_axis = 2
            diag = np.sqrt((bounds[1]-bounds[0])**2 +
                           (bounds[3]-bounds[2])**2 +
                           (bounds[5]-bounds[4])**2)
            self._move_step = diag * 0.02
            self._cut_active = True
            self._original_style = vtk_iren.GetInteractorStyle()
            vtk_iren.SetInteractorStyle(_CutPlaneStyle(self))
            self._update_cut_plane_visual()
            self._update_status(
                "PLANO — [Z/X/Y] eje rot. — Rueda: rotar — "
                "Ctrl/Alt/Shift+rueda: mover Z/X/Y — "
                "Ctrl+Shift+rueda: escalar — [F] cortar — [T] cerrar")

    def _set_cut_rot_axis(self, axis):
        if not self._cut_active:
            return
        self._cut_rot_axis = axis
        names = {0: "X", 1: "Y", 2: "Z"}
        self._update_status(f"Eje de rotación: {names[axis]}")

    def _on_wheel(self, direction):
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
            self._cut_scale *= (1.1 if direction > 0 else 0.9)
        elif ctrl:
            self._cut_origin[2] += direction * self._move_step
        elif alt:
            self._cut_origin[0] += direction * self._move_step
        elif shift:
            self._cut_origin[1] += direction * self._move_step
        else:
            self._rotate_cut_plane(self._cut_rot_axis,
                                   direction * self._rot_step)
        self._update_cut_plane_visual()

    def _execute_cut(self):
        """Tecla F: corta las celdas que el plano atraviesa."""
        if not self._cut_active:
            self._update_status("Primero activa el plano con [T]")
            return

        try:
            normal = self._get_cut_normal()
            origin = self._cut_origin.copy()
            plane_mesh, _ = self._build_cut_plane_mesh()

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

            # Test: vértices a ambos lados del plano
            points = np.asarray(self.grid.points)
            point_dist = np.dot(points - origin, normal)
            cells_array = self.grid.cells
            n_cells = self.grid.n_cells
            stride = cells_array[0] + 1
            vert_ids = cells_array.reshape(n_cells, stride)[:, 1:]
            vert_dists = point_dist[vert_ids]
            straddles = (vert_dists.min(axis=1) < 0) & (vert_dists.max(axis=1) > 0)

            # Test: centro dentro del rectángulo del plano
            grid_centers = self.grid.cell_centers().points
            v = grid_centers - plane_pts[0]
            in_rect = ((np.dot(v, u_i) >= 0) & (np.dot(v, u_i) <= len_i) &
                       (np.dot(v, u_j) >= 0) & (np.dot(v, u_j) <= len_j))

            cut_mask = straddles & in_rect
            signed_dist = np.dot(grid_centers - origin, normal)

            cut_any = False
            self._cut_counter += 1

            cam = self.plotter.camera
            cam_state = (cam.position, cam.focal_point,
                         cam.up, cam.clipping_range)

            for key in list(self._group_keys_ordered):
                if not self.visible.get(key, True):
                    continue

                group_mask = self.group_ids == key
                to_separate = group_mask & cut_mask

                n_sep = int(np.sum(to_separate))
                if n_sep == 0 or n_sep == int(np.sum(group_mask)):
                    continue

                cut_any = True
                new_key = -(self._cut_counter * 100
                            + len(self._group_keys_ordered))
                self.group_ids[to_separate] = float(new_key)

                self._group_keys_ordered.append(new_key)
                self.colors[new_key] = self.colors[key]
                self.visible[new_key] = True

                print(f"  Cortado {format_value(key)}: "
                      f"{n_sep} celdas separadas → grupo {new_key}")

            if not cut_any:
                self._update_status("El plano no intersecta ninguna celda")
                return

            self._refresh_all_actors()
            cam.position = cam_state[0]
            cam.focal_point = cam_state[1]
            cam.up = cam_state[2]
            cam.clipping_range = cam_state[3]

            # Línea roja recortada al rectángulo
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
                            pickable=False)
                        self._cut_line_actors.append(actor)
                except Exception:
                    pass

            self.plotter.render()
            n_groups = len(np.unique(
                self.group_ids[~np.isnan(self.group_ids)]))
            self._update_status(
                f"✓ Corte #{self._cut_counter} — {n_groups} grupos — "
                f"[S] navegar componentes — [F] cortar de nuevo")

        except Exception as e:
            self._update_status(f"Error al cortar: {e}")
            import traceback
            traceback.print_exc()

    def _clear_cut_lines(self):
        for actor in self._cut_line_actors:
            self.plotter.remove_actor(actor)
        self._cut_line_actors.clear()

    # ═══════════════════════════════════════════════════════════════════
    # Preparación para impresión (tecla L)
    # ═══════════════════════════════════════════════════════════════════

    def _prepare_for_printing(self):
        """
        Tecla L: aplica tolerancia de holgura en las caras de frontera.

        Identifica vértices de frontera (donde dos piezas se tocan)
        comparando la superficie de cada grupo con la superficie del
        grid completo. Vértices que NO están en la superficie del grid
        completo son de frontera y se desplazan hacia adentro.
        """
        if self._prepared:
            self._prepared = False
            self._refresh_all_actors()
            self._update_status("Tolerancia revertida — superficies originales")
            return

        try:
            tol = self.tolerance
            self._update_status(f"Aplicando tolerancia de {tol} mm...")
            self.plotter.render()

            # Superficie exterior del grid completo (precalculada en __init__)
            full_surface = self._full_surface.triangulate()
            full_pts = np.round(np.asarray(full_surface.points), decimals=6)
            full_set = set(map(tuple, full_pts))

            for key in list(self.surfaces.keys()):
                surface = self.surfaces[key]
                if surface is None or surface.n_cells == 0:
                    continue

                pts = np.asarray(surface.points).copy()
                n_pts = len(pts)

                # Identificar vértices de frontera:
                # vértices que NO están en la superficie exterior del grid
                is_boundary = np.zeros(n_pts, dtype=bool)
                for i in range(n_pts):
                    pt_rounded = tuple(np.round(pts[i], decimals=6))
                    if pt_rounded not in full_set:
                        is_boundary[i] = True

                if not np.any(is_boundary):
                    continue

                # Calcular normales por cara
                surface.compute_normals(
                    cell_normals=True, point_normals=False,
                    auto_orient_normals=True, inplace=True)
                face_normals = np.asarray(surface.cell_data["Normals"])

                # Acumular normales de caras en cada vértice de frontera
                vert_normals = np.zeros((n_pts, 3))
                vert_count = np.zeros(n_pts)

                faces = surface.faces.reshape(-1, 4)  # [3, v0, v1, v2] por cara
                for fi in range(surface.n_cells):
                    v0, v1, v2 = faces[fi, 1], faces[fi, 2], faces[fi, 3]
                    fn = face_normals[fi]

                    for vi in [v0, v1, v2]:
                        if is_boundary[vi]:
                            vert_normals[vi] += fn
                            vert_count[vi] += 1

                # Normalizar y aplicar offset
                mask = vert_count > 0
                vert_normals[mask] /= np.linalg.norm(
                    vert_normals[mask], axis=1, keepdims=True) + 1e-12

                # Desplazar hacia adentro (opuesto a la normal)
                pts[mask] -= vert_normals[mask] * tol

                surface.points = pts
                self._add_mesh_to_plotter(key)

            self._prepared = True
            self.plotter.render()

            n_groups = len(self.surfaces)
            self._update_status(
                f"✓ Tolerancia de {tol} mm aplicada a {n_groups} grupos — "
                f"[L] revertir — [E] exportar")

        except Exception as e:
            self._update_status(f"Error: {e}")
            import traceback
            traceback.print_exc()

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
            color="black", name="status_text")

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

            n_cells = self.surfaces[key].n_cells if key in self.surfaces else 0
            self.plotter.add_text(
                f"{idx+1}: {self.prop}={format_value(key)}  ({n_cells})",
                position=(40, y_pos + 2), font_size=8, color="black")

        self.plotter.add_text(
            "[S] Seleccionar  [N/P] Navegar  [D+num] Destino  [G] Confirmar  "
            "[T] Plano de corte  [F] Cortar  [L] Preparar impresión  "
            "[Esc] Cancelar  [C] Reset líneas  [E] Exportar",
            position=(10, 30), font_size=7, color="grey")

        self.plotter.add_key_event("s", self._enter_select_mode)
        self.plotter.add_key_event("n", self._next_component)
        self.plotter.add_key_event("p", self._prev_component)
        self.plotter.add_key_event("l", self._prepare_for_printing)
        self.plotter.add_key_event("d", self._enter_destination_mode)
        self.plotter.add_key_event("g", self._confirm_reassign)
        self.plotter.add_key_event("t", self._toggle_cut_plane)
        self.plotter.add_key_event("f", self._execute_cut)
        self.plotter.add_key_event("z", lambda: self._set_cut_rot_axis(2))
        self.plotter.add_key_event("x", lambda: self._set_cut_rot_axis(0))
        self.plotter.add_key_event("y", lambda: self._set_cut_rot_axis(1))
        self.plotter.add_key_event("e", self._export_visible)
        self.plotter.add_key_event("Escape", self._cancel)
        self.plotter.add_key_event("c", self._clear_cut_lines)

        for i in range(min(9, len(self._group_keys_ordered))):
            def make_key_cb(idx):
                def cb(): self._on_number_key(idx)
                return cb
            self.plotter.add_key_event(str(i + 1), make_key_cb(i))

        self._update_status(
            f"{len(self._group_keys_ordered)} grupos — "
            f"[S] seleccionar  [T] plano de corte")

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