import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from pathlib import Path
from mesh_repair import repair_surface

def format_value(v):
    if v == 0:
        return "0"
    elif abs(v) < 1e-3 or abs(v) >= 1e4:
        return f"{v:.2e}"
    else:
        return f"{v:.4g}"

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

        # ── 3. Superficies por grupo ────────────────────────────────
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

        # ── 5. Estado del plano de corte ────────────────────────────
        self.clip_normal = None
        self.clip_origin = None
        self._plane_widget_id = None

        # ── 6. Estado de selección ──────────────────────────────────
        self.mode = self.MODE_VIEW
        self.source_group = None
        self.components = []
        self.component_idx = 0
        self.target_group = None
        self._highlight_actor = None

        # ── 7. Plotter ──────────────────────────────────────────────
        self.plotter = None
        self._group_keys_ordered = group_keys
        self._status_actor = None

    # ═══════════════════════════════════════════════════════════════════
    # Construcción de superficies
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

            if self.decimate > 0:
                surface = repair_surface(surface, decimate=self.decimate)

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
            show_edges=False,
            opacity=1.0,
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
                n = len(self._group_keys_ordered)
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
            comp["surface"],
            color="white",
            style="wireframe",
            line_width=1.5,
            opacity=1.0,
            name="highlight",
            pickable=False,
        )

        for gkey, actor in self.actors.items():
            p = actor.GetProperty()
            if gkey == self.source_group:
                p.SetOpacity(0.7)
            else:
                p.SetOpacity(self.DIM_OPACITY)

        self.plotter.render()

    def _restore_opacities(self):
        for key, actor in self.actors.items():
            actor.GetProperty().SetOpacity(1.0)
        self.plotter.render()

    # ═══════════════════════════════════════════════════════════════════
    # Funcionalidad 1: Visibilidad
    # ═══════════════════════════════════════════════════════════════════

    def _toggle_visibility(self, key, state):
        self.visible[key] = state
        if key in self.actors:
            self.actors[key].SetVisibility(state)

    # ═══════════════════════════════════════════════════════════════════
    # Funcionalidad 2: Plano de corte
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
    # Funcionalidad 3: Selección y reasignación
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
        self._update_status(
            f"Calculando componentes de {self.prop}="
            f"{format_value(self.source_group)}..."
        )
        self.plotter.render()

        try:
            self.components = self._compute_components(self.source_group)
        except Exception as e:
            self._update_status(f"Error: {e}")
            import traceback
            traceback.print_exc()
            self.mode = self.MODE_VIEW
            return

        if len(self.components) == 0:
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
        if not self.components:
            return
        comp = self.components[self.component_idx]
        total = len(self.components)
        idx = self.component_idx + 1
        self._update_status(
            f"{self.prop}={format_value(self.source_group)} — "
            f"Componente {idx}/{total} ({comp['n_cells']} celdas) — "
            f"[N]ext [P]rev [D]estino [Esc]cancelar"
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
            self._update_status("No puedes elegir el mismo grupo como destino")
            return

        self.target_group = target
        comp = self.components[self.component_idx]
        self._update_status(
            f"Componente {self.component_idx + 1} "
            f"({comp['n_cells']} celdas): "
            f"{self.prop}={format_value(self.source_group)} → "
            f"{self.prop}={format_value(self.target_group)} — "
            f"[G] confirmar [Esc] cancelar"
        )
        self.mode = self.MODE_BROWSE

    def _confirm_reassign(self):
        if self.mode != self.MODE_BROWSE:
            return
        if self.target_group is None:
            self._update_status("Primero elige destino con [D] + número")
            return
        if not self.components:
            return

        comp = self.components[self.component_idx]
        source = self.source_group
        target = self.target_group
        grid_indices = comp["grid_indices"]
        n_reassigned = len(grid_indices)

        # Reasignar SOLO las celdas del componente seleccionado
        self.group_ids[grid_indices] = target

        # Verificar
        source_remaining = np.sum(self.group_ids == source)
        print(f"  Reasignadas {n_reassigned} celdas: "
              f"{format_value(source)} → {format_value(target)}")
        print(f"  Grupo fuente '{format_value(source)}': "
              f"{source_remaining} celdas restantes")

        if source_remaining == 0:
            self.visible.pop(source, None)
            self.clip_enabled.pop(source, None)
            self.colors.pop(source, None)
            if source in self._group_keys_ordered:
                self._group_keys_ordered.remove(source)

        # Limpiar selección
        self._remove_highlight()
        self._restore_opacities()
        self.source_group = None
        self.target_group = None
        self.components = []
        self.component_idx = 0
        self.mode = self.MODE_VIEW

        # Reconstruir
        self._refresh_all_actors()

        self._update_status(
            f"✓ {n_reassigned} celdas: {self.prop}={format_value(source)} → "
            f"{self.prop}={format_value(target)} — [S] para seguir editando"
        )

    def _cancel(self):
        self._remove_highlight()
        self._restore_opacities()
        self.mode = self.MODE_VIEW
        self.source_group = None
        self.target_group = None
        self.components = []
        self.component_idx = 0
        self._update_status(
            f"{len(self._group_keys_ordered)} grupos — [S] seleccionar"
        )

    # ═══════════════════════════════════════════════════════════════════
    # Funcionalidad 4: Exportación
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
            text,
            position=(10, 8),
            font_size=9,
            color="black",
            name="status_text",
        )

    def _setup_ui(self):
        checkbox_size = 22
        row_height = checkbox_size + 10
        y_start = self.WINDOW_H - 50

        for idx, key in enumerate(self._group_keys_ordered):
            color = self.colors[key]
            y_pos = y_start - idx * row_height

            def make_vis_cb(k):
                def cb(state):
                    self._toggle_visibility(k, state)
                return cb

            self.plotter.add_checkbox_button_widget(
                make_vis_cb(key), value=True,
                position=(10, y_pos), size=checkbox_size,
                color_on=color, color_off="grey",
            )

            def make_clip_cb(k):
                def cb(state):
                    self._toggle_clip(k, state)
                return cb

            self.plotter.add_checkbox_button_widget(
                make_clip_cb(key), value=False,
                position=(40, y_pos), size=checkbox_size - 4,
                color_on="red", color_off="darkgrey",
            )

            n_cells = self.surfaces[key].n_cells if key in self.surfaces else 0
            self.plotter.add_text(
                f"{idx+1}: {self.prop}={format_value(key)}  ({n_cells})",
                position=(68, y_pos + 2),
                font_size=8, color="black",
            )

        # Instrucciones
        self.plotter.add_text(
            "[S] Seleccionar  [N/P] Navegar  "
            "[D+num] Destino  [G] Confirmar  "
            "[Esc] Cancelar  [C] Reset cortes  [E] Exportar",
            position=(10, 30), font_size=7, color="grey",
        )

        # Atajos
        self.plotter.add_key_event("s", self._enter_select_mode)
        self.plotter.add_key_event("n", self._next_component)
        self.plotter.add_key_event("p", self._prev_component)
        self.plotter.add_key_event("d", self._enter_destination_mode)
        self.plotter.add_key_event("g", self._confirm_reassign)
        self.plotter.add_key_event("e", self._export_visible)
        self.plotter.add_key_event("Escape", self._cancel)

        def reset_clips():
            for k in list(self.clip_enabled.keys()):
                self.clip_enabled[k] = False
                if k in self.original_surfaces:
                    self.surfaces[k] = self.original_surfaces[k].copy()
                    self._add_mesh_to_plotter(k)
            self._hide_plane_widget()
            self._update_status("Cortes desactivados")
        self.plotter.add_key_event("c", reset_clips)

        for i in range(min(9, len(self._group_keys_ordered))):
            def make_key_cb(idx):
                def cb():
                    self._on_number_key(idx)
                return cb
            self.plotter.add_key_event(str(i + 1), make_key_cb(i))

        self._update_status(
            f"{len(self._group_keys_ordered)} grupos — [S] para seleccionar"
        )

    # ═══════════════════════════════════════════════════════════════════
    # Método principal
    # ═══════════════════════════════════════════════════════════════════

    def show(self):
        self.plotter = pv.Plotter(
            window_size=[self.WINDOW_W, self.WINDOW_H],
        )
        self.plotter.background_color = "white"

        for key in self._group_keys_ordered:
            self._add_mesh_to_plotter(key)

        self._setup_ui()
        self.plotter.add_axes(xlabel="X (m)", ylabel="Y (m)", zlabel="Z (m)")

        n = len(self._group_keys_ordered)
        self.plotter.show(title=f"Aquifer Editor — {self.prop}: {n} grupos")