import numpy as np
from .utils import format_value


class Selection:
    MODE_VIEW = "view"
    MODE_SELECT = "select"
    MODE_BROWSE = "browse"
    MODE_DESTINATION = "dest"

    def enter_select_mode(self):
        """Inicia el modo de selección de grupo fuente"""
        if self.mode != self.MODE_VIEW:
            self.cancel()
            return
        self.mode = self.MODE_SELECT
        group_list = " | ".join(
            f"[{i+1}] {format_value(k)}"
            for i, k in enumerate(self._group_keys_ordered[:9]))
        self.update_status(f"SELECCIÓN — Elige grupo fuente: {group_list}")

    def on_number_key(self, index):
        """Maneja la selección de grupo o destino"""
        if self.mode == self.MODE_SELECT:
            self.select_source_group(index)
        elif self.mode == self.MODE_DESTINATION:
            self.select_destination(index)

    def select_source_group(self, index):
        """Selecciona el grupo fuente y calcula sus componentes conexos"""
        if index >= len(self._group_keys_ordered):
            return
        self.source_group = self._group_keys_ordered[index]
        self.update_status("Calculando componentes...")
        self.plotter.render()
        try:
            self.components = self.compute_components(self.source_group)
        except Exception as e:
            self.update_status(f"Error: {e}")
            self.mode = self.MODE_VIEW
            return
        if not self.components:
            self.update_status("Grupo sin celdas")
            self.mode = self.MODE_VIEW
            return
        self.component_idx = 0
        self.mode = self.MODE_BROWSE
        self.highlight_component()
        self.show_browse_status()

    def next_component(self):
        """Avanza al siguiente componente conexo en el modo browse"""
        if self.mode != self.MODE_BROWSE or not self.components:
            return
        self.component_idx = (self.component_idx + 1) % len(self.components)
        self.highlight_component()
        self.show_browse_status()

    def prev_component(self):
        """Avanza al componente conexo previo en el modo browse"""
        if self.mode != self.MODE_BROWSE or not self.components:
            return
        self.component_idx = (self.component_idx - 1) % len(self.components)
        self.highlight_component()
        self.show_browse_status()

    def show_browse_status(self):
        """Muestra el estado actual del componente seleccionado en modo browse"""
        comp = self.components[self.component_idx]
        total = len(self.components)
        self.update_status(
            f"{self.prop}={format_value(self.source_group)} — "
            f"Componente {self.component_idx+1}/{total} ({comp['n_cells']} celdas) — "
            f"[N/P] navegar [D] destino [Esc] cancelar")

    def enter_destination_mode(self):
        """Inicia el modo de selección de grupo destino"""
        if self.mode != self.MODE_BROWSE:
            return
        self.mode = self.MODE_DESTINATION
        group_list = " | ".join(
            f"[{i+1}] {format_value(k)}"
            for i, k in enumerate(self._group_keys_ordered[:9])
            if k != self.source_group)
        self.update_status(f"DESTINO — Elige grupo: {group_list}")

    def select_destination(self, index):
        """Selecciona el grupo destino para reasignar el componente actual"""
        if index >= len(self._group_keys_ordered):
            return
        target = self._group_keys_ordered[index]
        if target == self.source_group:
            self.update_status("No puedes elegir el mismo grupo")
            return
        self.target_group = target
        comp = self.components[self.component_idx]
        self.update_status(
            f"{format_value(self.source_group)} → {format_value(self.target_group)} "
            f"({comp['n_cells']} celdas) — [G] confirmar")
        self.mode = self.MODE_BROWSE

    def confirm_reassign(self):
        """Reasigna el componente seleccionado al grupo destino"""
        if self.mode != self.MODE_BROWSE or self.target_group is None:
            self.update_status("Elige destino con [D] + número")
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

        self.remove_highlight()
        self.restore_opacities()
        self.source_group = self.target_group = None
        self.components = []
        self.mode = self.MODE_VIEW
        self.refresh_all_actors()
        self.update_status(
            f"✓ {n} celdas: {format_value(source)} → {format_value(target)}")

    def cancel(self):
        """Cancela la selección o reasignación en curso, vuelve al modo view"""
        self.remove_highlight()
        self.restore_opacities()
        self.mode = self.MODE_VIEW
        self.source_group = self.target_group = None
        self.components = []
        self.update_status(
            f"{len(self._group_keys_ordered)} grupos — [S] seleccionar")