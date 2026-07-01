"""Renderizado: actores, visibilidad, highlight."""

import matplotlib.pyplot as plt
from .utils import format_value


class Rendering:
    DIM_OPACITY = 0.3

    def add_mesh_to_plotter(self, key):
        if key in self.actors:
            self.plotter.remove_actor(self.actors[key])
        surface = self.surfaces.get(key)
        if surface is None or surface.n_cells == 0:
            self.actors.pop(key, None)
            return
        actor = self.plotter.add_mesh(
            surface, color=self.colors.get(key, (0.5, 0.5, 0.5)),
            show_edges=False, opacity=1.0,
            name=f"group_{format_value(key)}",
            reset_camera=False)
        actor.SetVisibility(self.visible.get(key, True))
        self.actors[key] = actor

    def refresh_all_actors(self):
        self.rebuild_all_surfaces()
        for key in list(self.actors.keys()):
            if key not in self.surfaces:
                self.plotter.remove_actor(self.actors.pop(key))
        for key in self.surfaces:
            if key not in self.colors:
                n = max(len(self._group_keys_ordered), 1)
                cmap = plt.get_cmap("tab10") if n <= 10 else plt.get_cmap("tab20")
                self.colors[key] = cmap(len(self.colors) / max(n - 1, 1))[:3]
                self.visible[key] = True
            self.add_mesh_to_plotter(key)

    def remove_highlight(self):
        if self._highlight_actor is not None:
            self.plotter.remove_actor(self._highlight_actor)
            self._highlight_actor = None

    def highlight_component(self):
        self.remove_highlight()
        if not self.components or self.component_idx >= len(self.components):
            return
        comp = self.components[self.component_idx]
        self._highlight_actor = self.plotter.add_mesh(
            comp["surface"], color="white", style="wireframe",
            line_width=1.5, opacity=1.0, name="highlight",
            pickable=False, reset_camera=False)
        for gkey, actor in self.actors.items():
            p = actor.GetProperty()
            p.SetOpacity(0.7 if gkey == self.source_group else self.DIM_OPACITY)
        self.plotter.render()

    def restore_opacities(self):
        for key, actor in self.actors.items():
            actor.GetProperty().SetOpacity(1.0)
        self.plotter.render()

    def toggle_visibility(self, key, state):
        self.visible[key] = state
        if key in self.actors:
            self.actors[key].SetVisibility(state)