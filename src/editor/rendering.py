"""Renderizado: actores, visibilidad, highlight."""

import matplotlib.pyplot as plt
from .utils import format_value


class Rendering:
    DIM_OPACITY = 0.3

    def add_mesh_to_plotter(self, key):
        """Agrega la malla del grupo `key` al plotter, con color y visibilidad actuales."""
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
        """Redibuja todos los actores en la escena según las superficies, colores y visibilidad actuales."""
        self.rebuild_all_surfaces()
        for key in list(self.actors.keys()):
            if key not in self.surfaces:
                self.plotter.remove_actor(self.actors.pop(key))
        for key in self.surfaces:
            if key not in self.colors:
                n = max(len(self.group_keys_ordered), 1)
                cmap = plt.get_cmap("tab10") if n <= 10 else plt.get_cmap("tab20")
                self.colors[key] = cmap(len(self.colors) / max(n - 1, 1))[:3]
                self.visible[key] = True
            self.add_mesh_to_plotter(key)

    def remove_highlight(self):
        """Elimina el actor destacado si existe."""
        if self.highlight_actor is not None:
            self.plotter.remove_actor(self.highlight_actor)
            self.highlight_actor = None

    def highlight_component(self):
        """Destaca el componente actualmente seleccionado, si existe."""
        self.remove_highlight()
        if not self.components or self.component_idx >= len(self.components):
            return
        comp = self.components[self.component_idx]
        self.highlight_actor = self.plotter.add_mesh(
            comp["surface"], color="white", style="wireframe",
            line_width=1.5, opacity=1.0, name="highlight",
            pickable=False, reset_camera=False)
        for gkey, actor in self.actors.items():
            p = actor.GetProperty()
            p.SetOpacity(0.7 if gkey == self.source_group else self.DIM_OPACITY)
        self.plotter.render()

    def restore_opacities(self):
        """Restaura la opacidad de todos los actores a 1.0."""
        for key, actor in self.actors.items():
            actor.GetProperty().SetOpacity(1.0)
        self.plotter.render()

    def toggle_visibility(self, key, state):
        """Activa o desactiva la visibilidad del grupo `key`."""
        self.visible[key] = state
        if key in self.actors:
            self.actors[key].SetVisibility(state)

    def toggle_bounds(self):
        """Tecla I: muestra/oculta los borders del modelo con unidades reales."""
        if self.show_bounds:
            self.plotter.remove_bounds_axes()
            self.plotter.remove_bounding_box()
            self.show_bounds = False
            self.update_status("Bordes ocultos")
        else:
            xmin, xmax, ymin, ymax, zmin, zmax = self.grid.bounds
            z_range = (zmin / self.z_exag, zmax / self.z_exag)
            self.plotter.show_bounds(
                grid='front', location='outer', all_edges=True,
                font_size=14, n_xlabels=4, n_ylabels=4, n_zlabels=2,
                xtitle='X (m)', ytitle='Y (m)',
                ztitle=f'Z (m, vista x{self.z_exag:g})',
                bounds=self.grid.bounds,
                axes_ranges=(xmin, xmax, ymin, ymax,
                            zmin / self.z_exag, zmax / self.z_exag))

            self.orig_update_bounds_axes = self.plotter.renderer.update_bounds_axes
            def pinned_update_bounds_axes():
                cax = self.plotter.renderer.cube_axes_actor
                if cax is not None:
                    cax.update_bounds(self.grid.bounds)  
                    cax.z_axis_range = z_range           
            self.plotter.renderer.update_bounds_axes = pinned_update_bounds_axes

            self.show_bounds = True
            self.update_status("Cotas en metros (Z corregido a elevación real)")
        self.plotter.render()