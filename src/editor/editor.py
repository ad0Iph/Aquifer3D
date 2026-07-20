import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

from .utils import format_value
from .surfaces import Surfaces
from .rendering import Rendering
from .selection import Selection
from .cut_plane import CutPlane
from .printing import Printing
from .export_3mf import Export


class AquiferEditor(
    Surfaces,
    Rendering,
    Selection,
    CutPlane,
    Printing,
    Export,
):  
    # Resolución de la ventana de PyVista
    WINDOW_W = 1400
    WINDOW_H = 900

    def __init__(self, aquifer, prop="k", z_exag=1.0, decimate=0.0,
                 tolerance=0.15, cmap_name=None):
        self.prop = prop
        self.z_exag = z_exag
        self.decimate = decimate
        self.tolerance = tolerance
        self.cmap_name = cmap_name

        # Grid de PyVista desde el modelo MODFLOW
        self.grid = aquifer.build_grid(prop=prop, z_exag=z_exag)

        # Array de IDs de grupo por celda
        self.group_ids = self.grid.cell_data[prop].copy()

        # Superficies por grupo
        self.surfaces = {}
        self.rebuild_all_surfaces()

        # Actores, visibilidad y colores
        self.actors = {}
        self.visible = {}
        self.colors = {}

        group_keys = sorted(self.surfaces.keys())
        n = len(group_keys)
        if self.cmap_name:
            cmap = plt.get_cmap(self.cmap_name)
        else:
            cmap = plt.get_cmap("tab10") if n <= 10 else plt.get_cmap("tab20")

        for idx, key in enumerate(group_keys):
            self.colors[key] = cmap(idx / max(n - 1, 1))[:3]
            self.visible[key] = True

        # Estado de selección
        self.mode = self.MODE_VIEW
        self.source_group = None
        self.components = []
        self.component_idx = 0
        self.target_group = None
        self.highlight_actor = None

        # Plano de corte libre
        self.cut_active = False
        self.cut_origin = None
        self.cut_rotation = np.eye(3)
        self.cut_scale = 1.0
        
        self.cut_actor = None
        self.cut_border_actor = None
        self.cut_line_actors = []
        self.cut_counter = 0
        self.move_step = 2.0
        self.rot_step = 2.0
        self.original_style = None
        self.keys_down = set()
        self.key_observers = []

        # Preparación para impresión
        self.prepared = False
        self.full_surface = self.grid.extract_surface()

        # Plotter
        self.plotter = None
        self.group_keys_ordered = group_keys
        self.status_actor = None


    def update_status(self, text):
        """Actualiza el texto de estado en la barra inferior."""
        if self.status_actor is not None:
            self.plotter.remove_actor(self.status_actor)
        self.status_actor = self.plotter.add_text(
            text, position=(10, 8), font_size=9,
            color="black", name="status_text")

    def setup_ui(self):
        """Configura la UI con checkboxes, textos y atajos de teclado."""
        checkbox_size = 22
        row_height = checkbox_size + 10
        y_start = self.WINDOW_H - 50

        for idx, key in enumerate(self.group_keys_ordered):
            color = self.colors[key]
            y_pos = y_start - idx * row_height

            def make_vis_cb(k):
                def cb(state): self.toggle_visibility(k, state)
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

        self.plotter.add_key_event("s", self.enter_select_mode)
        self.plotter.add_key_event("n", self.next_component)
        self.plotter.add_key_event("p", self.prev_component)
        self.plotter.add_key_event("l", self.prepare_for_printing)
        self.plotter.add_key_event("d", self.enter_destination_mode)
        self.plotter.add_key_event("g", self.confirm_reassign)
        self.plotter.add_key_event("t", self.toggle_cut_plane)
        self.plotter.add_key_event("f", self.execute_cut)
        self.plotter.add_key_event("e", self.export_visible)
        self.plotter.add_key_event("Escape", self.cancel)
        self.plotter.add_key_event("c", self.clear_cut_lines)

        for i in range(min(9, len(self.group_keys_ordered))):
            def make_key_cb(idx):
                def cb(): self.on_number_key(idx)
                return cb
            self.plotter.add_key_event(str(i + 1), make_key_cb(i))

        self.update_status(
            f"{len(self.group_keys_ordered)} grupos — "
            f"[S] seleccionar  [T] plano de corte")


    def show(self):
        """ Muestra la ventana de PyVista con la escena 3D y la interfaz de usuario """
        self.plotter = pv.Plotter(window_size=[self.WINDOW_W, self.WINDOW_H])
        self.plotter.background_color = "white"
        for key in self.group_keys_ordered:
            self.add_mesh_to_plotter(key)
        self.setup_ui()
        self.plotter.add_axes(xlabel="X (m)", ylabel="Y (m)", zlabel="Z (m)")

        # Bloquear teclas que VTK usa para cerrar/zoom
        iren = self.plotter.iren
        if hasattr(iren, 'interactor') and iren.interactor is not None:
            vtk_iren = iren.interactor
        elif hasattr(iren, '_iren'):
            vtk_iren = iren.iren
        else:
            vtk_iren = iren

        def block_exit_keys(obj, event):
            """Bloqueo de teclas E, Q y F que VTK usa para salir del programa"""
            key = vtk_iren.GetKeySym()
            if key and key.lower() in ('e', 'q', 'f'):
                vtk_iren.SetKeyCode('\0')

        vtk_iren.AddObserver('CharEvent', block_exit_keys, 1.0)

        n = len(self.group_keys_ordered)
        self.plotter.show(title=f"Aquifer Editor — {self.prop}: {n} grupos")