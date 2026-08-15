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
    WINDOW_W = 1400
    WINDOW_H = 900

    def __init__(self, aquifer, prop="k", z_exag=1.0, decimate=0.0,
                 tolerance=0.5, cmap_name=None):
        self.prop = prop
        self.z_exag = z_exag
        self.decimate = decimate
        self.tolerance = tolerance
        self.cmap_name = cmap_name

        self.grid = aquifer.build_grid(prop=prop, z_exag=z_exag)

        self.group_ids = self.grid.cell_data[prop].copy()

        self.surfaces = {}
        self.rebuild_all_surfaces()

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

        self.mode = self.MODE_VIEW
        self.source_group = None
        self.components = []
        self.component_idx = 0
        self.target_group = None
        self.highlight_actor = None
        self.num_buffer = ""

        self.cut_active = False
        self.cut_origin = None
        self.cut_rotation = np.eye(3)
        self.cut_scale = 1.0
        
        self.cut_actor = None
        self.cut_border_actor = None
        self.cut_counter = 0
        self.move_step = 2.0
        self.rot_step = 2.0
        self.original_style = None
        self.keys_down = set()
        self.key_observers = []

        self.prepared = False
        self.full_surface = self.grid.extract_surface()

        self.plotter = None
        self.group_keys_ordered = group_keys
        self.status_actor = None

        self.ui_text_actors = []
        self.last_window_size = (self.WINDOW_W, self.WINDOW_H)
        self.resize_timer_id = None

        self.show_bounds = False

    def update_status(self, text):
        """Update the status text displayed in the UI."""
        if self.status_actor is not None:
            self.plotter.remove_actor(self.status_actor)
        self.status_actor = self.plotter.add_text(
            text, position=(10, 8), font_size=9,
            color="black", name="status_text")
        
    def build_panel(self):
        """Builds the UI panel with checkboxes for each group and instructions."""
        for actor in self.ui_text_actors:
            self.plotter.remove_actor(actor)
        self.ui_text_actors = []
        self.plotter.clear_button_widgets()

        w, h = self.plotter.render_window.GetSize()
        checkbox_size = 22
        row_height = checkbox_size + 10
        col_width = 230
        y_start = h - 50
        max_rows = 5

        for idx, key in enumerate(self.group_keys_ordered):
            col, row = divmod(idx, max_rows)
            x = 10 + col * col_width
            y = y_start - row * row_height

            def make_vis_cb(k):
                def cb(state): self.toggle_visibility(k, state)
                return cb

            self.plotter.add_checkbox_button_widget(
                make_vis_cb(key), value=self.visible.get(key, True),
                position=(x, y), size=checkbox_size,
                color_on=self.colors[key], color_off="grey")

            n_cells = self.surfaces[key].n_cells if key in self.surfaces else 0
            t = self.plotter.add_text(
                f"{idx+1}: {self.prop}={format_value(key)}  ({n_cells})",
                position=(x + 30, y + 2), font_size=8, color="black")
            self.ui_text_actors.append(t)

        t = self.plotter.add_text(
            "[S] Select  [N/P] Navigate  [D+num] Destination  [G] Confirm  "
            "[T] Cutting plane  [F] Cut  [L] Prepare for printing [I] Show bounds  "
            "[Esc] Cancel  [E] Export",
            position=(10, 30), font_size=7, color="grey")
        self.ui_text_actors.append(t)

    def setup_ui(self):
        self.build_panel()

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
        self.plotter.add_key_event("i", self.toggle_bounds)

        for i in "0123456789":
            def make_key_cb(digit):
                def cb(): self.on_number_key(digit)
                return cb
            self.plotter.add_key_event(i, make_key_cb(i))
        self.plotter.add_key_event(f"Return", self.confirm_number)
            

        self.update_status(
            f"{len(self.group_keys_ordered)} groups — "
            f"[S] Select  [T] Cutting plane")


    def show(self):
        """Show the interactive 3D plotter with the aquifer model and UI."""
        self.plotter = pv.Plotter(window_size=[self.WINDOW_W, self.WINDOW_H])
        self.plotter.background_color = "white"
        for key in self.group_keys_ordered:
            self.add_mesh_to_plotter(key)
        self.setup_ui()
        self.plotter.add_axes(xlabel="X (m)", ylabel="Y (m)", zlabel="Z (m)")

        iren = self.plotter.iren
        if hasattr(iren, 'interactor') and iren.interactor is not None:
            vtk_iren = iren.interactor
        elif hasattr(iren, '_iren'):
            vtk_iren = iren.iren
        else:
            vtk_iren = iren

        def block_exit_keys(obj, event):
            """Blocks certain VTK keys that would exit the application."""
            key = vtk_iren.GetKeySym()
            if key and key.lower() in ('e', 'q', '3', 'f'):
                vtk_iren.SetKeyCode('\0')

        def on_window_resize(obj, event):
            """Handles window resize events to rebuild the UI panel after resizing."""
            size = self.plotter.render_window.GetSize()
            if size == self.last_window_size:
                return
            self.last_window_size = size
            if self.resize_timer_id is not None:
                vtk_iren.DestroyTimer(self.resize_timer_id)
            self.resize_timer_id = vtk_iren.CreateOneShotTimer(250)

        def on_resize_timer(obj, event):
            """Handles the timer event after a window resize to rebuild the UI panel."""
            if self.resize_timer_id is None:
                return
            if obj.GetTimerEventId() != self.resize_timer_id:
                return               
            self.resize_timer_id = None
            self.build_panel()
            self.plotter.render()

        vtk_iren.AddObserver('CharEvent', block_exit_keys, 1.0)
        vtk_iren.AddObserver('ConfigureEvent', on_window_resize)
        vtk_iren.AddObserver('TimerEvent', on_resize_timer)

        n = len(self.group_keys_ordered)
        self.plotter.show(title=f"PyEarthFab - {self.prop}: {n} groups")