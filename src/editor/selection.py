import numpy as np
from .utils import format_value

class Selection:
    MODE_VIEW = "view"
    MODE_SELECT = "select"
    MODE_BROWSE = "browse"
    MODE_DESTINATION = "dest"

    def enter_select_mode(self):
        """Enter in selection mode to choose a source group for reassignment"""
        if self.mode != self.MODE_VIEW:
            self.cancel()
            return
        self.mode = self.MODE_SELECT
        if len(self.group_keys_ordered) <= 9:
            group_list = " | ".join(
                f"[{i+1}] {format_value(k)}"
                for i, k in enumerate(self.group_keys_ordered))
            self.update_status(f"SELECTION — Choose source group: {group_list}")
        else:
            self.update_status(f"SELECTION — Enter source group: [1-{len(self.group_keys_ordered)}] + [Enter] Confirm")

    def on_number_key(self, digit):
        """Handle the selection of a group or destination"""
        if self.mode not in (self.MODE_SELECT, self.MODE_DESTINATION):
            return
        if len(self.group_keys_ordered) <= 9:
            self.apply_group_number(digit)
            return
        self.num_buffer += str(digit)
        self.update_status(f"Group: {self.num_buffer} - [Enter] Confirm [Esc] Cancel")

    def confirm_number(self):
        """Confirm the group number entered in selection or destination mode"""
        buff = self.num_buffer
        if not buff or self.mode not in (self.MODE_SELECT, self.MODE_DESTINATION):
            return
        self.num_buffer = ""
        self.apply_group_number(int(buff))
    
    def apply_group_number(self, number):
        """Apply the group number entered in selection or destination mode"""
        index = int(number) - 1
        if index < 0 or index >= len(self.group_keys_ordered):
            self.update_status(f"Group {number} does not exist")
            return
        if self.mode == self.MODE_SELECT:
            self.select_source_group(index)
        elif self.mode == self.MODE_DESTINATION:
            self.select_destination(index)

    def select_source_group(self, index):
        """Select the source group and calculate its connected components"""
        if index >= len(self.group_keys_ordered):
            return
        self.source_group = self.group_keys_ordered[index]
        self.update_status("Calculating components...")
        self.plotter.render()
        try:
            self.components = self.compute_components(self.source_group)
        except Exception as e:
            self.update_status(f"Error: {e}")
            self.mode = self.MODE_VIEW
            return
        if not self.components:
            self.update_status("Group has no visible cells or is empty")
            self.mode = self.MODE_VIEW
            return
        self.component_idx = 0
        self.mode = self.MODE_BROWSE
        self.highlight_component()
        self.show_browse_status()

    def next_component(self):
        """Move to the next connected component in browse mode"""
        if self.mode != self.MODE_BROWSE or not self.components:
            return
        self.component_idx = (self.component_idx + 1) % len(self.components)
        self.highlight_component()
        self.show_browse_status()

    def prev_component(self):
        """Move to the previous connected component in browse mode"""
        if self.mode != self.MODE_BROWSE or not self.components:
            return
        self.component_idx = (self.component_idx - 1) % len(self.components)
        self.highlight_component()
        self.show_browse_status()

    def show_browse_status(self):
        """Show the current status of the selected component in browse mode"""
        comp = self.components[self.component_idx]
        total = len(self.components)
        self.update_status(
            f"{self.prop}={format_value(self.source_group)} — "
            f"Component {self.component_idx+1}/{total} ({comp['n_cells']} cells) — "
            f"[N/P] Navigate [D] Destination [Esc] Cancel")

    def enter_destination_mode(self):
        """Enter the destination group selection mode"""
        if self.mode != self.MODE_BROWSE:
            return
        self.mode = self.MODE_DESTINATION
        if len(self.group_keys_ordered) <= 9:
            group_list = " | ".join(
                f"[{i+1}] {format_value(k)}"
                for i, k in enumerate(self.group_keys_ordered[:9])
                if k != self.source_group)
            self.update_status(f"DESTINATION — Choose group: {group_list}")
        else:
            self.update_status(f"DESTINATION — Enter destination group: [1-{len(self.group_keys_ordered)}] + [Enter] Confirm")

    def select_destination(self, index):
        """Select the destination group for reassigning the current component"""
        if index >= len(self.group_keys_ordered):
            return
        target = self.group_keys_ordered[index]
        if target == self.source_group:
            self.update_status("Cannot reassign to the same group")
            return
        self.target_group = target
        comp = self.components[self.component_idx]
        self.update_status(
            f"{format_value(self.source_group)} → {format_value(self.target_group)} "
            f"({comp['n_cells']} cells) — [G] Confirm [Esc] Cancel")
        self.mode = self.MODE_BROWSE

    def confirm_reassign(self):
        """Reassign the selected component to the destination group"""
        if self.mode != self.MODE_BROWSE or self.target_group is None:
            self.update_status("Choose destination with [D] + number")
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
            if source in self.group_keys_ordered:
                self.group_keys_ordered.remove(source)

        self.remove_highlight()
        self.restore_opacities()
        self.source_group = self.target_group = None
        self.components = []
        self.mode = self.MODE_VIEW
        self.refresh_all_actors()
        self.update_status(
            f"{n} cells: {format_value(source)} → {format_value(target)}")

    def cancel(self):
        """Cancel the current selection or reassignment, return to view mode"""
        self.remove_highlight()
        self.restore_opacities()
        self.mode = self.MODE_VIEW
        self.source_group = self.target_group = None
        self.components = []
        self.num_buffer = ""
        self.update_status(
            f"{len(self.group_keys_ordered)} groups — [S] Select")