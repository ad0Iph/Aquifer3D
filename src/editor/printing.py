import numpy as np

class Printing:
    def prepare_for_printing(self):
        """Apply a tolerance to the surfaces to prepare them for 3D printing. This method modifies the surfaces in place and updates the plotter."""
        self.prepared = not self.prepared
        self.update_status("Aplicando tolerancia..." if self.prepared
                        else "Revirtiendo tolerancia...")
        self.plotter.render()
        self.refresh_all_actors()
        self.update_status(
            f"Tolerance of {self.tolerance} (model units) applied — [L] Revert — [E]Export"
            if self.prepared else "Tolerance reverted — original surfaces")