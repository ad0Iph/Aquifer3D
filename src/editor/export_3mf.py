import numpy as np
from pathlib import Path
from .utils import format_value


class Export:
    def export_visible(self):
        import trimesh

        out_dir = Path("export/editor")
        out_dir.mkdir(parents=True, exist_ok=True)

        exported = 0
        for key in self._group_keys_ordered:
            if not self.visible.get(key, False):
                continue
            surface = self.surfaces.get(key)
            if surface is None or surface.n_cells == 0:
                continue

            verts = np.asarray(surface.points)
            faces = surface.faces.reshape(-1, 4)[:, 1:4]

            rgb = self.colors.get(key, (0.5, 0.5, 0.5))
            rgba = [int(rgb[0]*255), int(rgb[1]*255),
                    int(rgb[2]*255), 255]
            face_colors = np.tile(rgba, (len(faces), 1))

            mesh = trimesh.Trimesh(
                vertices=verts, faces=faces,
                face_colors=face_colors)

            fname = f"{self.prop}_{format_value(key)}.3mf"
            out_path = out_dir / fname
            mesh.export(str(out_path), file_type="3mf")
            print(f"  → {out_path}  ({surface.n_cells} caras)")
            exported += 1

        self.update_status(f"Exportados {exported} archivos 3MF en {out_dir}/")