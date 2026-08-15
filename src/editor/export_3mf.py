import numpy as np
from pathlib import Path
from .utils import format_value
import lib3mf

def write_3mf_multi(pieces, path):
    """pieces: list of (name, verts, faces, rgba). All in a single 3MF."""
    wrapper = lib3mf.get_wrapper()
    model = wrapper.CreateModel()

    for name, verts, faces, rgba in pieces:
        mesh = model.AddMeshObject()
        mesh.SetName(name)

        positions = []
        for v in verts:
            pos = lib3mf.Position()
            pos.Coordinates[0] = float(v[0])
            pos.Coordinates[1] = float(v[1])
            pos.Coordinates[2] = float(v[2])
            positions.append(pos)

        triangles = []
        for f in faces:
            tri = lib3mf.Triangle()
            tri.Indices[0] = int(f[0])
            tri.Indices[1] = int(f[1])
            tri.Indices[2] = int(f[2])
            triangles.append(tri)

        mesh.SetGeometry(positions, triangles)

        color_group = model.AddColorGroup()
        color = wrapper.RGBAToColor(int(rgba[0]), int(rgba[1]),
                                    int(rgba[2]), int(rgba[3]))
        color_id = color_group.AddColor(color)
        mesh.SetObjectLevelProperty(color_group.GetResourceID(), color_id)

        model.AddBuildItem(mesh, wrapper.GetIdentityTransform())

    writer = model.QueryWriter("3mf")
    writer.WriteToFile(str(path))

class Export:
    def export_visible(self, max_size_mm=240.0):
        """Export each visible mesh as a 3MF with the color that is displayed on screen"""
        out_dir = Path("export/editor")
        out_dir.mkdir(parents=True, exist_ok=True)

        pieces = []
        for key in self.group_keys_ordered:
            if not self.visible.get(key, False):
                continue
            surface = self.surfaces.get(key)
            if surface is None or surface.n_cells == 0:
                continue
            tri = surface.triangulate().copy(deep=True)
            pieces.append((key, tri))

        if not pieces:
            self.update_status("No visible meshes to export")
            return

        all_pts = np.vstack([tri.points for _, tri in pieces])
        bb_min = all_pts.min(axis=0)
        bb_max = all_pts.max(axis=0)
        extent = bb_max - bb_min
        scale = max_size_mm / extent.max()

        to_write = []
        for key, tri in pieces:
            verts = (np.asarray(tri.points) - bb_min) * scale
            faces = tri.faces.reshape(-1, 4)[:, 1:4]

            rgb = self.colors.get(key, (0.5, 0.5, 0.5))
            rgba = [round(rgb[0]*255), round(rgb[1]*255),
                    round(rgb[2]*255), 255]

            name = f"{self.prop}_{format_value(key)}"
            to_write.append((name, verts, faces, rgba))

        out_path = out_dir / f"{self.prop}_model.3mf"
        write_3mf_multi(to_write, out_path)

        self.update_status(
            f"Exported to {out_path} with {len(to_write)} pieces - "
            f"scale 1:{1/scale:.1f}"
        )