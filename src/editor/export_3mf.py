import numpy as np
from pathlib import Path
from .utils import format_value


def write_3mf_colored(verts, faces, rgba, path, name="piece"):
    import lib3mf

    wrapper = lib3mf.get_wrapper()
    model = wrapper.CreateModel()

    mesh = model.AddMeshObject()
    mesh.SetName(name)

    # Vértices y triángulos en el formato de lib3mf
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

    # Grupo de color con un solo color, asignado a todo el objeto
    color_group = model.AddColorGroup()
    color = wrapper.RGBAToColor(int(rgba[0]), int(rgba[1]),
                                int(rgba[2]), int(rgba[3]))
    color_id = color_group.AddColor(color)
    mesh.SetObjectLevelProperty(color_group.GetResourceID(), color_id)

    model.AddBuildItem(mesh, wrapper.GetIdentityTransform())

    writer = model.QueryWriter("3mf")
    writer.WriteToFile(str(path))

class Export:
    def export_visible(self):
        """Exporta cada malla visible como un 3MF con el color que se muestra en pantalla"""
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

            name = f"{self.prop}_{format_value(key)}"
            out_path = out_dir / f"{name}.3mf"
            write_3mf_colored(verts, faces, rgba, out_path, name=name)
            print(f"  -> {out_path}  ({surface.n_cells} caras)")
            exported += 1

        self.update_status(f"Exportados {exported} archivos 3MF en {out_dir}\ ")