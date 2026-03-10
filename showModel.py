import numpy as np
import pyvista as pv

def visualizeModflow(filename="aquifer_grid.npz", prop="hk", showGrid=False):
    # cargamos la data
    data = np.load(filename)
    
    # 
    x_edges = data["x_edges"]      # (ncol+1,)
    y_edges = data["y_edges"]      # (nrow+1,)
    z_edges = data["z_edges"]      # (nlay+1, nrow, ncol)
    values = data[prop]            # (nlay, nrow, ncol)

    #
    nlay, nrow, ncol = values.shape

    # creamos una grilla vacía con pyvista
    grid = pv.UnstructuredGrid()

    hex_type = pv.CellType.HEXAHEDRON

    points = []
    cells = []
    cell_data = []

    def add_point(x, y, z, point_cache):
        """Avoid duplicates by caching."""
        key = (x, y, z)
        if key not in point_cache:
            point_cache[key] = len(point_cache)
            points.append([x, y, z])
        return point_cache[key]

    point_cache = {}

    for k in range(nlay):
        for i in range(nrow):
            for j in range(ncol):

                # Cell edges
                x0, x1 = x_edges[j], x_edges[j+1]
                y0, y1 = y_edges[i], y_edges[i+1]
                z_top    = z_edges[k,   i, j]
                z_bottom = z_edges[k+1, i, j]

                # 8 vertex coordinates of the hexahedral cell
                verts = [
                    (x0, y0, z_bottom),
                    (x1, y0, z_bottom),
                    (x1, y1, z_bottom),
                    (x0, y1, z_bottom),
                    (x0, y0, z_top),
                    (x1, y0, z_top),
                    (x1, y1, z_top),
                    (x0, y1, z_top),
                ]

                vert_ids = [add_point(*v, point_cache) for v in verts]

                # Add cell
                cells.append([8] + vert_ids)
                cell_data.append(values[k, i, j])

    # Convert to numpy arrays
    cells = np.array(cells, dtype=np.int64).ravel()
    cell_types = np.full(len(cell_data), hex_type, dtype=np.uint8)
    points = np.array(points)

    # Build the grid
    grid = pv.UnstructuredGrid(cells, cell_types, points)
    grid.cell_data[prop] = np.array(cell_data)

    # Plot
    p = pv.Plotter()
    p.add_mesh(grid, scalars=prop, cmap="viridis", show_edges=True, opacity=0.7)

    # Optional: add a grid overlay
    p.add_axes()
    if showGrid:
        p.show_grid()
    p.show()

# Example usage:
# visualize_voxels("aquifer_grid.npz", prop="hk")
if __name__ == "__main__":
    visualizeModflow("aquifer_grid.npz", prop="hk")
    visualizeModflow("aquifer_grid.npz", prop="hk", showGrid=True)
