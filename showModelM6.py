import numpy as np
import pyvista as pv

def visualizeModflowStructured(filename="aquifer_grid.npz", prop="k"):
    data = np.load(filename)

    x_edges = data["x_edges"]
    y_edges = data["y_edges"]
    z_edges = data["z_edges"]
    values = data[prop]

    nlay, nrow, ncol = values.shape

    # Crear malla con orden correcto (X, Y, Z)
    X = np.zeros((ncol+1, nrow+1, nlay+1))
    Y = np.zeros_like(X)
    Z = np.zeros_like(X)

    for j in range(ncol+1):
        for i in range(nrow+1):
            for k in range(nlay+1):

                X[j, i, k] = x_edges[j]
                Y[j, i, k] = y_edges[i]

                if k == 0:
                    Z[j, i, k] = z_edges[0, min(i, nrow-1), min(j, ncol-1)]
                else:
                    Z[j, i, k] = z_edges[k-1, min(i, nrow-1), min(j, ncol-1)]

    grid = pv.StructuredGrid(X, Y, Z)

    grid.points[:, 2] *= 2   # exageración vertical (factor 2)


    # ⚠️ flatten en orden Fortran (MUY importante)
    grid.cell_data[prop] = values.flatten(order="F")

    # Plot
    p = pv.Plotter()
    p.add_mesh(grid, scalars=prop, cmap="turbo", opacity=0.7)
    p.add_axes()
    p.show()

if __name__ == "__main__":
    visualizeModflowStructured("aquifer_grid_m6.npz", prop="k")
