import flopy
import numpy as np
from pathlib import Path

class AquiferGrid:
    def __init__(self, model_ws, sim_name):
        self.model_ws = Path(model_ws)
        self.sim_name = sim_name 

        self.nlay = None
        self.nrow = None
        self.ncol = None

        self.x_edges = None
        self.y_edges = None
        self.z_edges = None

        self.properties = {}

    def load(self):
        sim = flopy.mf6.MFSimulation.load(
            sim_name=self.sim_name,
            sim_ws=self.model_ws,
            verbosity_level=0
        )

        model = sim.get_model()

        dis = model.get_package("dis")

        npf = model.get_package("npf")

        self.nlay = dis.nlay.get_data()
        self.nrow = dis.nrow.get_data()
        self.ncol = dis.ncol.get_data()
        print("Total celdas:", self.nlay * self.nrow * self.ncol)

        delr = dis.delr.get_data()
        delc = dis.delc.get_data()

        self.x_edges = np.concatenate([[0], np.cumsum(delr)])
        self.y_edges = np.concatenate([[0], np.cumsum(delc)])

        top = dis.top.get_data()
        botm = dis.botm.get_data()

        self.z_edges = np.zeros((self.nlay + 1, self.nrow, self.ncol))
        self.z_edges[0] = top
        self.z_edges[1:] = botm

        self.properties["k"] = npf.k.get_data().copy()

        k33_data = getattr(npf, "k33", None)

        if k33_data is not None:
            k33_array = k33_data.get_data()
            if k33_array is not None:
                self.properties["k33"] = k33_array.copy()

    def export(self, filename="aquifer_grid_m6.npz"):
        np.savez(
            filename,
            x_edges=self.x_edges,
            y_edges=self.y_edges,
            z_edges=self.z_edges,
            **self.properties
        )

if __name__ == "__main__":
    aquifer = AquiferGrid(
        model_ws="models/Two-layer-1/lito+met_v3",
        sim_name="lito+met_v3"
    )
    aquifer.load()
    aquifer.export()