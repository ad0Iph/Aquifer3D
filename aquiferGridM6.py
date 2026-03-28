import flopy
import numpy as np
from pathlib import Path
import argparse

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

    def splitN(self, N):
        row_splits = np.linspace(0, self.nrow, N+1, dtype=int)
        col_splits = np.linspace(0, self.ncol, N+1, dtype=int)

        subdomains = {}

        for i in range(N):
            for j in range(N):
                r0, r1 = row_splits[i], row_splits[i+1]
                c0, c1 = col_splits[j], col_splits[j+1]

                key = f"block_{i}_{j}"

                subdomains[key] = {
                    "x_edges": self.x_edges[c0:c1+1],
                    "y_edges": self.y_edges[r0:r1+1],
                    "z_edges": self.z_edges[:, r0:r1, c0:c1],
                    "k": self.properties["k"][:, r0:r1, c0:c1],
                }

                if "k33" in self.properties:
                    subdomains[key]["k33"] = self.properties["k33"][:, r0:r1, c0:c1]

        return subdomains
    
    def split_by_property(self, prop="k", N=3):
        values = self.properties[prop]

        vmin = np.nanmin(values)
        vmax = np.nanmax(values)

        bins = np.linspace(vmin, vmax, N+1)

        groups = {}

        for i in range(N):
            mask = (values >= bins[i]) & (values < bins[i+1])

            k_masked = np.where(mask, values, np.nan)

            group_name = f"{prop}_{i}"

            groups[group_name] = {
                "x_edges": self.x_edges,
                "y_edges": self.y_edges,
                "z_edges": self.z_edges,
                "k": k_masked,
            }

            if "k33" in self.properties:
                k33 = self.properties["k33"]
                groups[group_name]["k33"] = np.where(mask, k33, np.nan)

        return groups
    
    def export_by_property(self, prop="k", N=3):
        groups = self.split_by_property(prop, N)

        for name, g in groups.items():
            np.savez(
                f"aquifer_{name}.npz",
                x_edges=g["x_edges"],
                y_edges=g["y_edges"],
                z_edges=g["z_edges"],
                k=g["k"],
                **({"k33": g["k33"]} if "k33" in g else {})
            )
    
    def export_splitN(self, N):
        blocks = self.splitN(N)

        for name, b in blocks.items():
            np.savez(
                f"aquifer_{name}.npz",
                x_edges=b["x_edges"],
                y_edges=b["y_edges"],
                z_edges=b["z_edges"],
                k=b["k"],
                **({"k33": b["k33"]} if "k33" in b else {})
            )

    def export(self, filename="aquifer_grid_m6.npz"):
        np.savez(
            filename,
            x_edges=self.x_edges,
            y_edges=self.y_edges,
            z_edges=self.z_edges,
            **self.properties
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='aquiferGridM6')
    parser.add_argument('model_workspace', help='directory where the model is located')
    parser.add_argument('simulation_name', help='name of the simulation')
    parser.add_argument('cuts', help='number of cuts to the model')

    args = parser.parse_args()
    aquifer = AquiferGrid(
        args.model_workspace,
        args.simulation_name,
    )

    aquifer.load()
    aquifer.export()
    #aquifer.export_splitN(int(args.cuts))
    #aquifer.export_by_property('k', 3)