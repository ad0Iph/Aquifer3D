import flopy
import numpy as np
from pathlib import Path

class AquiferGrid:
    def __init__(self, model_ws, model_name):
        self.model_ws = Path(model_ws) # directorio donde estan los ficheros del MODFLOW
        self.model_name = model_name # nombre (.nam) del modelo MODFLOW

        # Dimensiones para la grilla tridimensional (nlay x nrow xncol = numero total de celdas)
        self.nlay = None # Número de capas del acuífero
        self.nrow = None # Número de filas del la grilla MODFLOW
        self.ncol = None # Número de columnas de la grilla MODFLOW

        # Límites geométricos
        self.x_edges = None # array de coordenadas x's de todos los límites verticales de la grilla
        self.y_edges = None # array de coordenadas y's de todos los límites horizontales de la grilla
        self.z_edges = None # array de coordenadas z's de todos los "pisos" y "techos" de cada celda 

        # Propiedades hidrogeológicas
        # hk: horizontal hydraulic conductivity -> capacidad de un acuifero de transferir agua horizontalmente
        # vka: vertical hydraulic conductivity -> capacidad de un acuifero de transferir agua entre capas 
        self.properties = {}

    def load(self):
        # Cargamos el modelo MODFLOW son con Discretization (DIS) Y Layer Property Flow (LPF)
        ml = flopy.modflow.Modflow.load(
            self.model_name,
            model_ws=self.model_ws,
            load_only=["DIS", "LPF"],
            forgive=True,
            check=False,
        )

        # obtenemos la data de los ficheros
        dis = ml.get_package("DIS")
        lpf = ml.get_package("LPF")

        # las asignamos en la clase
        self.nlay = dis.nlay
        self.nrow = dis.nrow
        self.ncol = dis.ncol

        delr = dis.delr.array
        delc = dis.delc.array

        self.x_edges = np.concatenate([[0], np.cumsum(delr)])
        self.y_edges = np.concatenate([[0], np.cumsum(delc)])

        top = dis.top.array     # (nrow, ncol)
        botm = dis.botm.array   # (nlay, nrow, ncol)

        self.z_edges = np.zeros((self.nlay + 1, self.nrow, self.ncol))
        self.z_edges[0] = top
        self.z_edges[1:] = botm

        self.properties["hk"] = lpf.hk.array.copy()
        self.properties["vka"] = lpf.vka.array.copy()

    # exportamos un archivo .npz -> formato de Numpy que permite almacenar de forma comprimida data de matrices
    def export(self, filename="aquifer_grid.npz"):
        np.savez(
            filename,
            x_edges=self.x_edges,
            y_edges=self.y_edges,
            z_edges=self.z_edges,
            **self.properties
        )


if __name__ == "__main__":
    aquifer = AquiferGrid(model_ws="./2_Modelo", model_name="Melipilla.nam")
    aquifer.load()
    aquifer.export()