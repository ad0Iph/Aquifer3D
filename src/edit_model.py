"""
edit_model.py — Lanza el editor interactivo de mallas de acuíferos.

Uso:
    python edit_model.py ../modelo
    python edit_model.py ../modelo --z-exag 10
    python edit_model.py ../modelo --simplification 0.5
"""

import argparse
from aquifer_grid_m6 import AquiferGridM6
from aquifer_editor import AquiferEditor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="editModelM6",
        description="Editor interactivo de mallas de acuíferos MODFLOW 6.",
    )
    parser.add_argument("model_workspace")
    parser.add_argument("--prop",            default="k")
    parser.add_argument("--z-exag",          type=float, default=1.0)
    parser.add_argument("--simplification",  type=float, default=0.0,
                        help="Fracción de caras a eliminar con CGAL edge collapse "
                             "(0.0 = sin simplificar, 0.9 = eliminar 90%%)")

    args = parser.parse_args()

    aquifer = AquiferGridM6(args.model_workspace)
    aquifer.load()

    editor = AquiferEditor(
        aquifer,
        prop=args.prop,
        z_exag=args.z_exag,
        decimate=args.simplification,
    )
    editor.show()