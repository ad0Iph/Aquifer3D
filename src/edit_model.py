import argparse
from aquifer_grid_m6 import AquiferGridM6
from editor import AquiferEditor

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="editModelM6")
    parser.add_argument("model_workspace")
    parser.add_argument("--prop",            default="k")
    parser.add_argument("--z-exag",          type=float, default=1.0)
    parser.add_argument("--simplification",  type=float, default=0.0,
                        help="Fracción de caras a eliminar (0.0-0.99)")
    parser.add_argument("--tolerance",       type=float, default=0.15,
                        help="Holgura en mm por lado para impresión 3D")
    parser.add_argument("--cmap",            type=str,   default=None,
                        help="Colormap de matplotlib (ej: viridis, Set1, "
                             "Paired, Dark2, Pastel1, tab20, terrain)")

    args = parser.parse_args()

    aquifer = AquiferGridM6(args.model_workspace)
    aquifer.load()

    editor = AquiferEditor(
        aquifer, prop=args.prop, z_exag=args.z_exag,
        decimate=args.simplification,
        tolerance=args.tolerance,
        cmap_name=args.cmap,
    )
    editor.show()