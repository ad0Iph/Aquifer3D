import argparse
from aquifer_grid_m6 import AquiferGridM6
from aquifer_editor import AquiferEditor

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="editModelM6")
    parser.add_argument("model_workspace")
    parser.add_argument("--prop",            default="k")
    parser.add_argument("--z-exag",          type=float, default=1.0)
    parser.add_argument("--simplification",  type=float, default=0.0,
                        help="Fracción de caras a eliminar (0.0-0.99)")

    args = parser.parse_args()

    aquifer = AquiferGridM6(args.model_workspace)
    aquifer.load()

    editor = AquiferEditor(
        aquifer, prop=args.prop, z_exag=args.z_exag,
        decimate=args.simplification,
    )
    editor.show()