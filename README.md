# PyEarthFab

A Python framework for transforming Earth science models into 3D printable physical representations. Supported models: structured mesh from MODFLOW 6. 

## Project structure

```
PyEarthFab/
├── src/
│   ├── aquifer_grid_m6.py    # Loads MODFLOW model → PyVista grid
│   ├── edit_model.py         # Interactive editor (launcher)
│   ├── mesh_repair.py        # Repair/simplification wrapper (CGAL or PyVista)
│   └── editor/               # Interactive editor (package)
│       ├── __init__.py
│       ├── cut_plane.py      # Free cutting plane
│       ├── editor.py         # Main class AquiferEditor
│       ├── export_3mf.py     # 3MF export
│       ├── printing.py       # 3D-printing tolerance
│       ├── rendering.py      # Actors and visibility
│       ├── selection.py      # Connected-component selection
│       ├── surfaces.py       # Surface construction
│       └── utils.py          # Shared utilities
│
├── cgal_repair/              # C++ module (must be compiled once)
│   ├── src/cgal_repair.cpp   # pybind11 + CGAL binding
│   ├── CMakeLists.txt
│   └── setup.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

## Prerequisites (Windows)

Before starting, install:

1. **Python 3.11 or 3.12** — [python.org/downloads](https://www.python.org/downloads/)
   - **Python 3.13 or newer is NOT supported.** The pinned dependencies require
     `vtk < 9.4`, and VTK only ships wheels for Python 3.13+ starting from the
     9.6 series. With a newer Python, `pip install` fails with
     `No matching distribution found for vtk<9.4.0`.
   - During installation check  **"Add Python to PATH"**
2. **Git** — [git-scm.com/downloads](https://git-scm.com/downloads)
3. **Visual Studio Build Tools 2022** (only needed to compile the CGAL module) —
   [visualstudio.microsoft.com/downloads](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022)
   - In the installer select the **"Desktop development with C++"** workload
4. **CMake** — [cmake.org/download](https://cmake.org/download/)
   - During installation check **"Add CMake to the system PATH"**

Verify everything is installed (open a new terminal):

```powershell
py -3.12 --version   # Python 3.12.x
git --version        # git version 2.x
cmake --version      # cmake version 3.x
```

## Installation

### 1. Clone the repository and create the environment

```powershell
git clone <repo-url>
cd PyEarthFab
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

> **Note:** every time you open a new terminal to work on the project you must
> activate the environment with `.venv\Scripts\activate`.

### 2. Install vcpkg and the C++ dependencies

vcpkg is Microsoft's C++ package manager. It is installed **once, outside the
project**:

```powershell
cd C:\
mkdir dev
cd dev
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat
```

Install CGAL and pybind11 (this may take 30–60 minutes the first time):

```powershell
.\vcpkg install cgal:x64-windows pybind11:x64-windows
```

### 3. Compile the cgal_repair module

> **The compiled module is NOT distributed with the repository.** After
> cloning, `cgal_repair` does not exist until you build it. This step is
> mandatory for the CGAL repair/simplification pipeline (without it the project
> still runs, but falls back to a less robust PyVista repair).

Go back to the project directory:

```powershell
cd <path-to-project>\PyEarthFab\cgal_repair
mkdir build
cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=C:/dev/vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build . --config Release
```

Copy the compiled module and the DLLs into `src/`:

```powershell
copy Release\cgal_repair.*.pyd ..\..\src\
copy C:\dev\vcpkg\installed\x64-windows\bin\gmp.dll ..\..\src\
copy C:\dev\vcpkg\installed\x64-windows\bin\mpfr-6.dll ..\..\src\
```

> **Important:** the `.pyd` file is tagged with the Python version it was built
> for (e.g. `cgal_repair.cp312-win_amd64.pyd` only loads on Python **3.12**).
> The compilation must be done with the project's virtual environment
> **activated**, so the tag matches the interpreter of the environment. A `.pyd`
> built for a different Python version is silently ignored by the import system
> and produces `ModuleNotFoundError`.

### 4. Verify the installation

The verification must be run **from the `src/` directory** (the module lives
there and Python imports from the current directory):

```powershell
cd <path-to-project>\PyEarthFab\src
python -c "import cgal_repair; print('CGAL OK')"
python -c "import flopy, pyvista, lib3mf; print('Python deps OK')"
```

If both commands print OK, the installation is complete.

## Usage

All commands are run from the `src/` folder with the environment activated:

```powershell
cd <path-to-project>\PyEarthFab\src
..\.venv\Scripts\activate
```
### Launch options

| Flag | Description |
|------|-------------|
| `--prop k` | Property to visualize (default: `k`) |
| `--z-exag N` | Vertical exaggeration (default: 1)|
| `--simplification 0.X` |  Simplify on load (from 0 to 0.99, for e.g 0.5 = collapse 50% of faces, by default is 0) |
| `--tolerance N` |  Clearance in mm for 3D printing (default: 0,5 mm)|
| `--cmap name` | Matplotlib colormap (default: `tab10` or `tab20` depending on the number of layers) |

### Interactive editor

```powershell
# Basic editor
python edit_model.py ..\model_path

# With vertical exaggeration and simplification
python edit_model.py ..\model_path --z-exag 2.5 --simplification 0.85

# With custom printing tolerance (mm per side)
python edit_model.py ..\model_path --tolerance 0.2

# With a custom colormap
python edit_model.py ..\model_path --cmap Set1
```

#### Editor controls

#### Region reassignment

| Key | Action |
|-----|--------|
| `S` | Enter selection mode |
| `1-9` | Pick a group (source or destination depending on mode) |
| `N` / `P` | Navigate between connected components |
| `D` | Pick destination group |
| `G` | Confirm reassignment |
| `Esc` | Cancel |

#### Cutting plane 

| Key | Action |
|-----|--------|
| `T` | Toggle the cutting plane |
| Hold `Z` / `X` / `Y` + wheel | Rotate the plane around the world Z/X/Y axis |
| `Ctrl` + wheel | Move plane along Z |
| `Alt` + wheel | Move plane along X |
| `Shift` + wheel | Move plane along Y |
| `Ctrl`+`Shift` + wheel | Scale the plane |
| `F` | Execute the cut |
| `Esc` | Cancel |

#### Apply tolerance

| `L` | Apply/revert printing tolerance |

#### Export

| `E` | Export visible meshes to 3MF |

## 3D-printing workflow example

1. Open the editor and launch the program: `python edit_model.py ..\model --z-exag 2.5 --simplification 0.85`
2. (Optional) Reassign loose components with `S` → `1-9` → `N/P` → `D` → `1-9` → `G`.
3. (Optional) Cut the model into pieces with `T` → position the plane → `F`.
4. (Optional) Apply assembly tolerance with `L`.
5. Export with `E` → generates 3MF files in `export/editor/`.
6. Open the 3MF files in **Bambu Studio** (or your printer's slicer).

## Dependencies

- **Python** (pinned in `requirements.txt`): flopy, numpy, pyvista, matplotlib,
  pybind11, lib3mf, pyparsing
- **C++**: CGAL, GMP, MPFR (via vcpkg)
- **Build**: CMake, Visual Studio Build Tools 2022

> `pyparsing` is a transitive dependency of matplotlib, pinned explicitly:
> newer pyparsing releases emit deprecation warnings on every matplotlib
> import.

## Troubleshooting

**`pip` fails with `No matching distribution found for vtk<9.4.0`** — your
Python is too new (3.13+). The pinned pyvista requires VTK 9.3.x, which has no
wheels for Python 3.13+. Recreate the environment with Python 3.12:
`py -3.12 -m venv .venv`.

**`ModuleNotFoundError: No module named 'cgal_repair'`** — three possible causes:
1. The module was never built: it is not distributed with the repository; follow
   section 3 of the installation.
2. You are not in `src/`: the import resolves from the current directory; run
   `cd src` first.
3. Python version mismatch: check the tag in the file name
   (`cgal_repair.cp312-...pyd` requires Python 3.12). If it does not match your
   environment, rebuild with the venv activated.

**`ImportError: DLL load failed` when importing cgal_repair** — `gmp-10.dll`
and/or `mpfr-6.dll` are missing from `src/`. Copy them from
`C:\dev\vcpkg\installed\x64-windows\bin\`.

**Deprecation warnings from pyparsing when launching** — the environment has a
pyparsing newer than the pinned one. Run `pip install pyparsing==3.1.4`.

**CMake cannot find CGAL** — verify the vcpkg toolchain was passed:
`-DCMAKE_TOOLCHAIN_FILE=C:/dev/vcpkg/scripts/buildsystems/vcpkg.cmake`.

**`python` is not recognized as a command** — Python is not on the PATH.
Reinstall checking "Add Python to PATH", or use the `py` launcher.

**The window opens and closes immediately** — check that the model path is
correct and contains an `mfsim.nam` file. The path is relative to where you run
the command, not to where the script lives.

**The model looks flat** — use `--z-exag` with a larger value (aquifers are
usually far more extensive than deep).
