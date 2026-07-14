# Aquifer3D

Herramientas para visualización, edición, exportación e impresión 3D de modelos de acuíferos MODFLOW 6.

## Estructura del proyecto

```
Aquifer3D/
├── src/
│   ├── aquifer_grid_m6.py    # Carga modelo MODFLOW hacia grid en PyVista
│   ├── mesh_repair.py        # Wrapper reparación/simplificación 
│   ├── edit_model.py         # Launcher
│   └── editor/               # Editor interactivo (paquete)
│       ├── __init__.py
│       ├── editor.py         # Clase principal AquiferEditor
│       ├── utils.py          # Utilidades compartidas
│       ├── surfaces.py       # Construcción de superficies
│       ├── rendering.py      # Actores y visibilidad
│       ├── selection.py      # Selección de componentes conexos entre propiedades
│       ├── cut_plane.py      # Plano de corte
│       ├── printing.py       # Tolerancia para impresión 3D
│       └── export_3mf.py     # Exportación a .3mf
│
├── cgal_repair/              # Módulo C++
│   ├── src/cgal_repair.cpp   # Binding pybind11 + CGAL
│   ├── CMakeLists.txt
│   ├── setup.py
│   └── README.md
│
├── export/                   # Carpeta generada por los exportadores
├── requirements.txt
├── .gitignore
└── README.md
```

## Requisitos previos (Windows)

Antes de comenzar, instalar:

1. **Python 3.11 o superior** — [python.org/downloads](https://www.python.org/downloads/)
   - Durante la instalación marcar  **"Add Python to PATH"**
2. **Git** — [git-scm.com/downloads](https://git-scm.com/downloads)
3. **Visual Studio Build Tools 2022** (solo para compilar el módulo CGAL) — [visualstudio.microsoft.com/downloads](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022)
   - En el instalador, seleccionar la carga de trabajo **"Desarrollo para el escritorio con C++"**
4. **CMake** — [cmake.org/download](https://cmake.org/download/)
   - Durante la instalación marcar  **"Add CMake to the system PATH"**

Verificar que todo quedó instalado (abrir una terminal nueva):

```powershell
python --version    # Python 3.11+
git --version       # git version 2.x
cmake --version     # cmake version 3.x
```

## Instalación

### 1. Clonar el repositorio y crear el entorno

```powershell
git clone <url-del-repo>
cd Aquifer3D
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

> **Nota:** cada vez que abras una terminal nueva para usar el proyecto, debes activar el entorno con `.venv\Scripts\activate`.

### 2. Instalar vcpkg y las dependencias C++

vcpkg es el gestor de paquetes C++ de Microsoft. Se instala **una sola vez, fuera del proyecto**:

```powershell
cd C:\
mkdir dev
cd dev
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat
```

Instalar CGAL y pybind11 (esto puede tardar 30-60 minutos la primera vez):

```powershell
.\vcpkg install cgal:x64-windows pybind11:x64-windows
```

### 3. Compilar el módulo cgal_repair

Volver al directorio del proyecto:

```powershell
cd <ruta-al-proyecto>\Aquifer3D\cgal_repair
mkdir build
cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=C:/dev/vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build . --config Release
```

Copiar el módulo compilado y las DLLs a `src/`:

```powershell
copy Release\cgal_repair.*.pyd ..\..\src\
copy C:\dev\vcpkg\installed\x64-windows\bin\gmp.dll ..\..\src\
copy C:\dev\vcpkg\installed\x64-windows\bin\mpfr-6.dll ..\..\src\
```

### 4. Verificar la instalación

```powershell
cd <ruta-al-proyecto>\Aquifer3D\src
python -c "import cgal_repair; print('CGAL OK')"
python -c "import flopy, numpy, pyvista, matplotlib, pybind11, lib3mf; print('Python deps OK')"
```

Si ambos comandos imprimen OK, la instalación está completa.

## Uso

Todos los comandos se ejecutan desde la carpeta `src/` con el entorno activado:

```powershell
cd <ruta-al-proyecto>\Aquifer3D\src
..\.venv\Scripts\activate
```

### Editor interactivo

```powershell
# Editor básico
python edit_model.py ..\ruta_modelo

# Con exageración vertical y simplificación
python edit_model.py ..\ruta_modelo --z-exag 2.5 --simplification 0.85

# Con tolerancia de impresión personalizada (mm por lado)
python edit_model.py ..\ruta_modelo --tolerance 0.2

# Con colormap personalizado
python edit_model.py ..\ruta_modelo --cmap Set1
```

#### Controles del editor

| Tecla | Acción |
|-------|--------|
| `S` | Entrar en modo selección |
| `1-9` | Elegir grupo (fuente o destino según el modo) |
| `N` / `P` | Navegar entre componentes conexos |
| `D` | Elegir grupo destino |
| `G` | Confirmar reasignación |
| `Esc` | Cancelar |
| `T` | Activar/desactivar plano de corte |
|  `Z`+rueda | Rotar plano en Z |
|  `X`+rueda | Rotar plano en X |
|  `Y`+rueda | Rotar plano en Y |
| `Ctrl`+rueda | Mover plano en Z |
| `Alt`+rueda | Mover plano en X |
| `Shift`+rueda | Mover plano en Y |
| `Ctrl`+`Shift`+rueda | Escalar el plano |
| `F` | Ejecutar corte |
| `C` | Borrar líneas de corte |
| `L` | Aplicar/revertir tolerancia de impresión |
| `E` | Exportar mallas visibles a 3MF |


### Opciones de lanzamiento

| Flag | Descripción |
|------|-------------|
| `--prop k` | Propiedad a visualizar (default: `k`) |
| `--z-exag N` | Exageración vertical |
| `--decimate 0.X` | Simplificar malla (0.5 = eliminar 50% de caras) |
| `--simplification 0.X` | Simplificar al cargar |
| `--tolerance N` | Holgura en mm para impresión 3D |
| `--cmap nombre` | Colormap de matplotlib |

## Flujo de trabajo para impresión 3D

1. Abrir el editor: `python edit_model.py ..\modelo --z-exag 2.5 --simplification 0.85`.
2. (Opcional) Reasignar componentes sueltos con `S` → `1-9` → `N/P` → `D` → `G`.
3. (Opcional) Cortar el modelo en piezas con `T` → posicionar plano → `F`.
4. Aplicar tolerancia de ensamblaje con `L`, esto debe hacerse manualmente.
5. Exportar con `E` → genera archivos 3MF en `export/editor/`.
6. Abrir los 3MF en **Bambu Studio** (o el slicer de su preferencia).


## Dependencias

- **Python**: flopy, numpy, pyvista, matplotlib, pybind11, lib3mf
- **C++**: CGAL, GMP, MPFR (via vcpkg)
- **Compilación**: CMake, Visual Studio Build Tools 2022

## Problemas comunes

**`ImportError: DLL load failed` al importar cgal_repair** — faltan `gmp-10.dll` y/o `mpfr-6.dll` en `src/`. Copiarlas desde `C:\dev\vcpkg\installed\x64-windows\bin\`.

**CMake no encuentra CGAL** — verificar que se pasó el toolchain de vcpkg: `-DCMAKE_TOOLCHAIN_FILE=C:/dev/vcpkg/scripts/buildsystems/vcpkg.cmake`.
