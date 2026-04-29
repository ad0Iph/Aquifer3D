# Aquifer3D

Herramientas para visualización, exportación e impresión 3D de modelos de acuíferos MODFLOW 6.

## Estructura del proyecto

```
Aquifer3D/
├── src/
│   ├── aquifer_grid_m6.py    # Carga modelo MODFLOW → grid PyVista
│   ├── show_model.py         # Visualizador con checkboxes por propiedad
│   ├── export_model.py       # Exportador a PLY con colores
│   └── mesh_repair.py        # Wrapper reparación/simplificación (CGAL o PyVista)
│
├── cgal_repair/              # Módulo C++ (compilar una vez)
│   ├── src/cgal_repair.cpp   # Binding pybind11 + CGAL
│   ├── CMakeLists.txt
│   ├── setup.py
│   └── README.md
│
├── export/                   # Generado por export_model.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Instalación

### 1. Entorno Python

```bash
git clone <url-del-repo>
cd Aquifer3D
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. CGAL y pybind11 (necesario para reparación y simplificación de mallas)

Sin CGAL el proyecto funciona pero usa un fallback de PyVista menos robusto para reparar mallas.

#### Windows (vcpkg)

```bash
# Instalar vcpkg (una sola vez, fuera del proyecto)
cd C:\dev
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg
bootstrap-vcpkg.bat

# Instalar dependencias (requiere Visual Studio Build Tools con C++)
vcpkg install cgal:x64-windows pybind11:x64-windows
```

#### Linux (apt)

```bash
sudo apt install libcgal-dev libgmp-dev libmpfr-dev
```

#### macOS (brew)

```bash
brew install cgal
```

### 3. Compilar cgal_repair

#### Windows

```bash
cd Aquifer3D\cgal_repair
mkdir build
cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=C:/dev/vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build . --config Release
```

Copiar el módulo compilado y las DLLs necesarias a `src/`:

```bash
copy Release\cgal_repair.*.pyd ..\..\src\
copy C:\dev\vcpkg\installed\x64-windows\bin\gmp.dll ..\..\src\
copy C:\dev\vcpkg\installed\x64-windows\bin\mpfr-6.dll ..\..\src\
```

#### Linux / macOS

```bash
cd Aquifer3D/cgal_repair
mkdir build && cd build
cmake .. -Dpybind11_DIR=$(python -c "import pybind11; print(pybind11.get_cmake_dir())")
make -j$(nproc)
cp cgal_repair*.so ../../src/
```

### 4. Verificar instalación

```bash
cd src
python -c "import cgal_repair; print('CGAL OK')"
```

## Uso

### Visualización

```bash
cd src

# Visualización normal
python show_model.py ../ruta_modelo

# Con exageración vertical
python show_model.py ../ruta_modelo --z-exag 10

# Checkboxes por valor de k (un color por cada valor distinto)
python show_model.py ../ruta_modelo --split-unique

# Reparar mallas con CGAL
python show_model.py ../ruta_modelo --split-unique --repair

# Simplificar 50% de las caras
python show_model.py ../ruta_modelo --split-unique --decimate 0.5

# Plano de corte interactivo
python show_model.py ../ruta_modelo --clip

# Dividir en bloques NxN
python show_model.py ../ruta_modelo --cuts 3
```

### Exportación

```bash
# Exportar modelo completo
python export_model.py ../ruta_modelo

# Un PLY por cada valor distinto de k
python export_model.py ../ruta_modelo --split-unique

# Reparar + simplificar para impresión 3D
python export_model.py ../ruta_modelo --split-unique --repair --decimate 0.5

# Un PLY por capa geológica
python export_model.py ../ruta_modelo --layers

# Combinaciones
python export_model.py ../ruta_modelo --split-unique --layers
python export_model.py ../ruta_modelo --cuts 2 --repair
```

### Opciones comunes

| Flag | Descripción |
|------|-------------|
| `--prop k` | Propiedad a visualizar (default: `k`) |
| `--z-exag N` | Exageración vertical |
| `--no-log` | Escala lineal en vez de logarítmica |
| `--split-unique` | Separar por valores distintos de la propiedad |
| `--repair` | Reparar mallas (watertight) con CGAL |
| `--decimate 0.X` | Simplificar malla (0.5 = eliminar 50% de caras) |
| `--layers` | Exportar por capa geológica |
| `--cuts N` | Subdividir en NxN bloques |
| `--export ruta` | (show_model) Exportar además de visualizar |
| `--out archivo.ply` | (export_model) Archivo de salida |
| `--out-dir carpeta` | (export_model) Directorio de salida |

## Pipeline de reparación CGAL

Cuando se usa `--repair`, la reparación sigue este pipeline:

1. **repair_polygon_soup** — elimina vértices y caras duplicados/degenerados
2. **orient_polygon_soup** — orienta todas las caras consistentemente
3. **polygon_soup_to_polygon_mesh** — construye un Surface_mesh válido
4. **stitch_borders** — une bordes abiertos que coinciden geométricamente
5. **remove_degenerate_faces** — limpia caras con área cero
6. **triangulate_faces** — asegura que todo son triángulos
7. **orient_to_bound_a_volume** — normales hacia afuera (si la malla es cerrada)

Cuando se usa `--decimate`, la simplificación usa **edge collapse** de CGAL con costo por longitud de arista y colocación en punto medio, que preserva mejor la geometría que `pyvista.decimate`.

## Dependencias

- **Python**: flopy, numpy, pyvista, matplotlib, pybind11
- **C++ (opcional)**: CGAL, GMP, MPFR
- **Compilación**: CMake, Visual Studio Build Tools (Windows) o GCC (Linux)