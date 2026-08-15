from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import subprocess
import sys

def get_cgal_flags():
    """Attempts to obtain include/lib paths of CGAL."""
    try:
        inc = subprocess.check_output(
            ["pkg-config", "--cflags", "cgal"],
            stderr=subprocess.DEVNULL
        ).decode().strip().split()
        libs = subprocess.check_output(
            ["pkg-config", "--libs", "cgal"],
            stderr=subprocess.DEVNULL
        ).decode().strip().split()
        return inc, libs
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    return [], []

inc_flags, lib_flags = get_cgal_flags()

ext_modules = [
    Pybind11Extension(
        "cgal_repair",
        ["src/cgal_repair.cpp"],
        extra_compile_args=inc_flags + ["-std=c++17", "-O2"],
        extra_link_args=lib_flags,
        language="c++",
    ),
]

setup(
    name="cgal_repair",
    version="0.1.0",
    author="PyEarthFab",
    description="Reparation and simplification of meshes with CGAL for MODFLOW models in python.",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    install_requires=["pybind11>=2.10", "numpy"],
    python_requires=">=3.8",
)