#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

// Kernel y Surface_mesh
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>

// Polygon Mesh Processing (reparación)
#include <CGAL/Polygon_mesh_processing/repair_polygon_soup.h>
#include <CGAL/Polygon_mesh_processing/orient_polygon_soup.h>
#include <CGAL/Polygon_mesh_processing/polygon_soup_to_polygon_mesh.h>
#include <CGAL/Polygon_mesh_processing/orientation.h>
#include <CGAL/Polygon_mesh_processing/stitch_borders.h>
#include <CGAL/Polygon_mesh_processing/triangulate_faces.h>
#include <CGAL/Polygon_mesh_processing/self_intersections.h>
#include <CGAL/Polygon_mesh_processing/repair.h>

// Surface Mesh Simplification
#include <CGAL/Surface_mesh_simplification/edge_collapse.h>
#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/Count_ratio_stop_predicate.h>
#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/Edge_length_cost.h>
#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/Midpoint_placement.h>

#include <vector>
#include <array>
#include <unordered_map>

namespace py  = pybind11;
namespace PMP = CGAL::Polygon_mesh_processing;
namespace SMS = CGAL::Surface_mesh_simplification;

using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel;
using Point3 = Kernel::Point_3;
using Mesh   = CGAL::Surface_mesh<Point3>;

// ── Utilidades internas ─────────────────────────────────────────────

/// Construye un Surface_mesh desde arrays de NumPy (vértices + caras).
static Mesh build_mesh(py::array_t<double> vertices_np, py::array_t<int> faces_np)
{
    auto v_buf = vertices_np.unchecked<2>();
    auto f_buf = faces_np.unchecked<2>();

    size_t n_verts = v_buf.shape(0);
    size_t n_faces = f_buf.shape(0);

    Mesh mesh;
    std::vector<Mesh::Vertex_index> vi(n_verts);

    for (size_t i = 0; i < n_verts; ++i) {
        vi[i] = mesh.add_vertex(Point3(v_buf(i, 0), v_buf(i, 1), v_buf(i, 2)));
    }

    for (size_t i = 0; i < n_faces; ++i) {
        mesh.add_face(
            vi[static_cast<size_t>(f_buf(i, 0))],
            vi[static_cast<size_t>(f_buf(i, 1))],
            vi[static_cast<size_t>(f_buf(i, 2))]
        );
    }

    return mesh;
}

/// Extrae vértices y caras de un Surface_mesh a arrays de NumPy.
static std::tuple<py::array_t<double>, py::array_t<int>>
extract_mesh(const Mesh& mesh)
{
    size_t nv = mesh.number_of_vertices();
    size_t nf = mesh.number_of_faces();

    py::array_t<double> out_verts({nv, size_t(3)});
    py::array_t<int>    out_faces({nf, size_t(3)});

    auto ov = out_verts.mutable_unchecked<2>();
    auto of = out_faces.mutable_unchecked<2>();

    std::unordered_map<Mesh::Vertex_index, size_t> v_map;
    size_t idx = 0;
    for (auto v : mesh.vertices()) {
        auto& pt = mesh.point(v);
        ov(idx, 0) = pt.x();
        ov(idx, 1) = pt.y();
        ov(idx, 2) = pt.z();
        v_map[v] = idx;
        ++idx;
    }

    idx = 0;
    for (auto f : mesh.faces()) {
        auto h = mesh.halfedge(f);
        of(idx, 0) = static_cast<int>(v_map[mesh.target(h)]);
        h = mesh.next(h);
        of(idx, 1) = static_cast<int>(v_map[mesh.target(h)]);
        h = mesh.next(h);
        of(idx, 2) = static_cast<int>(v_map[mesh.target(h)]);
        ++idx;
    }

    return {out_verts, out_faces};
}

/// Repara una malla para hacerla watertight.
std::tuple<py::array_t<double>, py::array_t<int>>
repair_mesh(py::array_t<double> vertices_np, py::array_t<int> faces_np)
{
    auto v_buf = vertices_np.unchecked<2>();
    auto f_buf = faces_np.unchecked<2>();

    size_t n_verts = v_buf.shape(0);
    size_t n_faces = f_buf.shape(0);

    // 1. Polygon soup
    std::vector<Point3> points;
    points.reserve(n_verts);
    for (size_t i = 0; i < n_verts; ++i)
        points.emplace_back(v_buf(i, 0), v_buf(i, 1), v_buf(i, 2));

    std::vector<std::vector<size_t>> polygons;
    polygons.reserve(n_faces);
    for (size_t i = 0; i < n_faces; ++i)
        polygons.push_back({
            static_cast<size_t>(f_buf(i, 0)),
            static_cast<size_t>(f_buf(i, 1)),
            static_cast<size_t>(f_buf(i, 2))
        });

    // 2. Reparar y orientar
    PMP::repair_polygon_soup(points, polygons);
    PMP::orient_polygon_soup(points, polygons);

    // 3. Construir mesh
    Mesh mesh;
    PMP::polygon_soup_to_polygon_mesh(points, polygons, mesh);

    // 4. Unir bordes, limpiar, triangular
    PMP::stitch_borders(mesh);
    PMP::remove_degenerate_edges(mesh);
    PMP::remove_degenerate_faces(mesh);
    PMP::triangulate_faces(mesh);
    PMP::duplicate_non_manifold_vertices(mesh);

    std::vector<Mesh::Halfedge_index> cycles;
    PMP::extract_boundary_cycles(mesh, std::back_inserter(cycles));
    for (auto h : cycles) {
        std::vector<Mesh::Face_index> patch;
        PMP::triangulate_hole(mesh, h,
            CGAL::parameters::face_output_iterator(std::back_inserter(patch)));
    }

    mesh.collect_garbage();

    if (CGAL::is_closed(mesh))
        PMP::orient_to_bound_a_volume(mesh);

    return extract_mesh(mesh);
}

std::tuple<py::array_t<double>, py::array_t<int>>
simplify_mesh(py::array_t<double> vertices_np, py::array_t<int> faces_np,
              double ratio)
{
    Mesh mesh = build_mesh(vertices_np, faces_np);

    SMS::Count_ratio_stop_predicate<Mesh> stop(ratio);

    SMS::edge_collapse(
        mesh,
        stop,
        CGAL::parameters::get_cost(SMS::Edge_length_cost<Mesh>())
                         .get_placement(SMS::Midpoint_placement<Mesh>())
    );

    mesh.collect_garbage();

    return extract_mesh(mesh);
}

/// Repara + simplifica en un solo paso.
std::tuple<py::array_t<double>, py::array_t<int>>
repair_and_simplify(py::array_t<double> vertices_np, py::array_t<int> faces_np,
                    double ratio)
{
    // Primero reparar
    auto [rep_verts, rep_faces] = repair_mesh(vertices_np, faces_np);

    // Después simplificar
    auto [simp_verts, simp_faces] = simplify_mesh(rep_verts, rep_faces, ratio);

    return repair_mesh(simp_verts, simp_faces);
}

/// Verifica si una malla es cerrada (watertight).
bool is_closed(py::array_t<double> vertices_np, py::array_t<int> faces_np)
{
    auto v_buf = vertices_np.unchecked<2>();
    auto f_buf = faces_np.unchecked<2>();

    std::vector<Point3> points;
    for (size_t i = 0; i < (size_t)v_buf.shape(0); ++i)
        points.emplace_back(v_buf(i, 0), v_buf(i, 1), v_buf(i, 2));

    std::vector<std::vector<size_t>> polygons;
    for (size_t i = 0; i < (size_t)f_buf.shape(0); ++i)
        polygons.push_back({
            static_cast<size_t>(f_buf(i, 0)),
            static_cast<size_t>(f_buf(i, 1)),
            static_cast<size_t>(f_buf(i, 2))
        });

    PMP::repair_polygon_soup(points, polygons);
    PMP::orient_polygon_soup(points, polygons);

    Mesh mesh;
    PMP::polygon_soup_to_polygon_mesh(points, polygons, mesh);

    return CGAL::is_closed(mesh);
}

// ── Módulo pybind11 ─────────────────────────────────────────────────

PYBIND11_MODULE(cgal_repair, m) {
    m.doc() = "Reparación y simplificación de mallas usando CGAL";

    m.def("repair_mesh", &repair_mesh,
          py::arg("vertices"), py::arg("faces"),
          "Repara una malla triangulada para hacerla watertight.");

    m.def("simplify_mesh", &simplify_mesh,
          py::arg("vertices"), py::arg("faces"), py::arg("ratio"),
          R"doc(
          Simplifica una malla reduciendo caras mediante edge collapse.

          Parameters
          ----------
          vertices : np.ndarray (N, 3) float64
          faces : np.ndarray (M, 3) int32
          ratio : float
              Fracción de aristas a mantener (0.5 = ~50% de caras).

          Returns
          -------
          tuple(np.ndarray, np.ndarray)
          )doc");

    m.def("repair_and_simplify", &repair_and_simplify,
          py::arg("vertices"), py::arg("faces"), py::arg("ratio"),
          "Repara y simplifica en un solo paso.");

    m.def("is_closed", &is_closed,
          py::arg("vertices"), py::arg("faces"),
          "Verifica si una malla es cerrada (watertight).");
}