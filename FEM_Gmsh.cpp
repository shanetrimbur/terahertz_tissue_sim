#include <iostream>
#include <vector>
#include <gmsh.h>

// Function to generate a 2D mesh using Gmsh
void generateMesh() {
    gmsh::initialize();
    gmsh::model::add("tissue_mesh");

    // Define the tissue as a rectangle (e.g., 1m x 1m)
    double x0 = 0, y0 = 0, x1 = 1, y1 = 1;
    gmsh::model::geo::addPoint(x0, y0, 0, 1.0, 1);
    gmsh::model::geo::addPoint(x1, y0, 0, 1.0, 2);
    gmsh::model::geo::addPoint(x1, y1, 0, 1.0, 3);
    gmsh::model::geo::addPoint(x0, y1, 0, 1.0, 4);

    // Create lines between the points
    gmsh::model::geo::addLine(1, 2, 1);
    gmsh::model::geo::addLine(2, 3, 2);
    gmsh::model::geo::addLine(3, 4, 3);
    gmsh::model::geo::addLine(4, 1, 4);

    // Create a loop for the surface and define the surface
    gmsh::model::geo::addCurveLoop({1, 2, 3, 4}, 1);
    gmsh::model::geo::addPlaneSurface({1}, 1);

    // Synchronize the model and generate the mesh
    gmsh::model::geo::synchronize();
    gmsh::model::mesh::generate(2);  // Generate a 2D mesh
    gmsh::write("tissue.msh");

    gmsh::finalize();
}

int main() {
    generateMesh();
    return 0;
}

