#include <iostream>
#include <fstream>
#include <vector>

// Function to export field data to VTK format for visualization in Paraview
void exportToVTK(const std::vector<double>& nodes, const std::vector<double>& field, const std::string& filename) {
    std::ofstream vtkFile(filename);

    // Header for VTK format
    vtkFile << "# vtk DataFile Version 3.0\n";
    vtkFile << "Field Data\n";
    vtkFile << "ASCII\n";
    vtkFile << "DATASET STRUCTURED_GRID\n";
    vtkFile << "DIMENSIONS " << nodes.size() << " 1 1\n";
    vtkFile << "POINTS " << nodes.size() << " float\n";

    // Output node positions
    for (size_t i = 0; i < nodes.size(); ++i) {
        vtkFile << nodes[i] << " 0 0\n";  // 1D grid, so y and z are 0
    }

    // Output field data as point data
    vtkFile << "POINT_DATA " << field.size() << "\n";
    vtkFile << "SCALARS ElectricField float 1\n";
    vtkFile << "LOOKUP_TABLE default\n";
    
    for (double value : field) {
        vtkFile << value << "\n";
    }

    vtkFile.close();
    std::cout << "Exported field data to " << filename << std::endl;
}

int main() {
    // Example field data (replace with actual data from the FEM solver)
    std::vector<double> nodes = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5};  // Example node positions
    std::vector<double> electricField = {1.0, 0.8, 0.6, 0.4, 0.2, 0.1};  // Example field values

    // Export the data to a VTK file
    exportToVTK(nodes, electricField, "electric_field.vtk");

    return 0;
}

