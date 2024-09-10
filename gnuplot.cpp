#include <iostream>
#include <fstream>
#include <vector>

// Function to export data for gnuplot visualization
void exportToGnuplot(const std::vector<double>& nodes, const std::vector<double>& field, const std::string& filename) {
    std::ofstream file(filename);

    // Write node positions and field values
    for (size_t i = 0; i < nodes.size(); ++i) {
        file << nodes[i] << " " << field[i] << "\n";
    }

    file.close();
    std::cout << "Exported data to " << filename << " for gnuplot." << std::endl;
}

int main() {
    // Example field data (replace with actual data from the FEM solver)
    std::vector<double> nodes = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5};  // Example node positions
    std::vector<double> electricField = {1.0, 0.8, 0.6, 0.4, 0.2, 0.1};  // Example field values

    // Export the data to a file for gnuplot
    exportToGnuplot(nodes, electricField, "field_data.dat");

    return 0;
}

