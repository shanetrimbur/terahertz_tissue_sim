#include <Eigen/Sparse>
#include <iostream>
#include <vector>
#include <cmath>
#include <complex>

// Same Cole-Cole functions as defined earlier...

// FEM Solver setup
void assembleSystemMatrix(Eigen::SparseMatrix<double>& K, const std::vector<double>& nodes, double permittivity) {
    // Permittivity would now be frequency-dependent
    double epsilon = permittivity;

    for (int i = 0; i < nodes.size() - 1; ++i) {
        double dx = nodes[i + 1] - nodes[i];
        K.coeffRef(i, i) += epsilon / dx;
        K.coeffRef(i, i + 1) -= epsilon / dx;
        K.coeffRef(i + 1, i) -= epsilon / dx;
        K.coeffRef(i + 1, i + 1) += epsilon / dx;
    }
}

void applyPermittivityAndSolve(const std::vector<double>& frequencies, const ColeColeParams& tissueParams) {
    int numNodes = 100;
    std::vector<double> nodes(numNodes);
    for (int i = 0; i < numNodes; ++i) nodes[i] = i * 1.0 / (numNodes - 1);

    for (double frequency : frequencies) {
        std::complex<double> permittivityComplex = coleColePermittivity(tissueParams, frequency);
        double permittivityReal = permittivityComplex.real();

        Eigen::SparseMatrix<double> K(numNodes, numNodes);
        assembleSystemMatrix(K, nodes, permittivityReal);

        // Apply boundary conditions and solve system (as explained in previous FEM code)
        // Solve the system for the given permittivity at this frequency...
        std::cout << "Solving for frequency: " << frequency << " Hz with permittivity: " << permittivityReal << std::endl;

        // Further simulation steps go here...
    }
}

int main() {
    // Example Cole-Cole parameters for muscle tissue (same as before)
    ColeColeParams muscle {
        4.0,   // epsilon_infinity
        0.5,   // sigma_s (S/m)
        {6.0, 3.0},   // delta_epsilon values for two relaxation processes
        {1e-11, 1e-9},  // Relaxation times for the two processes
        {0.1, 0.2}  // Alpha values for the two processes
    };

    // Frequency range for the simulation (same as before)
    std::vector<double> frequencies;
    for (double f = 1e9; f <= 1e13; f *= 1.1) {
        frequencies.push_back(f);  // Logarithmic scaling
    }

    // Apply the permittivity in FEM simulation and solve
    applyPermittivityAndSolve(frequencies, muscle);

    return 0;
}

