#include <iostream>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>

// Function to solve the system Ax = b for each time step
void solveTimeStep(const SpMat& K, const SpMat& M, Vec& E, const Vec& b, double deltaTime) {
    // K and M are the stiffness and mass matrices, E is the electric field vector, b is the source vector

    // Define A = M + Δt * K (implicit Euler method)
    SpMat A = M + deltaTime * K;

    // Use a sparse matrix solver (e.g., LU decomposition) to solve the system A * E_new = b
    Eigen::SparseLU<SpMat> solver;
    solver.compute(A);
    if (solver.info() != Eigen::Success) {
        std::cerr << "Matrix factorization failed!" << std::endl;
        return;
    }

    // Solve the system
    E = solver.solve(b);
    if (solver.info() != Eigen::Success) {
        std::cerr << "Solving system failed!" << std::endl;
    }
}

int main() {
    // Set up the FEM mesh, system matrices, and initial conditions

    int numNodes = 100;  // Number of nodes in the mesh
    std::vector<double> nodes(numNodes);  // Mesh nodes
    for (int i = 0; i < numNodes; ++i) nodes[i] = i * 1.0 / (numNodes - 1);  // Uniform 1D mesh

    SpMat K(numNodes, numNodes), M(numNodes, numNodes);  // Stiffness and mass matrices
    Vec E(numNodes), b(numNodes);  // Electric field vector and source vector

    // Define the properties of the biological tissue
    TissueProperties tissue {9.0, 0.5, 1.0};  // Example values for permittivity, conductivity, and permeability

    // Assemble the system matrix
    assembleSystemMatrix(K, M, nodes, tissue);

    // Apply boundary conditions
    applyBoundaryConditions(K, b);

    // Time-stepping loop
    double deltaTime = 1e-15;  // Time step in seconds
    int numTimeSteps = 1000;   // Number of time steps
    for (int t = 0; t < numTimeSteps; ++t) {
        solveTimeStep(K, M, E, b, deltaTime);
        // Optionally: Visualize or store the field values here
    }

    return 0;
}

