int main() {
    // Define the properties of the tissue
    double permittivity = 9.0;  // Relative permittivity for biological tissue
    double conductivity = 0.5;  // Conductivity of the tissue in S/m
    double permeability = 1.0;  // Permeability (constant for non-magnetic tissues)
    
    Tissue tissue(permittivity, conductivity, permeability);

    // Define the properties of the RF signal
    double frequency = 1e12;  // Frequency of 1 THz
    double amplitude = 1.0;   // Signal amplitude

    RFSignal rfSignal(frequency, amplitude);

    // Setup FDTD grid and time/space steps
    int gridSize = 100;
    double deltaTime = 1e-15;  // Time step in seconds
    double deltaSpace = 1e-3;  // Space step in meters

    FDTD simulation(gridSize, deltaTime, deltaSpace, tissue, rfSignal);

    // Run the simulation for 100 steps
    simulation.runSimulation(100);

    return 0;
}

