class FDTD {
private:
    int gridSize;
    double deltaTime;
    double deltaSpace;
    std::vector<double> electricField;
    std::vector<double> magneticField;
    Tissue tissue;
    RFSignal signal;

public:
    FDTD(int grid, double dt, double ds, Tissue t, RFSignal s)
        : gridSize(grid), deltaTime(dt), deltaSpace(ds), tissue(t), signal(s),
          electricField(gridSize, 0.0), magneticField(gridSize, 0.0) {}

    // FDTD step update: Simulate electric and magnetic fields
    void updateFields() {
        // Update magnetic field based on Maxwell's equations
        for (int i = 0; i < gridSize - 1; i++) {
            magneticField[i] += (deltaTime / tissue.permeability) * (electricField[i + 1] - electricField[i]) / deltaSpace;
        }

        // Update electric field based on Maxwell's equations
        for (int i = 1; i < gridSize; i++) {
            electricField[i] += (deltaTime / tissue.permittivity) * (magneticField[i] - magneticField[i - 1]) / deltaSpace
                                - deltaTime * tissue.conductivity * electricField[i]; // Taking into account tissue conductivity
        }
    }

    // Function to run the simulation for a specific number of steps
    void runSimulation(int steps) {
        for (int t = 0; t < steps; ++t) {
            updateFields();
            if (t % 10 == 0) {
                visualize(t);
            }
        }
    }

    // Simple visualization of the electric field values at each step
    void visualize(int step) {
        std::cout << "Step: " << step << " Electric Field: ";
        for (int i = 0; i < gridSize; i++) {
            std::cout << electricField[i] << " ";
        }
        std::cout << std::endl;
    }
};

