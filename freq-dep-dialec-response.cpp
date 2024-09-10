#include <iostream>
#include <fstream>
#include <complex>
#include <vector>
#include <cmath>

const double epsilon_0 = 8.854e-12; // Permittivity of free space (F/m)
const double pi = 3.141592653589793;

struct ColeColeParams {
    double epsilon_infinity;   // Permittivity at infinite frequency
    double sigma_s;            // Static conductivity
    std::vector<double> delta_epsilon; // Differences between static and high-frequency permittivities
    std::vector<double> tau;   // Relaxation times
    std::vector<double> alpha; // Distribution parameter for each process
};

// Function to calculate complex permittivity using the Cole-Cole model
std::complex<double> coleColePermittivity(const ColeColeParams& params, double frequency) {
    std::complex<double> j(0.0, 1.0);  // Imaginary unit
    double omega = 2 * pi * frequency; // Angular frequency (rad/s)

    // Initial value for permittivity
    std::complex<double> epsilon = params.epsilon_infinity;

    // Sum over all relaxation processes
    for (size_t i = 0; i < params.tau.size(); i++) {
        epsilon += params.delta_epsilon[i] /
                   (1.0 + std::pow(j * omega * params.tau[i], 1.0 - params.alpha[i]));
    }

    // Add conductivity term (contributes to the imaginary part)
    epsilon -= j * (params.sigma_s / (omega * epsilon_0));

    return epsilon;
}

int main() {
    // Example Cole-Cole parameters for muscle tissue (hypothetical values)
    ColeColeParams muscle {
        4.0,   // epsilon_infinity
        0.5,   // sigma_s (S/m)
        {6.0, 3.0},   // delta_epsilon values for two relaxation processes
        {1e-11, 1e-9},  // Relaxation times for the two processes
        {0.1, 0.2}  // Alpha values for the two processes
    };

    // Frequency range: from 1 GHz to 10 THz
    std::vector<double> frequencies;
    for (double f = 1e9; f <= 1e13; f *= 1.1) {
        frequencies.push_back(f);  // Logarithmic scaling
    }

    // Output file to store the results
    std::ofstream file("permittivity_data.csv");
    file << "Frequency,Real_Permittivity,Imag_Permittivity\n";

    // Calculate permittivity for each frequency and save results
    for (double f : frequencies) {
        std::complex<double> epsilon = coleColePermittivity(muscle, f);
        file << f << "," << epsilon.real() << "," << epsilon.imag() << "\n";
    }

    file.close();
    std::cout << "Data saved to permittivity_data.csv" << std::endl;

    return 0;
}

