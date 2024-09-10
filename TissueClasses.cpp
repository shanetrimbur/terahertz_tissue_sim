#include <iostream>
#include <vector>
#include <cmath>

// Constants for the simulation
const double PI = 3.14159265358979323846;
const double SPEED_OF_LIGHT = 3e8;  // Speed of light in vacuum (m/s)

// Tissue Class: Represents the properties of biological tissue
class Tissue {
public:
    double permittivity;  // Relative permittivity of the tissue
    double conductivity;  // Conductivity of the tissue (S/m)
    double permeability;  // Permeability (constant in biological tissues)

    Tissue(double epsilon, double sigma, double mu)
        : permittivity(epsilon), conductivity(sigma), permeability(mu) {}
};

// RF Signal Class: Represents the properties of the terahertz signal
class RFSignal {
public:
    double frequency;  // Frequency of the RF signal (Hz)
    double amplitude;  // Amplitude of the RF signal

    RFSignal(double freq, double amp) : frequency(freq), amplitude(amp) {}

    // Function to calculate the wavelength of the RF signal in vacuum
    double getWavelength() const {
        return SPEED_OF_LIGHT / frequency;
    }
};

