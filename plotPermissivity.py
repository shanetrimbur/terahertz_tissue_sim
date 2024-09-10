# Python code using matplotlib to visualize the permittivity data

import matplotlib.pyplot as plt
import pandas as pd

# Read the CSV file
data = pd.read_csv('permittivity_data.csv')

# Plot the real and imaginary parts of permittivity
plt.figure(figsize=(10, 6))
plt.plot(data['Frequency'], data['Real_Permittivity'], label='Real Part of Permittivity')
plt.plot(data['Frequency'], data['Imag_Permittivity'], label='Imaginary Part of Permittivity', linestyle='--')

plt.xscale('log')
plt.yscale('linear')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Permittivity')
plt.title('Frequency-Dependent Dielectric Response of Muscle Tissue')
plt.legend()
plt.grid(True)
plt.show()

