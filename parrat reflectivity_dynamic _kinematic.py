import numpy as np
import matplotlib.pyplot as plt

# Constants
lambda_xray = 0.154e-9  # X-ray wavelength in meters (Cu Kα)
k0 = 2 * np.pi / lambda_xray  # wavevector in vacuum

# Layer structure: Air (0), Ni (1), Si (2)
# delta and beta values at Cu Kα wavelength (approximate)
# Define new layer stack with roughness included (50 nm Ni on Si with some interface roughness)
'''layers_rough = [
    {"delta": 0.0, "beta": 0.0, "thickness": 0.0, "roughness": 0.0},         # Air
    {"delta": 6.9e-6, "beta": 1.1e-6, "thickness": 50e-9, "roughness": 0.5e-9}, # Ni with 0.5 nm roughness
    {"delta": 7.6e-6, "beta": 6.3e-8, "thickness": 0.0, "roughness": 0.5e-9}    # Si substrate
]'''

layers_rough = [
    {"delta": 3.4e-6, "beta": 1.0e-8, "thickness": 0.0, "roughness": 0.0},         # Air
    #{"delta": 6.9e-6, "beta": 1.1e-6, "thickness": 50e-9, "roughness": 0.5e-9}, # Ni with 0.5 nm roughness
    {"delta": 7.1e-6, "beta": 1.5e-8, "thickness": 50e-9, "roughness": 0.2e-9}, # SiO2 film
    {"delta": 7.6e-6, "beta": 6.3e-8, "thickness": 0.0, "roughness": 0.5e-9}    # Si substrate
]

# Redefine theta range with high resolution
angles_deg = np.linspace(0.1, 2.5, 4000)
angles_rad = np.radians(angles_deg)

# Calculate kz for each layer across all incident angles
def compute_kzs(layers, angles_rad):
    kzs = []
    for layer in layers:
        n = 1 - layer["delta"] + 1j * layer["beta"]
        kz_layer = k0 * np.sqrt(n**2 - np.cos(angles_rad)**2)
        kzs.append(kz_layer)
    return kzs

# Implement Parratt reflectivity with roughness using your function
def reflectivity_parratt(kzs, layers):
    num_layers = len(layers)
    r_total = np.zeros(len(angles_rad), dtype=complex)
    for j in range(num_layers - 2, -1, -1):
        kz_j = kzs[j]
        kz_j1 = kzs[j + 1]
        r_jj1 = (kz_j - kz_j1) / (kz_j + kz_j1)
        sigma = layers[j + 1]['roughness']
        roughness_factor = np.exp(-2 * kz_j.real * kz_j1.real * sigma**2)
        r_jj1 *= roughness_factor
        thickness = layers[j + 1]['thickness']
        phase = np.exp(2j * kz_j1 * thickness)
        r_total = (r_jj1 + r_total * phase) / (1 + r_jj1 * r_total * phase)
    return np.abs(r_total)**2



# Compute reflectivity
kzs_rough = compute_kzs(layers_rough, angles_rad)
reflectivity_rough = reflectivity_parratt(kzs_rough, layers_rough)

# Also compute reflectivity for ideal case (no roughness) for comparison
layers_no_rough = [dict(layer, roughness=0.0) for layer in layers_rough]
kzs_ideal = compute_kzs(layers_no_rough, angles_rad)
reflectivity_ideal = reflectivity_parratt(kzs_ideal, layers_no_rough)




# Plot comparison
plt.figure(figsize=(8, 5))
plt.semilogy(angles_deg, reflectivity_ideal, label='Ideal Interface')
plt.semilogy(angles_deg, reflectivity_rough, label='With Roughness (0.5 nm)', linestyle='--')
plt.xlabel('Incident Angle θ (degrees)')
plt.ylabel('Reflectivity (log scale)')
plt.title('Reflectivity: Ideal vs. Rough Interface (Parratt Formalism)')
plt.grid(True, which='both', ls='--')
plt.legend()
plt.tight_layout()
plt.show()

