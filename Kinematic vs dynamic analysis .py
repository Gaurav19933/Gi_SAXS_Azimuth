import numpy as np
import matplotlib.pyplot as plt

# Constants
wavelength = 1.54e-10  # Cu K-alpha in meters
K = 2 * np.pi / wavelength

# Layer definitions (air-film-substrate)
n_layers = np.array([1, 1 - 4.6e-6 + 1j*3.7e-7, 1 - 7.6e-6 + 1j*1.75e-7])
thickness = np.array([0, 20e-9, 0])  # Thickness: air (inf), film (20 nm), substrate (inf)

angles_deg = np.linspace(0.01, 5, 500)
angles_rad = np.deg2rad(angles_deg)

# Parratt's algorithm (dynamic)
def parratt_reflectivity(n, d, theta, K):
    kz = K * np.sqrt(n[:, None]**2 - np.cos(theta)**2)
    r = (kz[:-1] - kz[1:]) / (kz[:-1] + kz[1:])
    R = np.zeros(theta.shape, dtype=complex)

    for i in range(len(theta)):
        X = 0
        for j in reversed(range(len(d) - 1)):
            exp_term = np.exp(-2j * kz[j+1, i] * d[j+1])
            X = (r[j, i] + X * exp_term) / (1 + r[j, i] * X * exp_term)
        R[i] = X

    return np.abs(R)**2

# Corrected Kinematic approximation
def kinematic_reflectivity(n, d, theta, K):
    Qz = 2 * K * np.sin(theta)
    delta_rho = n[:-1]**2 - n[1:]**2
    z_interfaces = np.cumsum(d)
    R = np.zeros_like(Qz, dtype=complex)

    for j in range(len(d)-1):
        dz = d[j+1]
        z = z_interfaces[j]
        R += delta_rho[j] * np.exp(1j * Qz * z) * (1 - np.exp(-1j * Qz * dz)) / (1j * Qz)

    return np.abs(R)**2

# Normalize reflectivity explicitly

# Calculations
R_parratt = parratt_reflectivity(n_layers, thickness, angles_rad, K)
R_kinematic = kinematic_reflectivity(n_layers, thickness, angles_rad, K)

# normalize reflectivity
R_parratt_norm = R_parratt / np.max(R_parratt)
R_kinematic_norm = R_kinematic / np.max(R_kinematic)

# Combined Plotting with normalization
plt.figure(figsize=(8, 6))
plt.semilogy(angles_deg, R_parratt_norm, label='Dynamical (Parratt) - Normalized')
plt.semilogy(angles_deg, R_kinematic_norm, '--', label='Kinematical (Corrected) - Normalized')
plt.xlabel('Angle of incidence [deg]')
plt.ylabel('Normalized Reflectivity')
plt.title('Normalized Parratt vs Kinematical Reflectivity')
plt.legend()
plt.grid(True)
plt.show()