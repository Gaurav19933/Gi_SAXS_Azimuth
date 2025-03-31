import numpy as np
import matplotlib.pyplot as plt

# Constants
wavelength = 1.54e-10  # Cu K-alpha in meters
K = 2 * np.pi / wavelength
r_e = 2.81794e-15  # Classical electron radius (m)

# Corrected Electron densities (electrons/m³)
rho_e = np.array([0, 1.0e30, 7.0e29])  # air, SiO2 film, Si substrate
mu = np.array([0, 3.0e4, 1.4e4])       # corrected absorption coefficients (1/m)

# Calculate delta and beta explicitly
delta = r_e * wavelength**2 * rho_e / (2 * np.pi)
beta = mu * wavelength / (4 * np.pi)
n_layers = 1 - delta + 1j * beta

# Thickness of film layer
thickness = np.array([20e-9])  # film thickness (20 nm)

angles_deg = np.linspace(0.01, 5, 1000)
angles_rad = np.deg2rad(angles_deg)

# Corrected Parratt reflectivity (dynamic model)
def parratt_reflectivity(n, d, theta, wavelength):
    K = 2 * np.pi / wavelength
    kz = K * np.sqrt(n[:, None]**2 - np.cos(theta)**2)
    r = (kz[:-1] - kz[1:]) / (kz[:-1] + kz[1:])
    R = np.zeros(len(theta), dtype=complex)

    for idx in range(len(theta)):
        X = r[-1, idx]
        for j in reversed(range(len(d))):
            exp_factor = np.exp(-2j * kz[j+1, idx] * d[j])
            X = (r[j, idx] + X * exp_factor) / (1 + r[j, idx] * X * exp_factor)
        R[idx] = X
    return np.abs(R)**2

# Corrected Kinematic approximation
def kinematic_reflectivity(n, d, theta, K):
    Qz = 2 * K * np.sin(theta)
    delta_rho = n[:-1]**2 - n[1:]**2
    z_interfaces = np.hstack(([0], np.cumsum(d)))
    R = np.zeros_like(Qz, dtype=complex)

    for j in range(len(d)):
        dz = d[j]
        z = z_interfaces[j]
        R += delta_rho[j] * np.exp(1j * Qz * z) * (1 - np.exp(-1j * Qz * dz)) / (1j * Qz)

    return np.abs(R)**2

# Reflectivity calculations
R_parratt = parratt_reflectivity(n_layers, thickness, angles_rad, wavelength)
R_kinematic = kinematic_reflectivity(n_layers, thickness, angles_rad, K)

# Normalization
R_parratt_norm = R_parratt / np.max(R_parratt)
R_kinematic_norm = R_kinematic / np.max(R_kinematic)

# Critical angle calculation
critical_angle_rad = np.sqrt(2 * delta[1])
critical_angle_deg = np.rad2deg(critical_angle_rad)

# Plotting
plt.figure(figsize=(8, 6))
plt.semilogy(angles_deg, R_parratt_norm, label='Dynamic (Parratt)')
plt.semilogy(angles_deg, R_kinematic_norm, '--', label='Kinematic')
plt.axvline(critical_angle_deg, color='red', linestyle=':', label=f'Critical Angle ≈ {critical_angle_deg:.3f}°')
plt.xlabel('Angle [deg]')
plt.ylabel('Normalized Reflectivity')
plt.title('Reflectivity curves ')
plt.legend()
plt.grid(True)
plt.show()
