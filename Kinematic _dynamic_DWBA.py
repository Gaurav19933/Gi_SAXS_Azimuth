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

# Parratt reflectivity (dynamic model)
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

# Kinematic approximation
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

# Fully Rigorous DWBA implementation (explicit integration)
def dwba_reflectivity(n, d, theta, wavelength):
    K = 2 * np.pi / wavelength
    kz = K * np.sqrt(n[:, None]**2 - np.cos(theta)**2)
    r = (kz[:-1] - kz[1:]) / (kz[:-1] + kz[1:])

    delta_rho = n[1]**2 - n[0]**2
    R_dwba = np.zeros(len(theta), dtype=complex)

    for idx in range(len(theta)):
        integral = 0
        z_points = np.linspace(0, d[0], 100)
        dz = z_points[1] - z_points[0]
        for z in z_points:
            psi_inc = np.exp(-1j * kz[0, idx] * z) + r[0, idx] * np.exp(1j * kz[0, idx] * z)
            psi_scat = np.exp(1j * kz[1, idx] * (d[0] - z))
            integral += psi_inc * delta_rho * psi_scat * dz
        R_dwba[idx] = integral

    return np.abs(R_dwba)**2

# Reflectivity calculations
R_parratt = parratt_reflectivity(n_layers, thickness, angles_rad, wavelength)
R_kinematic = kinematic_reflectivity(n_layers, thickness, angles_rad, K)
R_dwba = dwba_reflectivity(n_layers, thickness, angles_rad, wavelength)

# Normalization
R_parratt_norm = R_parratt / np.max(R_parratt)
R_kinematic_norm = R_kinematic / np.max(R_kinematic)
R_dwba_norm = R_dwba / np.max(R_dwba)

# Critical angle calculation
critical_angle_rad = np.sqrt(2 * delta[1])
critical_angle_deg = np.rad2deg(critical_angle_rad)

# Plotting
plt.figure(figsize=(8, 6))
plt.semilogy(angles_deg, R_parratt_norm, label='Dynamic (Parratt)')
plt.semilogy(angles_deg, R_kinematic_norm, '--', label='Kinematic')
plt.semilogy(angles_deg, R_dwba_norm, '-.', label='Fully Rigorous DWBA')
plt.axvline(critical_angle_deg, color='red', linestyle=':', label=f'Critical Angle ≈ {critical_angle_deg:.3f}°')
plt.xlabel('Angle [deg]')
plt.ylabel('Normalized Reflectivity')
plt.title('Fully Rigorous DWBA with Dynamic and Kinematic Models')
plt.legend()
plt.grid(True)
plt.show()
