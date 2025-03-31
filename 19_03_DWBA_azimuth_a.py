import numpy as np
import matplotlib.pyplot as plt
import math

# Parameters including azimuthal rotation and incident geometry
params = {
    'w': 120, 'h': 70, 'SWA': np.radians(90), 'pitch': 220,
    't_layers': [70, 500],
    'rho': [0.4, 0.7, 0.9], 
    'lambda': 0.124,
    'alpha_i': np.radians(0.12),
    'phi': np.radians(-1.0),  # Azimuthal rotation angle in degrees converted to radians
    'L': 200000,  # Characteristic length in y-direction
    'm': 801  # no of lines covered by the beam 
}

# Compute rotated wavevectors
k0 = 2 * np.pi / params['lambda']
kix = k0 * np.cos(params['alpha_i']) * np.sin(params['phi'])
kiy = k0 * np.cos(params['alpha_i']) * np.cos(params['phi'])
kiz = -k0 * np.sin(params['alpha_i'])
ki = np.array([kix, kiy, kiz])

# Reciprocal space grid
qx = np.linspace(-0.3, 0.3, 500)
qz = np.linspace(-0.01, 0.6, 500)
QX, QZ = np.meshgrid(qx, qz)

# Solve for QY from Ewald condition
kfx = kix + QX
kfz = kiz + QZ
arg = np.maximum(k0**2 - (kfx**2 + kfz**2), 0.0)
kfy = np.sqrt(arg)
QY = kfy - kiy

# Compute parallel wavevector component along rotated pitch direction
q_par = QX * np.cos(params['phi']) + QY * np.sin(params['phi'])

# Fourier transform (trapezoid shape function) with optimized broadcasting
def trapezoid_FT(qx, qz, w, h, SWA):
    z = np.linspace(0, h, 100)[:, None, None]  # Ensure broadcasting compatibility
    dz = z[1] - z[0]
    integral = np.sum(
        2 * np.sinc(qx * (w / 2 + (h - z) / np.tan(SWA)) / np.pi) * np.exp(1j * qz * z) * dz,
        axis=0
    )
    return integral

# Multilayer Factor including reflections at each interface
def multilayer_FT(qz, interfaces, rho):
    FT = np.zeros_like(qz, dtype=complex)
    for j in range(len(rho)-1):
        delta_rho = rho[j] - rho[j+1]
        FT += delta_rho * (np.exp(1j * qz * interfaces[j+1]) - np.exp(1j * qz * interfaces[j])) / (1j * qz + 1e-12)
    return FT

# Structure Factor for periodicity along rotated pitch direction
def structure_factor(q_par, pitch, m):
    qx_m = 2 * np.pi * m / pitch
    S_qx = np.sinc((q_par - qx_m) * pitch / (2 * np.pi))**2
    return S_qx


# Fresnel Reflection and Transmission Coefficients
def fresnel(qz, rho, wavelength):
    r_e = 2.82e-15  # Classical electron radius in meters
    delta = r_e * np.array(rho) * wavelength**2
    qc = np.sqrt(2 * delta[-1]) * (2 * np.pi / wavelength)
    R = (qz - np.sqrt(qz**2 - qc**2 + 0j)) / (qz + np.sqrt(qz**2 - qc**2 + 0j))
    T = 1 + R
    return R, T

# Compute DWBA terms (TT, RT, TR, RR)
def DWBA(qx, qy, kfz, kiz, params):
    interfaces = np.array([0, params['h'], params['h']+params['t_layers'][0], params['h']+sum(params['t_layers'])])
    rho = params['rho']

    qz_values = [kfz - kiz, kfz + kiz, -kfz - kiz, -kfz + kiz]
    FT_trap = [trapezoid_FT(qx, qz, params['w'], params['h'], params['SWA']) for qz in qz_values]
    FT_multi = [multilayer_FT(qz, interfaces, rho) for qz in qz_values]
    S_qx = structure_factor(qx, params['pitch'], params['m'])
    y_term = np.sinc(qy * params['L'] / 2)
    R, T = fresnel(qz_values[0], rho, params['lambda'])

    TT = T*T*(FT_trap[0] * FT_multi[0] * y_term * S_qx)
    RT = R*T*(FT_trap[1] * FT_multi[1] * y_term * S_qx)
    TR = T*R*(FT_trap[2] * FT_multi[2] * y_term * S_qx)
    RR = R*R*(FT_trap[3] * FT_multi[3] * y_term * S_qx)

    return TT, RT, TR, RR

# Compute DWBA terms
TT, RT, TR, RR = DWBA(QX, QY, QZ, kiz, params)
Intensity = np.abs(TT + RT + TR + RR)**2

# Normalize intensity
Intensity /= np.max(Intensity)

# Visualization
plt.figure(figsize=(8, 6))
plt.imshow(np.log10(Intensity + 1e-9), extent=[qx.min(), qx.max(), qz.min(), qz.max()],
           origin='lower', aspect='auto', cmap='inferno')
plt.xlabel('q_x (nm⁻¹)')
plt.ylabel('q_z (nm⁻¹)')
plt.title(f"w={params['w']}, h={params['h']}, p={params['pitch']}, SWA={math.degrees(params['SWA']):.2f}, Azimuth={math.degrees(params['phi'])}, alpha_i={math.degrees(params['alpha_i'])}")

plt.colorbar(label='Log(Intensity)')
plt.show()
