import numpy as np
import matplotlib.pyplot as plt
import math

# Parameters including azimuthal rotation and incident geometry
params = {
    'w': 120, 'h': 70, 'SWA': np.radians(90), 'pitch': 220,
    't_layers': [100, 0],
    'rho': [2.0e23, 3.0e23,5.0e23],
    'lambda': 0.124,
    'alpha_i': np.radians(0.2),
    'phi': np.radians(0),  # Azimuthal rotation angle in degrees converted to radians
    'L': 200000,  # Characteristic length in y-direction
    'm': 801  # # no of lines covered by the beam 
}

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

# Structure Factor for periodicity along x
def structure_factor(qx, pitch, m):
    qx_m = 2 * np.pi * m / pitch
    S_qx = np.sinc((qx - qx_m) * pitch / (2 * np.pi))**2
    return S_qx

# Fresnel Reflection and Transmission Coefficients
def fresnel(qz, rho, wavelength):
    r_e = 2.82e-15  # Classical electron radius in meters
    delta = r_e * np.array(rho) * wavelength**2
    qc = np.sqrt(2 * delta[-1]) * (2 * np.pi / wavelength)
    R = (qz - np.sqrt(qz**2 - qc**2 + 0j)) / (qz + np.sqrt(qz**2 - qc**2 + 0j))
    T = 1 + R
    return R, T

# Compute explicit DWBA terms (TT, RT, TR, RR)
def DWBA(qx, qy, kfz, kiz, params):
    interfaces = np.array([0, params['h'], params['h']+params['t_layers'][0], params['h']+sum(params['t_layers'])])
    rho = params['rho']

    qz_values = [kfz - kiz, kfz + kiz, -kfz - kiz, -kfz + kiz]
    FT_trap = [trapezoid_FT(qx, qz, params['w'], params['h'], params['SWA']) for qz in qz_values]
    FT_multi = [multilayer_FT(qz, interfaces, rho) for qz in qz_values]
    S_qx = structure_factor(qx, params['pitch'], params['m'])
    y_term = np.sinc(qy * params['L'] / 2)
    #R, T = fresnel(qz_values[0], rho, params['lambda'])
    R,T=1,2

    TT = T * FT_trap[0] * FT_multi[0] * T * y_term * S_qx
    RT = R * FT_trap[1] * FT_multi[1] * T * y_term * S_qx
    TR = T * FT_trap[2] * FT_multi[2] * R * y_term * S_qx
    RR = R * FT_trap[3] * FT_multi[3] * R * y_term * S_qx

    return TT, RT, TR, RR

# Reciprocal space grid
qx = np.linspace(-0.3, 0.3, 500)

qz = np.linspace(0, 0.6, 500)

QX, QZ = np.meshgrid(qx, qz)

# Incident and scattered wave vectors
k0 = 2 * np.pi / params['lambda']
kiy = k0 * np.cos(params['alpha_i']) 
kiz= -k0 * np.sin(params['alpha_i'])
ki = np.array([0, kiy, kiz])

# Apply azimuth rotation and Ewald sphere intersection explicitly

kf_y = np.sqrt(np.maximum(k0**2 - QX**2 - QZ **2, 0))  # Prevent negative values

qy = kf_y - kiy  # Define qy

# Compute DWBA terms
TT, RT, TR, RR = DWBA(QX, qy, QZ, kiz, params)
Intensity = np.abs(TT + RT + TR + RR)**2

# Normalize intensity
Intensity /= np.max(Intensity)

# Visualization of Intensity map in (qx, qz) space
plt.figure(figsize=(8, 6))
plt.imshow(np.log10(Intensity + 1e-9), extent=[qx.min(), qx.max(), qz.min(), qz.max()],
           origin='lower', aspect='auto', cmap='inferno')
plt.xlabel('q_x (nm⁻¹)')
plt.ylabel('q_z (nm⁻¹)')
plt.title(f' w={params['w']}, h={params['h']},p={params['pitch']},SWA={math.degrees(params['SWA']):.2f},Azimuth={math.degrees(params['phi'])},alpha_i={math.degrees(params['alpha_i'])}')
plt.colorbar(label='Log(Intensity)')
plt.show()


'''# Line cuts along qz
plt.figure(figsize=(8,5))
for qz_val in [0.10, 0.11, 0.12]:
    idx = (np.abs(qz - qz_val)).argmin()
    plt.plot(qx, np.log10(Intensity[idx, :] + 1e-9), label=f'qz = {qz_val:.2f} nm⁻¹')
plt.xlabel('q_x (nm⁻¹)')
plt.ylabel('Log(Intensity)')
plt.title('Intensity Line Cuts at Fixed q_z')
plt.legend()
plt.grid()
plt.show()

# Line cuts along qx
plt.figure(figsize=(8,5))
for qx_val in [-0.1, 0.0, 0.1]:
    idx = (np.abs(qx - qx_val)).argmin()
    plt.plot(qz, np.log10(Intensity[:, idx] + 1e-9), label=f'qx = {qx_val:.2f} nm⁻¹')
plt.xlabel('q_z (nm⁻¹)')
plt.ylabel('Log(Intensity)')
plt.title('Intensity Line Cuts at Fixed q_x')
plt.legend()
plt.grid()
plt.show()'''
