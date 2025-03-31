import numpy as np
import matplotlib.pyplot as plt
import math

# Parameters including azimuthal rotation and incident geometry
params = {
    'w': 120, 'h': 70, 'SWA': np.radians(90), 'pitch': 220,
    't_layers': [70, 500],
    'rho': [0.4, 0.7, 0.9],
    'lambda': 0.124,
    'alpha_i': np.radians(0.2),
    'phi': np.radians(0),  # Azimuthal rotation angle in degrees converted to radians
    'L': 200000,  # Characteristic length in y-direction
    'm': 801  # Single diffraction order
}

# Compute rotated wavevectors
k0 = 2 * np.pi / params['lambda']
kix = k0 * np.cos(params['alpha_i']) * np.sin(params['phi'])
iki_y = k0 * np.cos(params['alpha_i']) * np.cos(params['phi'])
kiz = -k0 * np.sin(params['alpha_i'])
ki = np.array([kix, iki_y, kiz])

# Reciprocal space grid
qx = np.linspace(-0.3, 0.3, 500)
qz = np.linspace(-0.01, 0.6, 500)
QX, QZ = np.meshgrid(qx, qz)

# Solve for QY from Ewald condition
kf_x = kix + QX
kf_z = kiz + QZ
arg = np.maximum(k0**2 - (kf_x**2 + kf_z**2), 0.0)
kf_y = np.sqrt(arg)
QY = kf_y - iki_y

# Compute parallel wavevector component along rotated pitch direction
q_par = QX * np.cos(params['phi']) + QY * np.sin(params['phi'])

# Define missing functions

def trapezoid_FT(qx, qz, w, h, SWA):
    z = np.linspace(0, h, 100)[:, None, None]  # Ensure broadcasting compatibility
    dz = z[1] - z[0]
    integral = np.sum(
        2 * np.sinc(qx * (w / 2 + (h - z) / np.tan(SWA)) / np.pi) * np.exp(1j * qz * z) * dz,
        axis=0
    )
    return integral

def multilayer_FT(qz, interfaces, rho):
    FT = np.zeros_like(qz, dtype=complex)
    for j in range(len(rho)-1):
        delta_rho = rho[j] - rho[j+1]
        FT += delta_rho * (np.exp(1j * qz * interfaces[j+1]) - np.exp(1j * qz * interfaces[j])) / (1j * qz + 1e-12)
    return FT

def structure_factor(qx, pitch, m):
    qx_m = 2 * np.pi * m / pitch
    S_qx = np.sinc((qx - qx_m) * pitch / (2 * np.pi))**2
    return S_qx

# Compute discrete diffraction orders
def discrete_diffraction_orders(pitch, lambda_m, alpha_i_deg, phi_deg, n_min, n_max):
    alpha_i = np.radians(alpha_i_deg)
    phi = np.radians(phi_deg)
    n_values = range(n_min, n_max+1)
    diffraction_orders = []
    
    for n in n_values:
        r = n * (lambda_m / pitch)
        qx = k0 * np.cos(phi) * r
        inside = (np.sin(alpha_i))**2 - r**2 - 2.0 * np.sin(phi) * np.cos(alpha_i) * r
        if inside < 0:
            continue  # No real solution
        qz = k0 * (np.sin(alpha_i) + np.sqrt(inside))
        diffraction_orders.append((qx, qz, n))
    
    return diffraction_orders

orders = discrete_diffraction_orders(params['pitch'], params['lambda'], math.degrees(params['alpha_i']), math.degrees(params['phi']), -3, 3)

# Compute DWBA terms
def DWBA(qx, qy, kfz, kiz, params):
    interfaces = np.array([0, params['h'], params['h']+params['t_layers'][0], params['h']+sum(params['t_layers'])])
    rho = params['rho']

    qz_values = [kfz - kiz, kfz + kiz, -kfz - kiz, -kfz + kiz]
    FT_trap = [trapezoid_FT(qx, qz, params['w'], params['h'], params['SWA']) for qz in qz_values]
    FT_multi = [multilayer_FT(qz, interfaces, rho) for qz in qz_values]
    S_qpar = structure_factor(qx, params['pitch'], params['m'])
    y_term = np.sinc(qy * params['L'] / 2)

    TT = FT_trap[0] * FT_multi[0] * y_term * S_qpar
    RT = FT_trap[1] * FT_multi[1] * y_term * S_qpar
    TR = FT_trap[2] * FT_multi[2] * y_term * S_qpar
    RR = FT_trap[3] * FT_multi[3] * y_term * S_qpar

    return TT, RT, TR, RR

TT, RT, TR, RR = DWBA(QX, QY, QZ, kiz, params)
Intensity = np.abs(TT + RT + TR + RR)**2
Intensity /= np.max(Intensity)

# Visualization
plt.figure(figsize=(8, 6))
plt.imshow(np.log10(Intensity + 1e-9), extent=[qx.min(), qx.max(), qz.min(), qz.max()],
           origin='lower', aspect='auto', cmap='inferno')
plt.title(f' w={params['w']}, h={params['h']}, p={params['pitch']}, SWA={math.degrees(params['SWA']):.2f}, Azimuth={math.degrees(params['phi'])}, alpha_i={math.degrees(params['alpha_i'])}')
plt.xlabel('q_x (nm⁻¹)')
plt.ylabel('q_z (nm⁻¹)')
plt.colorbar(label='Log(Intensity)')

# Plot discrete diffraction orders
for qx_val, qz_val, nval in orders:
    plt.scatter(qx_val, qz_val, color='cyan', label=f'n={nval}' if nval == -3 else "")
    plt.text(qx_val, qz_val, f'n={nval}', fontsize=8, ha='center', va='bottom', color='cyan')

plt.legend()
plt.show()
