import numpy as np
import matplotlib.pyplot as plt

def parratt(qz, thickness, delta, beta, wavelength):
    k0 = 2 * np.pi / wavelength
    kz = [np.sqrt((qz/2)**2 - 4*np.pi*(delta[i] - 1j*beta[i])/wavelength**2) for i in range(len(delta))]
    r = np.zeros(len(qz), dtype=complex)

    for i in range(len(qz)):
        rj = 0
        for j in range(len(thickness), 0, -1):
            kzj = kz[j][i]
            kzj_minus = kz[j-1][i]
            exp_factor = np.exp(2j * kzj * thickness[j-1])
            rj = (kzj_minus - kzj)/(kzj_minus + kzj) + rj * exp_factor
            rj /= 1 + ((kzj_minus - kzj)/(kzj_minus + kzj)) * rj * exp_factor
        r[i] = rj

    return np.abs(r)**2

# Example Parameters:
wavelength = 1.54  # Angstrom (Cu-Kα)
qz = np.linspace(0.01, 0.5, 1000)  # 1/Angstrom
thickness = np.array([50])  # Angstrom
delta = np.array([0, 7.6e-6, 7.6e-6])  # vacuum, layer, substrate
beta = np.array([0, 1.8e-8, 1.8e-8])

reflectivity = parratt(qz, thickness, delta, beta, wavelength)

plt.figure()
plt.semilogy(qz, reflectivity, label='Parratt Reflectivity')
plt.xlabel('qz (1/Angstrom)')
plt.ylabel('Reflectivity')
plt.title('Corrected Parratt Algorithm Reflectivity')
plt.legend()
plt.grid(True)
plt.show()
