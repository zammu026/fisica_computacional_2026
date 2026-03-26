#simula un paquete de ondas gaussiano (una partícula libre en mecánica cuántica)
# y calcula su transformada de Fourier para observar cómo se distribuye su momento
import numpy as np
import matplotlib.pyplot as plt
from numpy import zeros
from cmath import exp, pi
from DFT import dft # importar la función dft del archivo dft.py

N = 500
x_min = -20
x_max = 20 

x = np.linspace(x_min, x_max, N)
dx = x[1] - x[0]

sigma0 = 1.0
k0 = 5.0

psi = np.exp(-0.5*((x - 5.0)/sigma0)**2) * np.exp(1j*k0*x)

psi_p = dft(psi)

k = np.arange(len(psi_p))*(2*pi/(N*dx))
# Para visualizar el pico correctamente (centrar k), 
# se suele desplazar la mitad del espectro:
k_shift = k - (np.pi / dx) 
psi_p_shift = np.roll(psi_p, N // 2)

plt.figure(figsize = (10,4))

plt.subplot(1,2,1)
plt.plot(x, np.abs(psi)**2)
plt.xlabel('x')
plt.ylabel(r'$|\psi(x)|^2$')
plt.title('Paquete de ondas en posicion')

plt.subplot(1,2,2)
plt.plot(k, np.abs(psi_p))

plt.xlabel('k')
plt.ylabel(r'$|\psi (k)|$')
plt.title('Distribucion de momento')
plt.tight_layout()
plt.show()

# Usamos las versiones desplazadas para que el pico en k=5 sea visible al centro
plt.plot(k_shift, np.abs(psi_p_shift)) 
plt.xlabel('k')
plt.ylabel(r'$|\psi (k)|$')
plt.title('Distribucion de momento')
plt.tight_layout()
plt.show()

