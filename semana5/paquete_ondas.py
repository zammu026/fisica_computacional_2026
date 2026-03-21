import numpy as np
import matplotlib.pyplot as plt
from numpy import zeros
from cmath import exp, pi
from DFT import dft

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
plt.title('Distribuicuion de momento')

plt.tight_layaut()
plt.show()
