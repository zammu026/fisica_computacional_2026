## Algoritmo de DFT
import numpy as np
from cmath import exp, pi

# Definir constantes
N = 100  # number of points

# Crear señal de ejemplo
x = np.linspace(0, 2*np.pi, N)
y = 30*np.cos(x) + 60*np.sin(2*x) + 120*np.sin(3*x)

# Array para la transformada
Ycomplex = np.zeros(N, complex)

def Signal(y):
    """Compute DFT and return complex Y"""
    N = len(y)
    Y = np.zeros(N, complex)
    for n in range(N):
        for k in range(N):
            Y[n] += y[k] * exp(-1j * 2 * pi * k * n / N)
    return Y

def DFT(Ycomplex):
    """Placeholder for DFT function - calls Signal function"""
    return Signal(y)

# Call functions
Y_result = Signal(y)
print("DFT computed successfully")
print("First few coefficients:", Y_result[:5])
