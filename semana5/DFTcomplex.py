## Algoritmo de DFT
# Suma total de todas las coincidencias entre tu señal y(x), x = tiempo
# y una onda de referencia de frecuencia k
import numpy as np
from cmath import exp, pi

# Definir constantes
N = 100  # number of points

# Crear señal de ejemplo
x = np.linspace(0, 2*np.pi, N, endpoint=False)
y = 30*np.cos(x) + 60*np.sin(2*x) + 120*np.sin(3*x)

def Signal(y):
    """Compute DFT and return complex Y"""
    N = len(y)
    Y = np.zeros(N, complex)
    # n representa la frecuencia y k representa el tiempo
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
print(f"Amplitud en n=1 (30*cos): {abs(Y_result[1]) / (N/2):.1f}")
print(f"Amplitud en n=2 (60*sin): {abs(Y_result[2]) / (N/2):.1f}")
print(f"Amplitud en n=3 (120*sin): {abs(Y_result[3]) / (N/2):.1f}")
print("First few coefficients:\n", Y_result[:3])
