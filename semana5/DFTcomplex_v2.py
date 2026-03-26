# Cuánto aporta cada "armónico" a la señal total, mira que esta normalizada
from numpy import *
import cmath

N = 100
twopi = 2.0*pi
h = twopi/N
sq2pi = 1.0/sqrt(twopi)

# Create a sample signal
t = linspace(0, 2*pi, N)
y = sin(t) + 0.5*sin(2*t)

def DFT(y):
    """Compute Discrete Fourier Transform"""
    N = len(y)
    Y = zeros(N, complex)
    for n in range(N):
        zsum = complex(0.0, 0.0)
        for k in range(N):
            zsum += y[k] * exp(-1j * twopi * k * n / N)
        Y[n] = zsum/N
    return Y

# Test the function
Y_result = DFT(y)
print("DFT computed successfully")
print("First few coefficients:\n", Y_result[:3])
