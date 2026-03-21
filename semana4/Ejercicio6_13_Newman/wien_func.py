from math import exp

# Constantes físicas (SI)
h  = 6.626e-34   # Planck
c  = 3.000e8     # velocidad de la luz
kB = 1.381e-23   # Boltzmann

def f(x):
    """Ecuación a resolver: 5e^{-x} + x - 5 = 0"""
    return 5 * exp(-x) + x - 5
