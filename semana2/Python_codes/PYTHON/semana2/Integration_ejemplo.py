import Integration as integ
import numpy as np

# Define tu función e intervalo
f = lambda x: x**2
a, b = 0, 1
n = 1000
N = 100000

# Llama los métodos del módulo
print(f"Trapecio:   {integ.trapecio(f, a, b, n):.6f}")
print(f"Simpson:    {integ.simpson(f, a, b, n):.6f}")
print(f"Montecarlo: {integ.montecarlo(f, a, b, N):.6f}")
print(f"Exacto:     {1/3:.6f}")  # integral exacta de x^2 entre 0 y 1
