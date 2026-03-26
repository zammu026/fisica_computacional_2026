import math
from scipy.constants import h, c, k, Wien

# PARÁMETROS: x0 DEBE ser cercano a 5 para Wien
x0 = 5.0; dx = 3.e-4; eps = 1e-6; Nmax = 100

def f(x):
    # ESTA ES LA FUNCIÓN CORRECTA PARA WIEN
    return 5 * math.exp(-x) + x - 5

def NewtonR(x, dx, eps, Nmax):
    for it in range(Nmax + 1):
        F = f(x)
        if abs(F) <= eps:
            return x
        
        # Derivada numérica
        df = (f(x + dx/2) - f(x - dx/2)) / dx
        
        # Salto de Newton (¡No sobrescribas dx!)
        salto = -F / df
        x += salto
    return x

# 1. Resolver para x (Debe dar ~4.965)
x_sol = NewtonR(x0, dx, eps, Nmax)

# 2. Constante b = (h*c)/(x*k)
b_calc = (h * c) / (x_sol * k)

# 3. Temperatura Sol (lambda ~ 500nm)
T_sol = b_calc / 500e-9

print(f"Raíz hallada (x): {x_sol:.6f}") # Debe ser 4.965114
print(f"Constante b:      {b_calc:.6e}") # Debe ser 2.897e-3
print(f"Temperatura Sol:  {T_sol:.2f} K") # Debe ser ~5795 K