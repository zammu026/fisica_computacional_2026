import scipy.constants as cte
import numpy as np

def f(x):
  return 5 * np.exp(-x) + x - 5

def biyeccion(m, n, eps):
  while (n - m) / 2 > eps:
    c = (m + n) / 2
    if f(c) == 0:
      return c
    elif f(m) * f(c) < 0:
      n = c
    else:
      m = c
  return (m + n) / 2

m = 1
n = 10
eps = 1e-6
x = biyeccion(m, n, eps)
b=cte.h*cte.c/(x*cte.k)

print(f"Desplazamiento de Wien: {b:.6e} m K")