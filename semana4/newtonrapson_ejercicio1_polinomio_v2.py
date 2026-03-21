from math import cos
import numpy as np
import matplotlib.pyplot as plt

# ── Polinomio P(x) ──────────────────────────────────────────────
def f(x):
    return 924*x**6 - 2772*x**5 + 3150*x**4 - 1680*x**3 + 420*x**2 - 42*x + 1

# ── Newton-Raphson (newtonR.py proporcionado) ────────────────────
def NewtonR(x, dx, eps, Nmax):
    for it in range(0, Nmax + 1):
        F = f(x)
        if abs(F) <= eps:
            return x
        df = (f(x + dx/2) - f(x - dx/2)) / dx
        incremento = -F / df
        x += incremento
    return x

# Parámetros
dx_step = 3.e-4
eps     = 1e-12      # alta precisión → 10 decimales
Nmax    = 1000

# ── a) Gráfica de P(x) en [0, 1] ────────────────────────────────
x_vals = np.linspace(0, 1, 1000)
y_vals = [f(x) for x in x_vals]

plt.figure(figsize=(8, 4))
plt.plot(x_vals, y_vals, 'b-', linewidth=2)
plt.axhline(0, color='k', linewidth=0.8, linestyle='--')
plt.title("P(x) = 6.º polinomio de Legendre (intervalo [0,1])")
plt.xlabel("x")
plt.ylabel("P(x)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
#for r in roots:
#    plt.plot(r, f(r), 'ro', markersize=7)

# ── b) Estimaciones iniciales (inspeccionando la gráfica) ────────
x0_guesses = [0.03, 0.17, 0.38, 0.62, 0.83, 0.97]

print("=" * 55)
print(f"{'#':>2}  {'x0 inicial':>12}  {'Raíz (10 dec)':>18}  {'P(raíz)':>14}")
print("=" * 55)

roots = []
for i, x0 in enumerate(x0_guesses):
    r = NewtonR(x0, dx_step, eps, Nmax)
    roots.append(r)
    print(f"{i+1:>2}  {x0:>12.4f}  {r:>18.10f}  {f(r):>14.2e}")

print("=" * 55)

# Mostrar puntos en la grafica, aqui puse a roots porque alla arriba no habia sido definida
for r in roots:
    plt.plot(r, f(r), 'ro', markersize=7)
plt.show()
