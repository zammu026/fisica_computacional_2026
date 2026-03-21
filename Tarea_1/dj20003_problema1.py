"""
Problema 1 - Polinomios de Legendre via formula de Rodrigues
Metodo: diferencias centradas para derivadas numericas
"""
import numpy as np
import matplotlib.pyplot as plt
from math import factorial

# Primero trabajamos la derivada n-esima por diferencias centradas
def deriv_central(f, x, n, h):
    """Derivada n-esima de f en x usando diferencias centradas (orden h^2)."""
    if n == 0:
        return f(x)
    # Aplica diferencias centradas recursivamente usando la formula de coeficientes
    # D^n f(x) ≈ sum_{k=0}^{n} (-1)^k C(n,k) f(x + (n/2 - k)*h) / h^n
    coef = np.array([(-1)**k * factorial(n) / (factorial(k) * factorial(n - k))
                     for k in range(n + 1)])
    shifts = np.array([n / 2 - k for k in range(n + 1)])
    return sum(c * f(x + s * h) for c, s in zip(coef, shifts)) / h**n


def legendre_rodrigues(n, x, h):
    """P_n(x) via formula de Rodrigues con diferencias centradas."""
    f = lambda t: (t**2 - 1)**n
    return deriv_central(f, x, n, h) / (2**n * factorial(n))


# Coeficiente principal 
def an_analitico(n):
    return factorial(2 * n) / (2**n * factorial(n)**2)

print("Coeficientes principales a_n:")
for n in range(1, 9):
    print(f"  a_{n} = {an_analitico(n):.6f}")

# Polinomios de Legendre 
from numpy.polynomial.legendre import legval

def legendre_analitico(n, x):
    coefs = [0] * n + [1]          # polinomio L_n de numpy
    return legval(x, coefs)

# Grafica comparativa h=0.01 vs h=1e-3
x = np.linspace(-1, 1, 300)

for h, label in [(0.01, "h = 0.01"), (1e-3, "h = 1e-3")]:
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    fig.suptitle(f"Polinomios de Legendre — Rodrigues + dif. centradas ({label})", fontsize=13)

    for n, ax in enumerate(axes.flat, start=1):
        P_num = np.array([legendre_rodrigues(n, xi, h) for xi in x])
        P_ana = legendre_analitico(n, x)

        ax.plot(x, P_ana, "k-",  lw=2,   label="Analítico")
        ax.plot(x, P_num, "r--", lw=1.5, label="Numérico")
        ax.set_title(f"$P_{n}(x)$")
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1.2, 1.2)
        ax.axhline(0, color="gray", lw=0.5)
        ax.legend(fontsize=7)

    plt.tight_layout()
    plt.show()
