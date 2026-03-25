import numpy as np
from scipy.integrate import quad

def integral_infinita(f):
    """∫₀^∞ f(x) dx — Ec. 5.68, Newman book"""
    def integrando(z):
        x = z / (1 - z)
        return f(x) / (1 - z)**2
    return quad(integrando, 0, 1)

def integral_desde_a(f, a):
    """∫ₐ^∞ f(x) dx — Ec. 5.71, Newman book"""
    def integrando(z):
        x = z / (1 - z) + a
        return f(x) / (1 - z)**2
    return quad(integrando, 0, 1)

def integral_total(f):
    """∫₋∞^∞ f(x) dx usando x = tan(z) — Ec. 5.75, Newman book"""
    def integrando(z):
        return f(np.tan(z)) / np.cos(z)**2
    return quad(integrando, -np.pi/2, np.pi/2)

# ∫₀^∞ e^(-x) dx = 1

val, err = integral_infinita(lambda x: np.exp(-x))
print(f"∫₀^∞ e^(-x) dx        = {val:.6f}  (error ≈ {err:.2e})(esperado: 1.000000)")
# METODOS DE INTEGRACION NUMERICA:
# Trapecio, Simpson, Romberg, Gauss-Legendre and Monte Carlo.

# ∫₂^∞ e^(-x) dx = e^(-2)
val, err = integral_desde_a(lambda x: np.exp(-x), a=2)
print(f"∫₂^∞ e^(-x) dx        = {val:.6f}  (error ≈ {err:.2e})(esperado: {np.exp(-2):.6f})")

# ∫₋∞^∞ e^(-x²) dx = √π
val, err = integral_total(lambda x: np.exp(-x**2))
print(f"∫₋∞^∞ e^(-x²) dx      = {val:.6f}  (error ≈ {err:.2e})(esperado: {np.sqrt(np.pi):.6f})")