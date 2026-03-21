"""
Problema 2 - Energia de gas de fotones con densidad de estados modificada
Metodos: Trapecio, Simpson, Monte Carlo, scipy.quad
"""
import numpy as np
from scipy import integrate, constants
import matplotlib.pyplot as plt

# Constantes fisicas
hbar = constants.hbar
kB   = constants.k
A    = 1.0
wc   = 5e13          # frecuencia de corte [rad/s]

# Integrando
def integrando(w, T):
    beta  = 1.0 / (kB * T)
    g     = A * w**2 * np.exp(-w / wc)
    bose  = np.expm1(beta * hbar * w)        # e^x - 1 = expm1(x), para evitar 0 con x small (x = 1e-100, por ejemplo)
    return hbar * w / bose * g

# Limite superior efectivo (donde el integrando es ~0)
w_max = 50 * wc

# Metodos numericos
def trapecio(f, a, b, N=10_000):
    w = np.linspace(a, b, N)
    return np.trapz(f(w), w)

def simpson(f, a, b, N=10_000):
    N = N if N % 2 == 0 else N + 1
    w = np.linspace(a, b, N + 1)
    return integrate.simpson(f(w), x=w)

def monte_carlo(f, a, b, N=500_000, seed=42):
    rng = np.random.default_rng(seed)
    w   = rng.uniform(a, b, N)
    return (b - a) * np.mean(f(w))

def quad(f, a, b):
    val, _ = integrate.quad(f, a, b, limit=200)
    return val

# Calculo para cada temperatura
temperaturas = [3, 300, 6000]
metodos      = ["Trapecio", "Simpson", "Monte Carlo", "scipy.quad"]
w_min        = 1e6          # evita w=0

resultados = {}
for T in temperaturas:
    f = lambda w, T=T: integrando(w, T)
    vals = [
        trapecio(f, w_min, w_max),
        simpson (f, w_min, w_max),
        monte_carlo(f, w_min, w_max),
        quad(f, w_min, w_max),
    ]
    resultados[T] = vals
    print(f"\nT = {T} K")
    ref = vals[-1]
    for m, v in zip(metodos, vals):
        err = abs(v - ref) / abs(ref) * 100 if ref != 0 else 0
        print(f"  {m:<14}: U = {v:.6e}   err_rel = {err:.4f}%")

# Grafica errores relativos 
fig, ax = plt.subplots(figsize=(9, 5))
x_pos = np.arange(len(temperaturas))
width = 0.25

for i, metodo in enumerate(metodos[:-1]):     # quad es referencia
    errs = []
    for T in temperaturas:
        ref = resultados[T][-1]
        err = abs(resultados[T][i] - ref) / abs(ref) * 100
        errs.append(err)
    ax.bar(x_pos + i * width, errs, width, label=metodo)

ax.set_xticks(x_pos + width)
ax.set_xticklabels([f"T={T} K" for T in temperaturas])
ax.set_ylabel("Error relativo respecto a quad (%)")
ax.set_title("Comparación de métodos numéricos de integración")
ax.legend()
plt.tight_layout()
plt.show()

"""
(d) ¿Cuál método numérico resultó más estable y por qué?
Trapecio y Simpson alcanzan errores menores al 0.05% con 10,000 puntos, suficiente para este integrando suave.
Monte Carlo es el menos preciso (~4% a T=3 K) porque el integrando está concentrado en frecuencias bajas y el
muestreo uniforme desperdicia la mayoría de los puntos en la zona plana. scipy.quad es la referencia por usar
cuadratura adaptativa, pero para integrales con esta estructura los métodos deterministas simples son perfectamente competitivos.
"""
