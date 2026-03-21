import matplotlib.pyplot as plt
import numpy as np

# ── Parámetros ──────────────────────────────────────────────────────
wo = 0.8

# ── Ecuaciones del sistema ──────────────────────────────────────────
# dx/dt = v
# dv/dt = -wo²·x

def f1(x):
    return -wo**2 * x   # dv/dt = f1

def f2(v):
    return v             # dx/dt = v

# ── Intervalo y paso ────────────────────────────────────────────────
a = 0.0
b = 10.0
N = 1000
h = (b - a) / N

# ── Arrays ──────────────────────────────────────────────────────────
t = np.linspace(a, b, N)
x = np.zeros(N)
v = np.zeros(N)

# ── Condiciones iniciales ───────────────────────────────────────────
x[0] = 1.0   # amplitud inicial
v[0] = 0.0   # parte del reposo

# ── Método de Euler ─────────────────────────────────────────────────
for n in range(N - 1):
    x[n+1] = x[n] + h * f2(v[n])
    v[n+1] = v[n] + h * f1(x[n])

# ── Solución analítica ──────────────────────────────────────────────
# Con x(0)=1, v(0)=0:
#   x(t) = x0·cos(wo·t)
#   v(t) = -x0·wo·sin(wo·t)
x0 = x[0]
x_analitica = x0 * np.cos(wo * t)
v_analitica = -x0 * wo * np.sin(wo * t)

# ── Error absoluto ──────────────────────────────────────────────────
error_x = np.abs(x - x_analitica)
error_v = np.abs(v - v_analitica)

# ── Gráficas ────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(10, 12))
fig.suptitle(f"Oscilador armónico (wo={wo}) — Euler vs Analítica", fontsize=14)

# Panel 1: posición
axes[0].plot(t, x_analitica, 'b-',  linewidth=2,   label="x(t) analítica")
axes[0].plot(t, x,           'r--', linewidth=1.5,  label="x(t) Euler")
axes[0].set_ylabel("x(t)")
axes[0].set_title("Posición")
axes[0].legend()
axes[0].grid(True)

# Panel 2: velocidad
axes[1].plot(t, v_analitica, 'b-',  linewidth=2,   label="v(t) analítica")
axes[1].plot(t, v,           'r--', linewidth=1.5,  label="v(t) Euler")
axes[1].set_ylabel("v(t)")
axes[1].set_title("Velocidad")
axes[1].legend()
axes[1].grid(True)

# Panel 3: error absoluto
axes[2].plot(t, error_x, 'g-',  linewidth=1.5, label="|error| x(t)")
axes[2].plot(t, error_v, 'm--', linewidth=1.5, label="|error| v(t)")
axes[2].set_xlabel("t")
axes[2].set_ylabel("Error absoluto")
axes[2].set_title("Error acumulado")
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.show()

# ── Métricas ─────────────────────────────────────────────────────────
print(f"Error máximo en x: {error_x.max():.6f}")
print(f"Error máximo en v: {error_v.max():.6f}")
print(f"Error RMS    en x: {np.sqrt(np.mean(error_x**2)):.6f}")
print(f"Error RMS    en v: {np.sqrt(np.mean(error_v**2)):.6f}")
