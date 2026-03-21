"""
Problema 3 - Ecuaciones de Lorenz con Runge-Kutta de orden 2 (metodo del punto medio)
"""
import numpy as np
import matplotlib.pyplot as plt

# Parametros de Lorenz
sigma, r, b = 10.0, 28.0, 8.0 / 3.0

def lorenz(t, state):
    x, y, z = state
    return np.array([
        sigma * (y - x),
        r * x - y - x * z,
        x * y - b * z,
    ])

# RK2 (metodo del punto medio)
def rk2(f, t0, tf, y0, h):
    t = np.arange(t0, tf + h, h)
    y = np.zeros((len(t), len(y0)))
    y[0] = y0
    for i in range(len(t) - 1):
        k1    = f(t[i], y[i])
        k2    = f(t[i] + h / 2, y[i] + h / 2 * k1)
        y[i+1] = y[i] + h * k2
    return t, y

# Integracion
t, sol = rk2(lorenz, 0, 50, [1.0, 0.0, 0.0], h=0.01)
x, y, z = sol[:, 0], sol[:, 1], sol[:, 2]

# Graficas (a) y(t) vs t
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(t, y, lw=0.5, color="steelblue")
ax.set_xlabel("t")
ax.set_ylabel("y(t)")
ax.set_title("Lorenz — componente y(t)  [RK2, h=0.01]")
plt.tight_layout()
plt.show()

# (b) atractor de Lorenz z vs x
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(x, z, lw=0.3, color="darkorange", alpha=0.8)
ax.set_xlabel("x")
ax.set_ylabel("z")
ax.set_title("Atractor de Lorenz — z vs x")
plt.tight_layout()
plt.show()

