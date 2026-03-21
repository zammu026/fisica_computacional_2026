"""
Problema 4 - Tiro parabolico con resistencia de aire (RK4)
Modelo: F_f = -k*m*|v|^n * v/|v|
"""
import numpy as np
import matplotlib.pyplot as plt

# Parametros
g  = 9.8
k  = 0.8
m  = 1.0          # la masa se cancela en la ecuacion de movimiento

# Condiciones iniciales
x0, y0     = 0.0, 0.0
v0x, v0y   = 18.23, 12.3

# Ecuaciones de movimiento
def eom(t, state, n):
    x, y, vx, vy = state
    speed = np.sqrt(vx**2 + vy**2)
    ax = -k * speed**(n - 1) * vx
    ay = -g - k * speed**(n - 1) * vy
    return np.array([vx, vy, ax, ay])

# RK4 
def rk4(f, t0, tf, y0, h, n):
    t = np.arange(t0, tf + h, h)
    y = np.zeros((len(t), len(y0)))
    y[0] = y0
    for i in range(len(t) - 1):
        k1 = f(t[i],           y[i],           n)
        k2 = f(t[i] + h/2,     y[i] + h/2*k1,  n)
        k3 = f(t[i] + h/2,     y[i] + h/2*k2,  n)
        k4 = f(t[i] + h,       y[i] + h*k3,    n)
        y[i+1] = y[i] + h/6 * (k1 + 2*k2 + 2*k3 + k4)
    return t, y

# Integracion para n = 1, 3/2, 2
ns     = [1, 1.5, 2]
labels = ["n=1", "n=3/2", "n=2"]
colors = ["royalblue", "tomato", "green"]
y0_vec = [x0, y0, v0x, v0y]
h      = 0.01

trayectorias = {}
for n in ns:
    t, sol = rk4(eom, 0, 25, y0_vec, h, n)
    trayectorias[n] = (t, sol)

# Solucion analitica sin friccion
t_ana  = np.linspace(0, 25, 2000)
x_ana  = x0 + v0x * t_ana
y_ana  = y0 + v0y * t_ana - 0.5 * g * t_ana**2

# (b) trayectorias y vs x
fig, ax = plt.subplots(figsize=(10, 5))

for n, lbl, col in zip(ns, labels, colors):
    t, sol = trayectorias[n]
    mask   = sol[:, 1] >= 0          # solo mientras y >= 0
    ax.plot(sol[mask, 0], sol[mask, 1], color=col, lw=2, label=lbl)

mask_ana = y_ana >= 0
ax.plot(x_ana[mask_ana], y_ana[mask_ana], "k--", lw=1.5, label="Sin fricción")
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_title("Tiro parabólico con fricción — RK4")
ax.legend()
ax.set_ylim(bottom=0)
plt.tight_layout()
plt.show()

# (c) |v| vs t
fig, ax = plt.subplots(figsize=(9, 4))

for n, lbl, col in zip(ns, labels, colors):
    t, sol = trayectorias[n]
    speed  = np.sqrt(sol[:, 2]**2 + sol[:, 3]**2)
    ax.plot(t, speed, color=col, lw=2, label=lbl)

ax.set_xlabel("t [s]")
ax.set_ylabel("|v| [m/s]")
ax.set_title("Velocidad vs tiempo — distintos modelos de fricción")
ax.legend()
plt.tight_layout()
plt.show()
