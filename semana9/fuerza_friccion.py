# Consideracion de la fuerza de friccion en el movimiento de cuerdas
# REVISAR ESTE CODIGO
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as FuncAnimation

L = 1.0
c = 1.0
Nx = 200
dx = L / (Nx - 1)

dt = 0.004
Nt = 800

gamma = 1.0 # friccion

x = np.linspace(0, L, Nx)
y = np.zeros((Nt, Nx))

y[1, :] = np.exp(-100 * (x - 0.5)**2)
y[0, :] = y[1, :]

r = (c * dt / dx)**2

for j in range(1, Nt-1):
    y[j + 1, 1: -1] = ((2 - gamma * dt) * y[j, 1: -1] - (1 - gamma * dt) * y[j - 1, 1: -1] + r * (y[j, 2:] - 2 * y[j, 1: -1] + y[j, 0: -2]))

# ANIMATION

fig, ax = plt.subplots()
line, = ax.plot([], [], lw=2)
ax.set_ylim(-1.2, 1.2)

def update(frame):
    line.set_ydata(y[frame, :])
    ax.set_title(f"t = {frame * dt:.2f}")
    return line,

ani = FuncAnimation(fig, update, frames = Nt, interval = 20)
plt.show()