import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# parametros
N = 201
L = 14.0
dx = L/N
dt = dx/np.sqrt(2)
steps_to_plot = [0, 5, 10, 20]

# grillas
x = np.linspace(-L/2, L/2, N)
y = np.linspace(-L/2, L/2, N)
X, Y = np.meshgrid(x, y)
# variables
u = np.zeros((N, N)) # campo actual
u_prev = np.zeros((N, N)) # paso anterior
u_next = np.zeros((N, N)) # siguiente paso

# condicion inicial (ring soliton)
R = np.sqrt(X**2 + Y**2)
u = 4 * np.arctan(np.exp(3 - R))
u_prev = np.copy(u)

# funcion laplaciano
def laplacian(u):
    return (
        np.roll(u, 1, axis = 0) +
        np.roll(u, -1, axis = 0) +
        np.roll(u, 1, axis = 1) +
        np.roll(u, -1, axis = 1)
        - 4*u
    ) / dx**2

# evoluvion temporal
snapshots = []

for t in range(max(steps_to_plot) + 1):
    # guardar snapshots
    if t in steps_to_plot:
        snapshots.append(np.sin(u/2))
#leapfrog
u_next = (
    2*u - u_prev + dt**2 * (laplacian(u) - np.sin(u))
)

# condiciones de frontera
u_next[0, :] = u_next[1, :]
u_next[-1, :] = u_next[-2, :]
u_next[:, 0] = u_next[:, 1]
u_next[:, -1] = u_next[:, -2]

#avanzar
u_prev = np.copy(u)
u = np.copy(u_next)

# graficas 3D
fig = plt.figure(figsize=(16, 4))
for i, snap in enumerate(snapshots):
    ax = fig.add_subplot(1, len(snapshots), i+1, projection= '3d')
    ax.plot_surface(X, Y, snap, rstride = 4, cstride = 4)
    ax.set_title(f't = {steps_to_plot[i]}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('sin (u/2)')
plt.tight_layout()
plt.show()