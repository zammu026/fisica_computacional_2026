import numpy as np
import matplotlib.pyplot as plt

N = 71
dx =  1.0 / N
dt = 0.001
c = 1.0

r = (c * dt/dx) ** 2

u = np.zeros((N, N))
u_prev = np.zeros((N, N))
u_next = np.zeros((N, N))

# inicializacion
for i in range(N):
    for j in range(N):
        x = i * dx
        y = j * dx
        u[i, j] = np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)
u_prev = u.copy()

# evolucion temporal
for step in range(200):
    for i in range(1, N-1): # por que crea una matriz 1,N-1 y no 1,N
        for j in range(1, N-1):
            u_next[i, j] = 2 * u[i, j] - u_prev[i, j] + r * (u[i+1, j] + u[i-1, j] + u[i, j+1] + u[i, j-1] - 4 * u[i, j])
    u_prev = u.copy()
    u = u_next.copy()

# visualizacion
X, Y = np.meshgrid(range(N), range(N))

plt.imshow(u, cmap = 'viridis')
plt.colorbar()
plt.title('Membrana vibrante')
plt.xlabel('x')
plt.ylabel('y')
plt.show()