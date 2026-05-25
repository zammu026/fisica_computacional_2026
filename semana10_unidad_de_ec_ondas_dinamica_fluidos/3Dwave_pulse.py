# 3D wave pulse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

N = 71
dx =  1.0 / N
dt = 0.001
c = 1.0

r = (c * dt/dx) ** 2

# condicion de estabilidad 
if r > 0.5:
    raise ValueError('Esquema inestable: reduce dt')

# inicializacion
u = np.zeros((N, N))
u_prev = np.zeros((N, N))
u_next = np.zeros((N, N))

# condicion inicial (pulso)
for i in range(N):
    for j in range(N):
        x = i * dx
        y = j * dx
        u[i, j] = np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)
u_prev = u.copy()

#figura 
fig, ax = plt.subplots()
im = ax.imshow(u, animated=True)
plt.colorbar(im)

# funcion de actualizacion
def update(frame):
    global u, u_prev, u_next
    for i in range(1, N-1): # por que crea una matriz 1,N-1 y no 1,N
        for j in range(1, N-1): 
            u_next[i, j] = 2 * u[i, j] - u_prev[i, j] + r * (u[i+1, j] + u[i-1, j] + u[i, j+1] + u[i, j-1] - 4 * u[i, j])

# Bordes fijos
    u_next[0, :] = 0
    u_next[-1, :] = 0
    u_next[:, 0] = 0
    u_next[:, -1] = 0

    u_prev = u.copy()
    u = u_next.copy()
    im.set_array(u)
    return [im]

ani = FuncAnimation(fig, update, interval=1, blit=True)
plt.title('Membrana vibrante')
plt.xlabel('x')
plt.ylabel('y') 
plt.show()
