import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

N = 250
dx = 0.4
dt = 0.05
eps = 0.2
mu = 0.1
Nt = 200

x = np.arange(N) * dx
t_vals = np.arange(Nt) * dt

# condicion inicial = ruido
np.random.seed(0)
u = 0.2*np.random.rand(N)

# suavizado
u = (np.rol(u,1) + u + np.rol(u,-1)) / 3

u_old = u.copy()
u_new = np.zeros_like(u)

# almacenamiento
U = np.zeros((N, Nt))

# evolucion
for n in range(Nt):
    for i in range(2, N-2):
        nonlinear = (u[i+1] + u[i] + u[i-1]) * (u[i] - u[i-1])
        dispersion = (u[i+2] + 2*u[i-1] - 2*u[i+1] - u[i-2])

        u_new[i] = (u_old[i] - eps*dt/(3*dx) * nonlinear - mu*dt/(dx**3)*dispersion)

    u_new[0:2] = 0
    u_new[-2:] = 0

    u_old = u.copy()
    u = u_new.copy()
    U[n, :] = u

# malla
X, T = np.meshgrid(x, t_vals)

# graficas
fig = plt.figure()
