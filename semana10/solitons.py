import numpy as np
import matplotlib.pyplot as plt 

N = 200
dx = 0.4
dt = 0.1
eps = 0.2
mu = 0.1

u = np.zeros(N)
u_old = np.zeros(N)
u_new = np.zeros(N)

x = np.arange(N) * dx
u = 0.5 * (1 - np.tanh(x/5 - 5))
u_old = u.copy()

# evaluacion temporal
for n in range(200):
    for i in range(2, N-2):
        nonlinear = (u[i+1] + u[i] + u[i-1]) * (u[i] - u[i-1])
        dispersion = (u[i+2] + 2*u[i-1] - 2*u[i+1] - u[i-2])

        u_new[i] = (u_old[i] - eps*dt/(3*dx) * nonlinear - mu*dt/(dx**3)*dispersion)

    u_old = u.copy()
    u = u_new.copy()

# visualizacion
plt.plot(x, u)
plt.xlabel('x')
plt.ylabel('u(x, t)')
plt.title('Solitons')
plt.show()