# ecuacion de transporte
import numpy as np
import matplotlib.pyplot as plt

N = 200
L = 1.0
dx = L / N
dt = 0.005
c = 1

u = np.zeros(N)
u_old = np.zeros(N)
u_new = np.zeros(N)

# condiciones iniciales
x = np.linspace(0, L, N)
# pulso gausiano
u = np.exp(-100 * (x - 0.5)**2)
u_old = u.copy()

# evolucion temporal
for n in range(300):
    for i in range(1, N-1):
        u_new[i] = u_old[i] - (c*dt/dx) * (u[i+1] - u[i-1])
    u_old = u.copy()
    u = u_new.copy()

# visualizacion
plt.plot(x, u)
plt.title('Adveccion')
plt.xlabel('x')
plt.ylabel('u(x, t)')
plt.show()