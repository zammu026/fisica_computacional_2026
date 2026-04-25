import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

N = 300
dx = 0.4
dt = 0.05
eps = 0.2
mu = 0.1
Nt = 300

x = np.arange(N) * dx

# condiciones iniciales (pulso del soliton)
u = 0.5 * (1 - np.tanh(x/5 - 5)) # tipo soliton (default)
#u = np.exp(-(x-30)**2/50) # gaussiano
#u = np.exp(-(x-20)**2/40) + 0.5*np.exp(-(x-60)**2/40) # dos solitones
#u = np.zeros_like(x) # rectangular
#u[60:100] = 1

u_old = u.copy()
u_new = np.zeros_like(u)

posiciones = []
tiempos = []

fig, ax = plt.subplots()
line, = ax.plot(x, u)
#ax.set_ylim(-0.2, 10.2)

for  n in range(Nt): 
    for i in range(2, N-2):
        nonlinear = (u[i+1] + u[i] + u[i-1]) * (u[i] - u[i-1])
        dispersion = (u[i+2] + 2*u[i-1] - 2*u[i+1] - u[i-2])

        u_new[i] = (u_old[i] - eps*dt/(3*dx) * nonlinear - mu*dt/(dx**3)*dispersion)

    # actualizar
    u_old = u.copy()
    u = u_new.copy()

    x_max = x[np.argmax(u)]
    posiciones.append(x_max)
    tiempos.append(n * dt)

# ajustes lineales

coef = np.polyfit(tiempos, posiciones, 1)
velocidad = coef[0]
print(f"Velocidad estimada del puslso: {velocidad:.4f} m/s")

plt.figure()
plt.plot(tiempos, posiciones, 'o', label = 'Datos')
plt.plot(tiempos, np.polyval(coef, tiempos), '-', label = 'Ajuste lineal')
plt.xlabel('Tiempo')
plt.ylabel('Posicion del maximo')
plt.title('Medicion de la velocidad del soliton')
plt.legend()
plt.grid()
plt.show()