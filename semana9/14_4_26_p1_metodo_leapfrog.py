# Metodo de leapfrog
import numpy as np
import matplotlib.pyplot as plt

# parametros fisicos
L = 1.0
c = 1.0

# discretizacion
Nx = 100
dx = L / (Nx - 1)

dt = 0.005 # ajustar para cumplir Courant
Nt = 300
print((dt*c/dx)**2) # condicion de Courant

# estabilidad
courant = c * dt / dx
print(f"Courant: {courant}")

# malla espacial
x = np.linspace(0, L, Nx)

# inicializacion
y = np.zeros((Nt, Nx))

# condicion inicial
y[1, :] = np.exp(-100 * (x - 0.5)**2)

# velocidad incial cer
y[0, :] = y[1, :]

# evolucion temporal (leapfrog)
for  j in range(1, Nt -1):
    for i in range(1, Nx -1):
        y[j + 1, i] = (2*y[j, i] - y[j - 1, i] + (courant**2) * (y[j, i + 1] + y[j, i - 1]- 2*y[j, i] ))
    
    # condiciones de frontera
    y[j + 1, 0] = 0
    y[j + 1, - 1] = 0

# visualizacion
for i in range(0, Nt, 20):
    plt.plot(x, y[i, :], label = f"t = {j * dt:.2f}")

plt.xlabel("x")
plt.ylabel("y (x, t)")
plt.title(f"Evolucion temporal de una cuerda con Metodo de Leapfrog (courant = {courant:.2f})")
plt.legend()
plt.show()