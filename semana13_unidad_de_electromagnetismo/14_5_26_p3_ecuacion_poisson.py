# Placas del capacitor que no estan conectadas a una bateria.
r'''
ENUNCIADO:
Placas de un capacitor que no estan conectadas a una bateria el cual
posee una densidad de carga uniforme en la placa superior +rho y su placa inferior -rho.

RESOLVER:
El problema consiste en resolver delta^2 V = -rho/epsilon_0, en las placas y
delta^2 V = 0 fuera de ellas (en la caja).
'''

r'''
Recordando que al aplicar diferencias infinitas a la ecuacion de Poisson quedaria:
(V_{i+1, j} + V_{i-1, j} + V_{i, j+1} + V_{i, j-1} - 4V_{i, j})/(delta^2) = -rho/epsilon_0

despejando a V_{i, j}:
1/4 (V_{i+1, j} + V_{i-1, j} + V_{i, j+1} + V_{i, j-1} + delta^2 rho/epsilon_0) = V_{i, j}

La densidad de carga se modela numericamente mediante un arreglo rho[i, j]
La placa se representa como regiones cargadas rho > 0 para la placa superior y
rho < 0 para la placa inferior y fuera de las placas rho = 0
'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parametros
N = 100
delta = 1.0

epsilon0 = 1.0

rho0 =  1.0

tolerance = 1e-4
max_iterations = 10000

V = np.zeros((N, N))
rho = np.zeros((N, N))

# definicion de placas cargadas
# placa superior de +100 V
rho[30, 30:70] = rho0

# placa inferior de de -100 V
rho[60, 30:70] = -rho0

# codigo de gauss seidel
for iteration in range(max_iterations):
    Vold = V.copy()
    for i in range(1, N-1):
        for j in range(1, N-1):
            V[i, j] = 0.25*(
                V[i+1, j] +
                V[i-1, j] +
                V[i, j+1] +
                V[i, j-1] +
                delta**2*rho[i, j]/epsilon0 # agregado
            )

    # error maximo
    error =np.max(np.abs(V - Vold))
    if iteration % 100 == 0:
        print("Iteration: ", iteration, "Error: ", error)
    if error < tolerance:
        print("\nConvergencia alcanzada")
        break
    


# coordenadas
x = np.arange(0, N, 1)
y = np.arange(0, N, 1)
X, Y = np.meshgrid(x, y)


# Graficación 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection = "3d")

surface = ax.plot_surface(
    X, Y, V,
    cmap = "coolwarm",
    edgecolor = "k",
    linewidth = 0.2
)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("V(x, y)")

ax.set_title("Potencial electrico - Ecuacion de Poisson")

plt.show()

# contornos equipotenciales
plt.figure(figsize=(7, 6))
cont = plt.contourf(X, Y, V, 40, cmap = "coolwarm")
plt.colorbar(label = "Potencial (V)")
plt.contour(X, Y, V, 20, color = "black")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Contornos equipotenciales")

plt.show()

# distribucion de carga (Falta revisar)
plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection = "3d")

surface = ax.plot_surface(
    X, Y, rho,
    cmap = "coolwarm",
    edgecolor = "k",
    linewidth = 0.2
)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("rho(x, y)")

ax.set_title("Distribucion de carga")

plt.show()