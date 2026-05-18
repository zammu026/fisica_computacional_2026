# Placas del capacitor que no estan conectadas a una bateria. En su lugar la placa superior
# posee una densidad de carga uniforme +rho y la placa inferior posee -rho
# Hint. El problema consiste en resolver \Delta^2 V = -rho/permitividad_0


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

N = 100
delta = 1.0
epsilon0 = 1.0
rho0 =  1.0
tolerance = 1e-4
max_iterations = 10000

V = np.zeros((N, N))
rho0 = np.zeros((N, N))
mask = np.zeros((N, N))

# placas
# placa superior de +100 V
rho[30, 30:70] = rho0

# placa inferior de de -100 V
rho[60, 30:70] = -rho0

# metodo de G S
for iteration in range(max_iterations):
    Vold = V.copy()
    for 

# error maximo
error = np.max(np.abs(V - Vold))

if iteration % 100 == 0:
    print("Iteration: ", iteration, "Error: ", error)
    

# Graficación 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, proyection = "3D")

surface = ax.plot_surface(
    X, Y, V,
    cmap = "coolwarm",
    edgecolor = "k",
    linewidth = 0.2
)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("V(x, y)")

# contornos equipotenciales
plt.figure(figsize=(7, 6))
cont = plt.contourf(X, Y, V, 40, cmap = "coolwarm")
plt.colorbar(label = "Potencial (V)")
plt.contour(X, Y, V, 20, color ="")