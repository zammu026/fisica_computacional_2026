# Problema de capacitor de placas paralelas dentro de una caja conductora aterrizada.
# Las placas son conductoras muy delgadas con un voltaje superior e inferior dado.
# Hint. Resolver numéricamente la ecuación de laplace \Delta^2 V = 0

import numpy as np
import matplotlib.pyplot as plt
from

N = 100
tolerance = 1e-4
max_iterations = 10000

V = np.zeros((N, N))
mask = np.zeros((N, N))

# placas
# placa superior de +100 V
V[30, 30:70] = 100
mask[30, 30:70] = 1

# placa inferior de de -100 V
V[60, 30:70] = -100
mask[60, 30:70] = 1

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
    linewidth
)

# contornos equipotenciales
plt.figure(figsize=(7, 6))
cont = plt.contourf(X, Y, V, 40, cmap = "coolwarm")
plt.colorbar(label = "Potencial (V)")
plt.contour(X, Y, V, 20, color ="")