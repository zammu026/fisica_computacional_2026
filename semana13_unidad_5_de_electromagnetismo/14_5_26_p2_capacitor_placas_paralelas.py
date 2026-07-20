# Problema de capacitor de placas paralelas dentro de una caja conductora aterrizada.
r'''
ENUNCIADO:     
Las placas son conductoras muy delgadas con un voltaje superior = +100V e inferior = -100V 
(como si estuvieran conectadas a una bateria).    
La caja entera se encuantra aterrizada a V = 0
 
RESOLVER:
Resolver numéricamente la ecuación de laplace \Delta^2 V = 0 usando
1. Diferencias finitas
2. Metodo de Gauss-Seidel
3. Condiciones de frontera realistas
Ademas:
1. Visualizar lineas equipotenciales
2. Observar efectos de borde
3. Estudiar campos de fuga (fringe fields)
'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

N = 100
tolerance = 1e-4
max_iterations = 10000

V = np.zeros((N, N))
mask = np.zeros((N, N)) # la mask sirve

# placas
# placa superior de +100 V
V[30, 30:70] = 100
mask[30, 30:70] = 1

# placa inferior de de -100 V
V[60, 30:70] = -100
mask[60, 30:70] = 1

# codigo de gauss seidel
for iteration in range(max_iterations):
    Vold = V.copy()
    for i in range(1, N-1):
        for j in range(1, N-1):
            V[i, j] = 0.25*(
                V[i+1, j] +
                V[i-1, j] +
                V[i, j+1] +
                V[i, j-1]
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

ax.set_title("Potencial electrico de un capacitor realista")

plt.show()

# contornos equipotenciales
plt.figure(figsize=(7, 6))
cont = plt.contourf(X, Y, V, 40, cmap = "coolwarm")
plt.colorbar(label = "Potencial (V)")
plt.contour(X, Y, V, 20, color = " ")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Contornos equipotenciales")

plt.show()