# Considere una región cuadrada bidimencional discretizada mediante una malla uniforme
# de tamaño N por N, use N = 50
# El potencial electrico V(x, y) satisface la ecuacion de Poisson
#  \Delta V = - \rho(x, y)/e0
# CONDICONES DE FRONTERA 
# 1) Supongamos que elborde superior de la caja se tiene un potencial V0 = 500
# mientras que los otros tres lados de la region permancen aterrizados
# 2) Existe una pequeña región cargada en el centro de la malla con densidad de carga
# uniforme positiva: rho(x, y) = rho_0 = 50
# ACTIVIDADES 
# a) Utilice el método de Gauss-Seidel para resolver numéricamente 
# b) Realice una gráfica 3D del potencial V(x, y)
# c) Grafique las lineas equipotenciales
# d) Analice cómo cambia el potencial al modificar: - magnitud de la densidad, tamaño de la región y signo de rho

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parametros
N = 50
delta = 1.0

epsilon0 = 1.0

rho0 = 100.0

tolerance = 1e-5
max_iterations = 10000

# Inicialización
V = np.zeros((N, N))
rho = np.zeros((N, N))

#Método de Gauss-Seidel
for iteration in range(max_iterations):
    Vold = V.copy()
    for i in range(1, N-1):
        for j in range(1, N-1):

            V[i, j] = 0.25*(
                V[i+1, j] +
                V[i-1, j] +
                V[i, j+1] +
                V[i, j-1] +
                delta**2 * rho[i, j]/epsilon0
            )
# 
# error maximo
error = np.max(np.abs(V - Vold))

if iteration % 100 == 0:
    print("Iteration: ", iteration, "Error: ", error)
if error < tolerance:
    print("\nConvergencia alcanzada")
    

# Graficación 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, proyection = "3D")

surface = ax.plot_surface(
    X, Y, V,
    cmap = "coolwarm",
    edgecolor = "k",
    linewidth = 0.2
)

# Visualización del potencial
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("V(x, y)")

# Contornos equipotenciales
plt.figure(figsize=(7, 6))
cont = plt.contourf(X, Y, V, 40, cmap = "coolwarm")
plt.colorbar(label = "Potencial (V)")
plt.contour(X, Y, V, 20, color ="")

# Visualización del campo eléctrico
plt.figure(figure = (7, 7))
plt.contour(X, Y, V, 20)
plt.quiver(X, Y, Ex, Ey)
plt.title("Campo electrico y equipotenciales")

plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.show()