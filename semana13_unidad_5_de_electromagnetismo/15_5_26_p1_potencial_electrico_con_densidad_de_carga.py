# Region cuadrada bidimencional discretizada
r'''
ENUNCIADO:
Considere una región cuadrada bidimencional discretizada mediante una malla uniforme 
de tamaño N por N, use N = 50. El potencial electrico V(x, y) satisface la ecuacion de Poisson
delta V = - \rho(x, y)/epsilon_0:

1) Supongamos que elborde superior de la caja se tiene un potencial V0 = 500
mientras que los otros tres lados de la region permancen aterrizados
2) Existe una pequeña región cargada en el centro de la malla con densidad de carga
uniforme positiva: rho(x, y) = rho_0 = 50

RESOLVER: 
a) Utilice el método de Gauss-Seidel para resolver numéricamente 
b) Realice una gráfica 3D del potencial V(x, y)
c) Grafique las lineas equipotenciales
d) Analice cómo cambia el potencial al modificar: - magnitud de la densidad, tamaño de la región
y signo de rho
'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parametros (Ajustados según el enunciado)
N = 50
delta = 1.0

epsilon0 = 1.0
rho0 = 50.0
V0 = 500.0

tolerance = 1e-4
max_iterations = 10000

V = np.zeros((N, N))
rho = np.zeros((N, N))

# Condiciones de frontera: borde superior con potencial V0 (Otras fronteras quedan en 0)
V[0, :] = V0

# Definición de la región cargada en el centro de la malla (Punto 2 del enunciado)
centro_inicio = N // 2 - 5
centro_fin = N // 2 + 5
rho[centro_inicio:centro_fin, centro_inicio:centro_fin] = rho0

# codigo de gauss seidel
for iteration in range(max_iterations):
    Vold = V.copy()
    for i in range(1, N-1):
        for j in range(1, N-1):
            # Gauss-Seidel real utiliza los valores ya actualizados de la iteración en curso
            V[i, j] = 0.25*(
                V[i+1, j] +
                V[i-1, j] +
                V[i, j+1] +
                V[i, j-1] +
                delta**2*rho[i, j]/epsilon0 # agregado
            )

    # error maximo
    error = np.max(np.abs(V - Vold))
    if iteration % 100 == 0:
        print("Iteration: ", iteration, "Error: ", error)
    if error < tolerance:
        print("\nConvergencia alcanzada en la iteración:", iteration)
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

# Calculo del campo electrico (E = -grad(V))
Ex = np.zeros((N, N))
Ey = np.zeros((N, N))

for i in range(1, N-1):
    for j in range(1, N-1):
        Ex[i, j] = -(V[i+1, j] - V[i-1, j])/(2*delta)
        Ey[i, j] = -(V[i, j+1] - V[i, j-1])/(2*delta)

# Contornos equipotenciales
plt.figure(figsize=(7, 6))
cont = plt.contourf(X, Y, V, 40, cmap = "coolwarm")
plt.colorbar(label = "Potencial (V)")
plt.contour(X, Y, V, 20, colors="black", linewidths=0.5) # Corregido: 'colors' en plural y sin cadena vacía
plt.title("Lineas Equipotenciales")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# Visualización del campo eléctrico
plt.figure(figsize = (7, 7)) # Corregido: 'figsize' en lugar de 'figure'
plt.contour(X, Y, V, 20, cmap = "coolwarm")
plt.quiver(X, Y, Ex, Ey)
plt.title("Campo electrico y equipotenciales")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.show()