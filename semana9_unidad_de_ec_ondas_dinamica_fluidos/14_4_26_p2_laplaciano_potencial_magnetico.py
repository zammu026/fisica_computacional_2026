"""
================================================================================
ENUNCIADO
================================================================================
Resolver numéricamente la ecuación de Laplace elíptica (delta^2 phi = 0) en un 
dominio bidimensional cuadrado empleando un esquema de diferencias finitas 
centradas de segundo orden y el método iterativo de relajación. El sistema está 
gobernado por condiciones de contorno fijas de Dirichlet en sus cuatro paredes.

================================================================================
RESOLVER
================================================================================
1. Configurar la grilla discreta de dimensiones Nx×Ny y definir el espaciamiento 
   espacial homogéneo h.
2. Imponer condiciones de frontera de Dirichlet en los extremos de la matriz `phi` 
   (potenciales fijos en las paredes superior, inferior, izquierda y derecha).
3. Implementar el algoritmo iterativo de relajación en la función `Relax()` para 
   actualizar los nodos internos mediante el promedio ponderado por el parámetro `omega`.
4. Controlar el avance numérico mediante un bucle extendido a `Niter` iteraciones.
5. Construir un mapa de contornos bidimensional relleno (`plt.contourf`) para 
   visualizar la distribución espacial del potencial electrostático o térmico.

================================================================================
CONSIDERACIONES IMPORTANTES
================================================================================
* ALGORITMO ELÍPTICO DE RELAJACIÓN:
  Al no existir un término de forzamiento externo (vorticidad o densidad de carga), 
  el residuo algebraico `r1` se reduce puramente a la diferencia entre el promedio 
  de los cuatro vecinos contiguos y el estado del nodo central estudiado:
    r1 = omega * [ 0.25 * (phi_izq + phi_der + phi_inf + phi_sup) - phi[i, j] ]

* INTEGRIDAD GEOMÉTRICA EN LAS MATRICES:
  Debido a que las mallas discretas se dimensionan hasta `Nx+1` y `Ny+1` para 
  contener las paredes exteriores, los bucles de cálculo espacial de la función 
  `Relax()` deben barrer estrictamente desde `1` hasta `Nx` (excluyendo los bordes 
  fijos) para evitar la sobrescritura accidental de las condiciones de contorno.
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt

# parametros discretizacion: sirve para controlar la resolución
Nx = 50
Ny = 50
h = 1.0

# parametros de relajacion: sirve para controlar la velocidad de convergencia
omega = 0.1
Niter = 2000

# arreglos (matrices vacias)
phi = np.zeros((Nx+1, Ny+1), float)

# condiciones de frontera (Dirichlet de ejemplo)
def Borders():
    # Pared inferior (y = 0)
    for i in range(Nx + 1):
        phi[i, 0] = 0.0

    # Pared superior (y = Ny)
    for i in range(Nx + 1):
        phi[i, Ny] = 100.0  # Potencial alto en el borde superior

    # Pared izquierda (x = 0)
    for j in range(Ny + 1):
        phi[0, j] = 0.0

    # Pared derecha (x = Nx)
    for j in range(Ny + 1):
        phi[Nx, j] = 0.0

def Relax():
    Borders()
    
    # resolver phi (Ecuación de Laplace)
    for i in range(1, Nx):
        for j in range(1, Ny):
            # El residuo algebraico r1 para Laplace prescinde de forzamiento externo
            r1 = ((phi[i+1, j] + phi[i-1, j] + phi[i, j+1] + phi[i, j-1]) * 0.25 - phi[i, j])
            phi[i, j] += omega * r1
            
    Borders()

# iteraciones
for iteration in range(Niter):
    Relax()
    if iteration % 500 == 0:
        print("Iteration: ", iteration)

# malla para graficacion
x = np.arange(Nx + 1) * h
y = np.arange(Ny + 1) * h
X, Y = np.meshgrid(x, y)

# visualizacion de los contornos del potencial phi
plt.figure(figsize=(8, 6))
# Se transpone phi (.T) para acoplar la orientacion indexada de Python con la malla X, Y
plt.contourf(X, Y, phi.T, levels=40, cmap='viridis')
plt.colorbar(label='Potencial phi(x, y)')
plt.title('Solución de la Ecuación de Laplace $\\nabla^2 \\phi = 0$')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
