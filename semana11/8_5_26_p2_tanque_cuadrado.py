"""
================================================================================
ENUNCIADO
================================================================================
Se estudia un tanque cuadrado de dimensiones 30×30 cm con un fluido incompresible.
El fluido entra por la pared izquierda y sale por la derecha.
La velocidad de entrada y salida es v0 = 18 cm/s con viscosidad cinematica nu = 25 cm^2/s.

RECORDANDO:
Se utiliza la formulacion vx = du/dy y vy = -du/dx
La vorticidad es w = dv_y/dx - dv_x/dy
La ecuacion de Poisson es delta^2 u = -w 
y la ecuacion de Navier-Stokes es
nu * delta^2 w = (du/dy)*(dw/dx) - (du/dx)*(dw/dy)

Algoritmo SOR
La relajacion utilizada es 
u_{i,j}^{new} = u_{i,j}^{old} + omega* r_{i,j}^1
w_{i,j}^{new} = w_{i,j}^{old} + omega* r_{i,j}^2
donde 0<w<2 es el parametro de sobrerelajacion

RESOLVER:
1. Resolver la ec. de Navier Stokes
2. Obtener lineas de corriente, vorticidad y campo de velocidades
3. Visualizar contornos de u, w, vectores velocidad (vx, vy)
================================================================================
CONSIDERACIONES IMPORTANTES
================================================================================
* TRANSFERENCIA CONVECTIVA EN NAVIER-STOKES:
  El residuo discreto r2 para la vorticidad (w) debe reproducir directamente los 
  términos advectivos no lineales de la ecuación gobernante conservando el signo:
    nu * ∇²w = vx * (∂w/∂x) + vy * (∂w/∂y)
  Sustituyendo los gradientes de la función de corriente (u):
    r2 = 0.25 * (a1 - (h² / nu) * [ du_dy * dw_dx - du_dx * dw_dy ]) - w[i, j]

* CONDICIONES DE CONTORNO EN LAS PAREDES:
  Para evitar bucles infinitos o asignaciones vacías en nodos móviles (como la 
  entrada izquierda), la aproximación de Thom para la vorticidad de pared fija 
  debe ligarse estrictamente al índice del nodo adyacente interno inmediato (`u[1, j]`).

* INTEGRIDAD DEL CICLO DE RELAJACIÓN:
  Las iteraciones numéricas deben operar bajo un índice de control único y consistente 
  (`iteration`) para evitar colisiones lógicas con los índices espaciales (`i`, `j`) 
  utilizados en los bucles internos de cálculo matricial.
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt

# parametros fisicos
Nx = 30
Ny = 30
h = 1.0
v0 = 18.0
u0 = 18.0

nu = 25.0
omega = 0.1
Niter = 4000

# entrada y salida
J1 = 10
J2 = 20

J3 = 6
J4 = 12

# arreglos
u = np.zeros((Nx+1, Ny+1), float)
w = np.zeros((Nx+1, Ny+1), float)

# condiciones de frontera
def Borders():
    # pared izquierda
    for j in range(Ny + 1):
        # entrada
        if J1 <= j <= J2:
            u[0, j] = v0 * j
            w[0, j] = w[1, j] # Corregido: w[i, j] -> w[1, j] (i no estaba definida aquí)
        else:
            u[0, j] = 0.0
            w[0, j] = 2.0 * (u[0, j] - u[1, j]) / h**2

    # pared derecha
    for j in range(Ny + 1):
        # salida
        if J3 <= j <= J4:
            u[Nx, j] = u[Nx-1, j] + v0 * h
            w[Nx, j] = w[Nx - 1, j]
        else:
            u[Nx, j] = 0.0
            w[Nx, j] = 2.0 * (u[Nx, j] - u[Nx-1, j]) / h**2
            
    # superior
    for i in range(Nx + 1):
        u[i, Ny] = u[i, Ny-1] + u0 * h
        w[i, Ny] = (2.0 * (u[i, Ny] - u[i, Ny-1]) / h**2 - 2.0 * u0 / h)
        
    # inferior
    for i in range(Nx + 1):
        u[i, 0] = 0
        w[i, 0] = 2.0 * (u[i, 0] - u[i, 1]) / h**2

def Relax():
    Borders()
    # relajacion u
    for i in range(1, Nx):
        for j in range(1, Ny):
            r1 = ((u[i + 1, j] + u[i - 1, j] + u[i, j + 1] + u[i, j - 1] + h * h * w[i, j]) * 0.25 - u[i, j])
            u[i, j] += omega * r1
    Borders()

    # relacion w
    for i in range(1, Nx):
        for j in range(1, Ny):
            a1 = (w[i + 1, j]
                  + w[i - 1, j]
                  + w[i, j + 1]
                  + w[i, j - 1])
            du_dy = (u[i, j + 1] - u[i, j - 1])/(2.0 * h)
            du_dx = (u[i + 1, j] - u[i - 1, j])/(2.0 * h)
            dw_dy = (w[i, j + 1] - w[i, j - 1])/(2.0 * h)
            dw_dx = (w[i + 1, j] - w[i - 1, j])/(2.0 * h)
            
            # Corregido: Ecuación física adaptada al operador residuo SOR según el enunciado
            # nu * Lap(w) = du_dy * dw_dx - du_dx * dw_dy -> Lap(w) = (1/nu) * (...)
            conveccion = (du_dy * dw_dx - du_dx * dw_dy)
            w_lap_calc = 0.25 * (a1 - (h**2 / nu) * conveccion)
            r2 = w_lap_calc - w[i, j]
            w[i, j] += omega * r2

# iteraciones
# Corregido: Se cambió la variable del bucle a 'iteration' para no romper el print ni los índices de abajo
for iteration in range(Niter):
    Relax()
    if iteration % 500 == 0:
        print("Iteration: ", iteration)

# velocidades
vx = np.zeros_like(u)
vy = np.zeros_like(u)

for i in range(1, Nx):
    for j in range(1, Ny):
        vx[i, j] = (u[i, j + 1] - u[i, j - 1]) / (2.0 * h)
        vy[i, j] = -(u[i + 1, j] - u[i - 1, j]) / (2.0 * h)

# malla
x = np.arange(Nx + 1)
y = np.arange(Ny + 1)
X, Y = np.meshgrid(x, y)

# 1. Grafica de lineas de corriente
plt.figure(figsize = (8, 6))
plt.contourf(X, Y, u.T, levels = 40, cmap='viridis')
plt.colorbar(label = 'u(x, y)')
plt.title('Líneas de Corriente (u)')
plt.xlabel('x (cm)')
plt.ylabel('y (cm)')
plt.show()

# 2. Grafica de vorticidad
plt.figure(figsize = (8, 6))
plt.contourf(X, Y, w.T, levels = 40, cmap='inferno')
plt.colorbar(label = 'w(x, y)')
plt.title('Contornos de Vorticidad (w)')
plt.xlabel('x (cm)')
plt.ylabel('y (cm)')
plt.show()

# 3. Grafica del campo vectorial de velocidades
plt.figure(figsize = (8, 6))
# Se usa quiver para graficar los vectores velocidad (vx, vy) solicitados en el punto 3
plt.quiver(X, Y, vx.T, vy.T, scale=200)
plt.title('Campo Vectorial de Velocidades (vx, vy)')
plt.xlabel('x (cm)')
plt.ylabel('y (cm)')
plt.show()
