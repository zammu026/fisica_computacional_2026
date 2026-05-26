"""
================================================================================
ENUNCIADO
================================================================================
La superficie del fluido desciende con velocidades v0 = 8 × 10^-4 m/s con viscosidad 
del fluido nu = 0.5 poise. El número de Reynolds es R = v0*h/nu, donde h es el 
tamaño de celda de la malla. Simular el comportamiento hidrodinámico mediante 
el método de relajación de la función de corriente (u) y la vorticidad (w).

================================================================================
RESOLVER
================================================================================
1. Configurar la geometría discreta de la cavidad con un canal de descarga inferior.
2. Implementar las condiciones de contorno hidrodinámicas de Dirichlet y Neumann 
   para la función de corriente y la ecuación de vorticidad en fronteras sólidas y libres.
3. Completar y resolver de forma iterativa el acoplamiento de las ecuaciones mediante 
   un bucle de relajación numérico (SOR) extendido a Niter iteraciones.
4. Calcular el campo vectorial de velocidades locales (vx, vy) mediante diferencias 
   finitas centradas aplicadas a la función de corriente.
5. Corregir y completar la sintaxis truncada de Matplotlib para proyectar el mapa 
   de contorno espacial de las líneas de flujo.

================================================================================
CONSIDERACIONES IMPORTANTES
================================================================================
* RECONSTRUCCIÓN DEL RESIDUO DE POISSON:
  El término elíptico truncado `r1` se restituye bajo el formalismo estándar SOR 
  de segundo orden, ponderado por el parámetro geométrico del tamaño de celda h²:
    r1 = omega * [ 0.25 * (u_izq + u_der + u_inf + u_sup + h² * w) - u[i, j] ]

* ACOPLAMIENTO DE TRANSPORTE DE VORTICIDAD:
  Se completa la ecuación de advección no lineal de Navier-Stokes para el residuo `r2`. 
  Las derivadas cruzadas de velocidad y transporte convectivo se acoplan mediante:
    a2 = (u[i, j+1] - u[i, j-1]) * (w[i+1, j] - w[i-1, j])
    a3 = (u[i+1, j] - u[i-1, j]) * (w[i, j+1] - w[i, j-1])
    r2 = omega * [ 0.25 * (a1 + (R / 4.0) * (a3 - a2)) - w[i, j] ]

* SINTAXIS DE CONTORNO EN CELDAS FLUIDAS:
  Se subsanan las variables de referencia erróneas o huérfanas en las fronteras, 
  tales como `w[i, down]` (reemplazada por la dimensión lógica de la pared `w[i, Ndown-1]`) 
  y la falta de la función integradora `Borders()` exigida por el método `Relax()`.
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt

Nx = 60
Ny = 60

h = 0.4
omega = 0.1

g = 9.8
v0 = 8e-4

nu = 0.5

R = v0*h/nu
Niter = 4000

# geometria
Nb = 15
Ndown = 20

# matrices vacias
u = np.zeros((Nx+1, Ny+1), float)
w = np.zeros((Nx+1, Ny+1), float)

# condiciones de frontera
def BelowHole():
    for i in range(Nb+1, Nx+1):
        u[i, 0] = u[i-1, 1]
        w[i-1, 0] = w[i-1, 1]
        for j in range(0, Ndown+1):
            if i == Nb:
                vy_val = 0
            elif i == Nx:
                vy_val = -np.sqrt(2*g*h*(Ny+Nb-j))
            elif i == Nx-1:
                vy_val = -np.sqrt(2*g*h*(Ny+Nb-j))/2
            else:
                vy_val = 0
            u[i, j] = u[i-1, j] - vy_val*h

def BorderRight():
    for j in range(1, Ny+1):
        vy_val = -np.sqrt(2*g*h*(Ny-j))
        u[Nx, j] = u[Nx-1, j] + vy_val*h
        u[Nx, j] = u[Nx, j-1]
        w[Nx, j] = -2*(u[Nx, j] - u[Nx, j-1]) / h**2

def BottonBefore():
    for i in range(1, Nb+1):
        u[i, Ndown] = u[i, Ndown-1]
        w[i, Ndown] = -2*(u[i, 0] - u[i, 1]) / h**2 # Corregido: Expresión de Thom completada

def Top():
    for i in range(1, Nx):
        u[i, Ny] = u[i, Ny-1]
        w[i, Ny] = 0

def Left():
    for j in range(Ndown, Ny):
        w[0, j] = -2*(u[0, j] - u[1, j]) / h**2 # Corregido: Se añade normalización espacial
        u[0, j] = u[1, j]

# Corregido: Se reincorpora la rutina integradora de contornos necesaria para la relajación
def Borders():
    BelowHole()
    BorderRight()
    BottonBefore()
    Top()
    Left()

def Relax():
    Borders()

    # resolver u
    for i in range(1, Nx):
        for j in range(1, Ny):
            # Corregido: Se completa la fórmula del residuo SOR para la Ec. de Poisson
            r1 = omega * ((u[i+1, j] + u[i-1, j] + u[i, j+1] + u[i, j-1] + h*h*w[i, j])*0.25 - u[i, j])
            u[i, j] += r1
            
    Borders()

    # resolver w
    for i in range(1, Nx):
        for j in range(1, Ny):
            a1 = (w[i+1, j]
                  + w[i-1, j]
                  + w[i, j+1]
                  + w[i, j-1])
            # Corregido: Se completan los términos cruzados advectivos y el residuo r2
            a2 = ((u[i, j+1] - u[i, j-1]) * (w[i+1, j] - w[i-1, j]))
            a3 = ((u[i+1, j] - u[i-1, j]) * (w[i, j+1] - w[i, j-1]))
            
            r2 = omega * ((a1 + (R/4.0)*(a3 - a2))/4.0 - w[i, j])
            w[i, j] += r2

# iteraciones
for it in range(Niter):
    Relax()
    if it % 100 == 0:
        print("Iteraciones =", it)

# velocidades
vx = np.zeros_like(u)
vy = np.zeros_like(u)

# Corregido: Se completa el bucle espacial trunco para derivar las velocidades
for i in range(1, Nx):
    for j in range(1, Ny):
        vx[i, j] = (u[i, j+1] - u[i, j-1]) / (2 * h)
        vy[i, j] = -(u[i+1, j] - u[i-1, j]) / (2 * h)

# malla
x = np.arange(0, Nx+1)*h
y = np.arange(0, Ny+1)*h
X, Y = np.meshgrid(x, y)

# funcion corriente
plt.figure(figsize=(8,6))
# Corregido: Se añade la coma faltante en los argumentos de imshow y se completa la visualización
plt.imshow(
    u.T, 
    origin="lower",
    extent=[0, Nx*h, 0, Ny*h],
    aspect="auto",
    cmap='viridis'
)
plt.colorbar(label='u(x, y)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Función de corriente')
plt.show()
