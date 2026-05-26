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
3. Resolver de forma iterativa el acoplamiento de las ecuaciones mediante un bucle 
   de relajación numérico extendido a Niter iteraciones.
4. Calcular el campo vectorial de velocidades locales (vx, vy) mediante derivadas 
   parciales de primer orden a partir del potencial de la función de corriente.
5. Generar los mapas de contorno espaciales para la función de corriente, la 
   vorticidad y el campo de velocidades resultante.

================================================================================
CONSIDERACIONES IMPORTANTES
================================================================================
* ECUACIONES ACOPLADAS (SISTEMA DE NAVIER-STOKES):
  El algoritmo resuelve numéricamente la formulación Función de Corriente (u) - 
  Vorticidad (w) para flujos incompresibles bidimensionales:
    ∇²u = -w
    ∇²w = R * [ (∂u/∂y)*(∂w/∂x) - (∂u/∂x)*(∂w/∂y) ]

* ALGORITMO DE ITERACIÓN DE RELAJACIÓN:
  Las ecuaciones diferenciales se aproximan mediante diferencias finitas centradas 
  y se resuelven a través del método de sobrerrelajación sucesiva (SOR):
    u_new = u + omega * [ 0.25 * (u_izq + u_der + u_inf + u_sup + h² * w) - u ]

* DERIVACIÓN DEL CAMPO DE VELOCIDADES:
  Las componentes cinemáticas del campo vectorial se obtienen analíticamente a 
  partir del gradiente espacial de la función de corriente (u) usando diferencias centradas:
    vx =  ∂u / ∂y  ≈ ( u[i, j+1] - u[i, j-1] ) / (2 * h)
    vy = -∂u / ∂x  ≈ -( u[i+1, j] - u[i-1, j] ) / (2 * h)

* CONDICIÓN DE FRONTERA DE NO DESLIZAMIENTO:
  En las paredes rígidas, la vorticidad (w) se calcula en función de la deformación 
  del campo escalar adyacente (Fórmula de Thom de primer orden), expresada como:
    w_pared = -2 * (u_interno - u_pared) / h²
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
    
def BottomBefore():
    for i in range(1, Nb + 1):
        u[i, Ndown] = u[i, Ndown-1]
        w[i, Ndown] = -2*(u[i, 0] - u[i, 1])

def Top():
    for i in range(1, Nx):
        u[i, Ny] = u[i, Ny-1]
        w[i, Ny] = 0

def Left():
    for j in range (Ndown, Ny):
        w[0, j] = -2*(u[0, j] - u[1, j])/h**2
        u[0, j] = u[1, j]

def Borders():
    BelowHole()
    BorderRight()
    BottomBefore()
    Top()
    Left()

def relax():
    Borders()

    #resolver u
    for i in range(1, Nx):
        for j in range(1, Ny):
            r1 = omega*((u[i+1,j]+u[i-1,j]+u[i,j+1]+u[i,j-1]+h*h*w[i,j])*0.25 - u[i,j])
            u[i,j] += r1
    Borders()

    # resolver w
    for i in range(1, Nx):
        for j in range(1, Ny):
            a1 = (w[i+1, j]
                  + w[i-1, j]
                  + w[i, j+1]
                  + w[i, j-1])
            a2 = ((u[i, j+1] - u[i, j-1])*
                  (w[i+1, j] - w[i-1, j]))
            a3 = ((u[i+1, j] - u[i-1, j])*
                  (w[i, j+1] - w[i, j-1]))
            
            r2 = omega*((a1 + (R/4.0)*(a3 - a2))/4.0 - w[i, j])
            w[i, j] += r2

# --- EJECUCIÓN DEL BUCLE DE RELAJACIÓN ---
# Corregido: Se añade el bucle temporal para ejecutar las iteraciones numéricas
for _ in range(Niter):
    relax()

# --- CÁLCULO DEL CAMPO DE VELOCIDADES (vx, vy) ---
# Corregido: Inicialización y cálculo matemático de las componentes físicas de velocidad
vx = np.zeros_like(u)
vy = np.zeros_like(u)

# Diferencias finitas centradas para nodos internos
vx[1:-1, 1:-1] = (u[1:-1, 2:] - u[1:-1, :-2]) / (2 * h)
vy[1:-1, 1:-1] = -(u[2:, 1:-1] - u[:-2, 1:-1]) / (2 * h)

# malla
x = np.arange(0, Nx+1)*h
y = np.arange(0, Ny+1)*h
X, Y = np.meshgrid(x, y)

# funcion corriente
plt.figure(figsize=(8, 6))
plt.imshow(u.T, origin='lower', extent=[0, Nx*h, 0, Ny*h], aspect='auto')
plt.colorbar(label='u(x, y)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Funcion de corriente')
plt.show()

# funcion vorticidad
plt.figure(figsize=(8, 6))
plt.imshow(w.T, origin='lower', extent=[0, Nx*h, 0, Ny*h], aspect='auto')
plt.colorbar(label='w(x, y)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Funcion de vorticidad')
plt.show()

# campo vectorial de velocidades
plt.figure(figsize=(8, 6))
# Corregido: Se eliminó el argumento inválido 'density' que genera error en plt.quiver
plt.quiver(X, Y, vx.T, vy.T)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Campo vectorial de velocidades')
plt.show()
