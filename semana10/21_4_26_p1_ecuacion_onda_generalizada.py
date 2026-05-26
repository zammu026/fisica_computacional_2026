"""
================================================================================
ENUNCIADO
================================================================================
Simular numéricamente la evolución temporal de la ecuación de onda generalizada 
en una cuerda unidimensional de longitud L con extremos fijos empleando el método 
de integración de Leapfrog. Los coeficientes físicos de tensión T(x) y densidad 
lineal de masa rho(x) varían espacialmente de acuerdo con un modelo exponencial.

La ecuación de onda generalizada responde a la relación analítica:
  d/dx [ T(x) * (dy/dx) ] = rho(x) * (d²y/dt²)
Desarrollando la derivada por regla del producto:
  dT(x)/dx * dy(x,t)/dx + T(x) * d²y(x,t)/dx² = rho(x) * d²y(x,t)/dt²

================================================================================
RESOLVER
================================================================================
1. Configurar la malla espacial discreta y establecer los vectores de coeficientes 
   variables T(x), rho(x) y sus derivadas espaciales de primer orden dT/dx.
2. Determinar el paso temporal dt óptimo que garantice el criterio de estabilidad 
   de Courant-Friedrichs-Lewy (CFL) basado en la velocidad máxima local Vmax.
3. Modelar la condición de velocidad inicial cero (dy/dt = 0) adaptando el operador 
   espacial no homogéneo para el cálculo riguroso del paso t = 1.
4. Resolver la ecuación diferencial parcial iterando en el espacio y en el tiempo 
   mediante el esquema centrado de Leapfrog.
5. Imponer condiciones de frontera de Dirichlet homogéneas (extremos fijos, y=0).
6. Graficar instantáneas periódicas del perfil de la cuerda en función del tiempo físico.

================================================================================
CONSIDERACIONES IMPORTANTES
================================================================================
* DISCRETIZACIÓN DEL OPERADOR GENERALIZADO:
  Para aproximar el término de coeficientes variables de segundo orden conservando 
  la precisión del esquema centrado O(dx²), las derivadas espaciales se expanden como:
    dT/dx * dy/dx ≈ ( (T[i+1] - T[i-1]) / (2*dx) ) * ( (y[j, i+1] - y[j, i-1]) / (2*dx) )
    T(x) * d²y/dx² ≈ T[i] * (y[j, i+1] + y[j, i-1] - 2*y[j, i]) / dx²
  El aporte neto de aceleración local queda ponderado por la densidad de masa:
    aceleracion = ( (dT_dx * dy_dx) + (T[i] * d2y_dx2) ) / rho[i]

* AJUSTE DEL PASO DE TIEMPO DEL CRITERIO CFL:
  Debido a que la velocidad local del frente de onda varía punto a punto bajo la 
  relación V(x) = sqrt(T(x) / rho(x)), el paso del tiempo `dt` debe calcularse 
  utilizando el máximo global absoluto de la velocidad (`Vmax`) para blindar el 
  algoritmo contra divergencias numéricas catastróficas.
================================================================================
"""

# Ecuación de onda generalizada
import numpy as np
import matplotlib.pyplot as plt

# parametros fisicos
L = 1.0
rho0 = 0.01
T0 = 40.0
alpha = 0.5

# discretizacion
Nx = 200
dx = L / (Nx - 1)
Nt = 600

# malla espacial
x = np.linspace(0, L, Nx)

# coeficientes variables (Modelo Exponencial de ejemplo)
rho = rho0 * np.exp(alpha * x)
T = T0 * np.exp(alpha * x)

# Derivada analítica de la tensión T(x) respecto a x
dT_dx = alpha * T0 * np.exp(alpha * x)

# Criterio de estabilidad basado en la velocidad máxima local V = sqrt(T/rho)
V = np.sqrt(T / rho)
Vmax = np.max(V)
dt = 0.4 * dx / Vmax # factor de seguridad 0.4 para garantizar Courant < 1

# inicializacion
y = np.zeros((Nt, Nx))

# condicion inicial (pulso gaussiano)
y[0, :] = np.exp(-100 * (x - 0.5)**2)

# para velocidad inicial cero (Paso t = 1 adaptado al operador generalizado)
for i in range(1, Nx - 1):
    derivada_primera = ((T[i+1] - T[i-1]) / (2*dx)) * ((y[0, i+1] - y[0, i-1]) / (2*dx))
    derivada_segunda = T[i] * (y[0, i+1] + y[0, i-1] - 2*y[0, i]) / dx**2
    aceleracion = (derivada_primera + derivada_segunda) / rho[i]
    
    y[1, i] = y[0, i] + 0.5 * (dt**2) * aceleracion

# condiciones de frontera iniciales
y[0, 0] = y[0, -1] = 0
y[1, 0] = y[1, -1] = 0

# evolucion temporal (leapfrog generalizado)
for j in range(1, Nt - 1):
    for i in range(1, Nx - 1):
        # Discretización centrada de dT(x)/dx * dy(x,t)/dx
        term_gradiente = ((T[i+1] - T[i-1]) / (2*dx)) * ((y[j, i+1] - y[j, i-1]) / (2*dx))
        
        # Discretización centrada de T(x) * d^2y(x,t)/dx^2
        term_curvatura = T[i] * (y[j, i+1] + y[j, i-1] - 2*y[j, i]) / dx**2
        
        # Aceleración neta de segundo orden (d^2y/dt^2) dividida por rho(x)
        aceleracion = (term_gradiente + term_curvatura) / rho[i]
        
        # Avance temporal Leapfrog
        y[j + 1, i] = 2 * y[j, i] - y[j - 1, i] + (dt**2) * aceleracion
    
    # condiciones de frontera de extremos fijos
    y[j + 1, 0] = 0
    y[j + 1, -1] = 0

# visualizacion
plt.figure(figsize=(10, 6))
for i in range(0, Nt, 40):
    plt.plot(x, y[i, :], label = f"t = {i * dt:.3f} s")

plt.xlabel("Posición x")
plt.ylabel("y (x, t)")
plt.title("Evolución Temporal - Ecuación de Onda Generalizada ($T(x)$, $\\rho(x)$ variables)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()

