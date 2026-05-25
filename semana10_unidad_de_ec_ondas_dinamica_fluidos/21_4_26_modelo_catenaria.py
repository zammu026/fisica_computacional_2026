"""
================================================================================
ENUNCIADO
================================================================================
Simular numéricamente la propagación de un pulso gaussiano en una cuerda unidimensional 
de longitud L, donde la densidad lineal de masa rho(x) y la tensión T(x) varían 
espacialmente de acuerdo con dos configuraciones físicas posibles: un modelo 
exponencial y un modelo de catenaria. La dinámica del sistema incorpora un término 
de amortiguamiento viscoso proporcional a la velocidad.

================================================================================
RESOLVER
================================================================================
1. Configurar la malla espacial discreta y determinar el paso de tiempo dt óptimo 
   mediante el cumplimiento de la condición de estabilidad de Courant-Friedrichs-Lewy 
   (CFL) basada en la velocidad máxima de la onda Vmax = max(sqrt(T/rho)).
2. Implementar un esquema de diferencias finitas centrado en el espacio y en el 
   tiempo para aproximar el operador laplaciano ponderado por la tensión espacial variable.
3. Resolver la ecuación diferencial parcial de segundo orden integrando de forma 
   estable el coeficiente de fricción kappa.
4. Imponer condiciones de frontera de Dirichlet homogéneas (extremos fijos, y=0) 
   y simular la evolución temporal de la onda mediante una animación interactiva.

================================================================================
CONSIDERACIONES IMPORTANTES
================================================================================
* ECUACIÓN GOBERNANTE (EDP):
  La física del sistema responde a la ecuación de onda generalizada con coeficientes
  variables y amortiguamiento viscoso:
    rho(x) * (d²y/dt²) + kappa * rho(x) * (dy/dt) = d/dx [ T(x) * (dy/dx) ]

* ALGORITMO DE DISCRETIZACIÓN ESPACIAL:
  El operador espacial (laplaciano ponderado) se aproxima en nodos intermedios (i ± 1/2) 
  para mantener el esquema centrado de segundo orden O(dx²):
    lap = [ T_(i+1/2) * (y_[i+1] - y_[i]) - T_(i-1/2) * (y_[i] - y_[i-1]) ] / dx²
  Donde las tensiones en las fronteras de la celda se promedian linealmente:
    T_(i+1/2) = 0.5 * (T[i] + T[i+1])
    T_(i-1/2) = 0.5 * (T[i] + T[i-1])

* ESQUEMA DE INTEGRACIÓN TEMPORAL (ALGORITMO ESTABLE):
  Sustituyendo las derivadas temporales por diferencias centradas y aplicando un
  tratamiento semi-implícito de primer orden al término de velocidad (dy/dt) para
  garantizar la estabilidad numérica absoluta frente a la disipación:
    d²y/dt² ≈ (y_new[i] - 2*y[i] + y_old[i]) / dt²
    dy/dt   ≈ (y_new[i] - y_old[i]) / (2 * dt)  --> Simplificado algorítmicamente a:
    y_new[i] = [ 2*y[i] - y_old[i] + (dt² * lap / rho[i]) ] / (1 + kappa * dt)

* CRITERIO CFL DE ESTABILIDAD (COURANT-FRIEDRICHS-LEWY):
  El paso del tiempo dt está estrictamente acotado por la velocidad local del frente
  de onda V(x) = sqrt(T(x)/rho(x)). Se define bajo el factor de seguridad C = 0.4:
    dt = C * dx / max( V(x) )
================================================================================
"""

# Modelo catenaria
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

L = 1.0 # longitud
nx = 200 # puntos
dx = L / (nx - 1)  # paso

# parametros fisicos
rho0 = 0.01 # densidad 
T0 = 40.0 # 
alpha = 0.5 #
kappa = 1.0 #
g = 9.8  

x = np.linspace(0, L, nx) # malla espacial

# modelo catenaria (Movido aquí para que T y rho existan antes de calcular V)
modelo = 'exponencial' # catenaria
if modelo == 'exponencial':
    rho = rho0 * np.exp(alpha * x)
    T = T0 * np.exp(alpha * x)

elif modelo == 'catenaria':
    g = 9.8
    rho = rho0 * np.ones_like(x) # Corregido: ones_likes -> ones_like
    T = T0 * np.cosh(rho0 * g * x/T0)

V = np.sqrt(T/rho)
Vmax = np.max(V)

# condicion de estabilidad
dt = 0.4 * dx / Vmax

print('dt = ', dt)

y = np.exp(-200 * (x - 0.5)**2) # pulso gausiano
y_old = y.copy()
y_new = y.copy() # Corregido: copy(nx) -> copy()

def step():
    global y, y_old, y_new
    for i in range(1, nx - 1):
        T_ip = 0.5*(T[i] + T[i+1])
        T_im = 0.5*(T[i] + T[i-1])

        # Corregido: T_im(...) -> T_im * (...) e inclusión del término de amortiguamiento correcto
        lap = (T_ip * (y[i+1] - y[i]) - T_im * (y[i] - y[i-1]))/dx**2
        y_new[i] = (2*y[i] - y_old[i] + dt**2 * lap/rho[i]) / (1 + kappa * dt)

    y_new[0] = 0
    y_new[-1] = 0

    y_old[:] = y[:]
    y[:] = y_new[:]


fig, ax = plt.subplots()
line, = ax.plot(x, y)

ax.set_ylim(-1.2, 1.2) # Ajustado de -0.2 a 1.2 para que el pulso inicial de amplitud 1.0 sea visible
ax.set_title('Onda con T(x), rho(x) variables')
def update(frame):
    for i in range(10):
        step()
    line.set_ydata(y)
    return line, # Corregido: agregado coma para retornar una tupla iterable
ani = FuncAnimation(fig, update, frames=1000, interval = 20, blit=True)
plt.show()
