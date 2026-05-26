"""
================================================================================
ENUNCIADO
================================================================================
Simular numéricamente la evolución temporal de un solitón de anillo (ring soliton) 
en dos dimensiones espaciales resolviendo la ecuación de Sine-Gordon. El estado 
del sistema debe capturarse en pasos de tiempo específicos para proyectar la 
transformación del frente de onda mediante superficies tridimensionales.

================================================================================
RESOLVER
================================================================================
1. Configurar los arreglos espaciales y determinar el paso temporal dt bajo el 
   límite establecido por el criterio de estabilidad de Courant-Friedrichs-Lewy.
2. Definir analíticamente el perfil topológico del solitón de anillo como la 
   condición inicial del campo escalar.
3. Estructurar el bucle principal de evolución de modo que el esquema de integración 
   temporal (Leapfrog), las condiciones de frontera de Neumann y el intercambio de 
   arreglos ocurran estrictamente dentro de cada iteración del tiempo.
4. Almacenar el estado matemático transformado sin(u/2) en los instantes de interés 
   definidos en la lista de control.
5. Construir un lienzo multirrecuadro (Subplots) para renderizar las superficies 3D 
   correspondientes a cada snapshot guardado.

================================================================================
CONSIDERACIONES IMPORTANTES
================================================================================
* ENCAPSULAMIENTO DEL BUCLE TEMPORAL:
  Todas las etapas del algoritmo numérico (cálculo del laplaciano, actualización 
  por Leapfrog, imposición de condiciones de frontera y desplazamiento de los 
  arreglos de memoria `u_prev = u.copy()`) deben ejecutarse *dentro* del bucle 
  `for t in range(...)`. Si se colocan fuera, la simulación se congela y solo 
  grafica el estado inicial repetidamente.

* CONSISTENCIA GEOMÉTRICA DE LA MALLA:
  El paso espacial discreto en una grilla regular se calcula dividiendo la extensión 
  total del dominio entre el número total de intervalos de la celda:
    dx = L / (N - 1)
  Utilizar una definición que no descuente el nodo extremo genera un desajuste sutil 
  entre los límites físicos de `np.linspace` y el operador de derivadas espaciales, 
  alterando la velocidad numérica del frente de onda solitario.
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# parametros
N = 201
L = 14.0
dx = L / (N - 1) # Corregido: Ajustado para que dx sea consistente con np.linspace (N puntos = N-1 intervalos)
dt = dx / np.sqrt(2)
steps_to_plot = [0, 5, 10, 20]

# grillas
x = np.linspace(-L/2, L/2, N)
y = np.linspace(-L/2, L/2, N)
X, Y = np.meshgrid(x, y)

# variables
u = np.zeros((N, N)) # campo actual
u_prev = np.zeros((N, N)) # paso anterior
u_next = np.zeros((N, N)) # siguiente paso

# condicion inicial (ring soliton)
R = np.sqrt(X**2 + Y**2)
u = 4 * np.arctan(np.exp(3 - R))
u_prev = np.copy(u)

# funcion laplaciano
def laplacian(u):
    return (
        np.roll(u, 1, axis = 0) +
        np.roll(u, -1, axis = 0) +
        np.roll(u, 1, axis = 1) +
        np.roll(u, -1, axis = 1)
        - 4*u
    ) / dx**2

# evolucion temporal
snapshots = []

# Corregido: Se identaron todas las operaciones de actualización para que ocurran DENTRO del ciclo temporal
for t in range(max(steps_to_plot) + 1):
    # guardar snapshots
    if t in steps_to_plot:
        snapshots.append(np.sin(u/2))
        
    # leapfrog
    u_next = (
        2*u - u_prev + dt**2 * (laplacian(u) - np.sin(u))
    )

    # condiciones de frontera
    u_next[0, :] = u_next[1, :]
    u_next[-1, :] = u_next[-2, :]
    u_next[:, 0] = u_next[:, 1]
    u_next[:, -1] = u_next[:, -2]

    # avanzar
    u_prev = np.copy(u)
    u = np.copy(u_next)

# graficas 3D
fig = plt.figure(figsize=(16, 4))
for i, snap in enumerate(snapshots):
    ax = fig.add_subplot(1, len(snapshots), i+1, projection= '3d')
    # Corregido: Agregado cmap='viridis' para mejorar sustancialmente la visualización de los relieves del solitón
    surf = ax.plot_surface(X, Y, snap, rstride = 4, cstride = 4, cmap='viridis', edgecolor='none')
    ax.set_title(f't = {steps_to_plot[i]}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('sin (u/2)')
    ax.set_zlim(-1.1, 1.1) # Límites fijos en Z para observar la contracción y colapso del anillo sin saltos de escala
plt.tight_layout()
plt.show()
