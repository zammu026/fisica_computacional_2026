"""
================================================================================
ENUNCIADO
================================================================================
Simular de forma dinámica mediante una animación interactiva la evolución temporal 
de la ecuación de Korteweg-de Vries (KdV) en una dimensión. El sistema parte de 
una condición inicial basada en ruido gaussiano aleatorio, con el propósito de 
observar numéricamente el proceso de autoorganización no lineal que da lugar a la 
formación y propagación de solitones.

================================================================================
RESOLVER
================================================================================
1. Configurar los arreglos espaciales basándose en los parámetros de discretización 
   proporcionados y definir la condición inicial de ruido aleatorio uniforme.
2. Implementar dentro de la función de actualización (`update`) el bucle espacial 
   que aproxima las derivadas espaciales del término advectivo no lineal y el término 
   dispersivo de tercer orden.
3. Corregir el intercambio y flujo algebraico de las variables de estado temporal 
   (`u`, `u_old`, `u_new`) para evitar el estancamiento numérico de la solución.
4. Ajustar dinámicamente los límites del eje vertical (`ylim`) de la gráfica para 
   contener el crecimiento de la amplitud provocado por las ondas solitarias.
5. Sincronizar el motor de animación (`FuncAnimation`) para actualizar y redibujar 
   el estado del sistema de manera fluida cuadro por cuadro.

================================================================================
CONSIDERACIONES IMPORTANTES
================================================================================
* ALGORITMO DE DISCRETIZACIÓN DE TERCER ORDEN:
  Para modelar correctamente la dispersión física (d³u/dx³), el esquema centrado 
  de cinco puntos en diferencias finitas debe conservar la coherencia de signos:
    dispersion ≈ ( u[i+2] - 2*u[i+1] + 2*u[i-1] - u[i-2] ) / (2 * dx³)
  Un error de signos o ponderación altera la simetría e induce inestabilidades 
  numéricas destructivas que deforman la onda rápidamente.

* INTERCAMBIO TEMPORAL DEL ESQUEMA EXPLÍCITO:
  El avance temporal requiere almacenar el estado presente en el historial previo 
  (`u_old = u.copy()`) exactamente un paso antes de sobrescribir el estado 
  presente con los nuevos valores de la malla espacial (`u = u_new.copy()`). 
  Realizar esta copia a la inversa o de forma incorrecta detiene la simulación.

* LÍMITES DE FRONTERA DE LA MALLA DISCRETA:
  Debido a que el bucle espacial excluye los extremos (`range(2, N-2)`), es mandatorio 
  asegurar que las fronteras se mantengan acotadas o estables en cada iteración temporal 
  para evitar que valores indeterminados afecten los nodos internos adyacentes.

* ESCALA VISUAL DE AMPLIACIÓN (EJE Y):
  Dado que el ruido inicial tiene una baja amplitud pero el acoplamiento KdV 
  acumula la energía en picos delgados de mayor altura (solitones), la escala 
  Y fija original de `(-0.2, 10.2)` resulta excesiva y oculta los detalles. Un rango 
  de `(-0.5, 1.5)` permite apreciar la evolución física con claridad.
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

N = 300
dx = 0.4
dt = 0.1
eps = 0.2
mu = 0.1

x = np.arange(N) * dx

# condiciones iniciales (pulso del soliton)
#u = 0.5 * (1 - np.tanh(x/5 - 5)) # tipo soliton (default)
#u = np.exp(-(x-30)**2/50) # gaussiano
#u = np.exp(-(x-20)**2/40) + 0.5*np.exp(-(x-60)**2/40) # dos solitones
#u = np.zeros_like(x) # rectangular
#u[60:100] = 1
np.random.seed(0) # Semilla para reproducibilidad del ruido
u = 0.2 * np.random.randn(N)

u_old = u.copy()
u_new = np.zeros_like(u)

fig, ax = plt.subplots()
line, = ax.plot(x, u)
ax.set_ylim(-0.5, 1.5) # Ajustado para visualizar correctamente la emergencia desde el ruido
ax.set_xlim(0, N*dx)
ax.set_xlabel('Posición x')
ax.set_ylabel('Amplitud u')
ax.set_title('Evolución dinámica KdV - Emergencia de Solitones')

def update(frame):
    global u, u_old, u_new

    # Copia el estado actual antes de calcular el nuevo paso
    u_old[:] = u[:]

    for i in range(2, N-2):
        nonlinear = (u[i+1] + u[i] + u[i-1]) * (u[i] - u[i-1])
        # Corregido: signos de la aproximación de la tercera derivada espacial
        dispersion = (u[i+2] - 2*u[i+1] + 2*u[i-1] - u[i-2])

        u_new[i] = (u_old[i] - eps*dt/(3*dx) * nonlinear - mu*dt/(2*dx**3)*dispersion)

    # Condiciones de frontera estables en los bordes no calculados
    u_new[0:2] = 0
    u_new[-2:] = 0

    # Actualiza el estado principal con los nuevos datos calculados
    u[:] = u_new[:]

    line.set_ydata(u)
    return line,

ani = FuncAnimation(fig, update, frames=200, interval=50, blit=True)
plt.show()
