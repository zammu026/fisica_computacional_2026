"""
================================================================================
ENUNCIADO
================================================================================
Simular numéricamente la evolución temporal de una onda unidimensional en una 
cuerda sujeta a extremos fijos considerando los efectos disipativos de una fuerza 
de fricción viscosa proporcional a la velocidad (gamma). El sistema se modela 
dinámicamente mediante una animación interactiva utilizando el método de Leapfrog.

================================================================================
RESOLVER
================================================================================
1. Configurar los parámetros de discretización espacial y temporal verificando que 
   se cumpla estrictamente el criterio de estabilidad de Courant.
2. Corregir el módulo de importación de Matplotlib para invocar correctamente la 
   clase constructora `FuncAnimation`.
3. Ajustar el esquema implícito de diferencias finitas de segundo orden para integrar 
   el coeficiente disipativo de fricción de manera numéricamente estable.
4. Definir e inicializar por completo los datos espaciales fijos de la curva (`x`) 
   dentro de la rutina de actualización de la animación para evitar pantallas vacías.
5. Ejecutar la animación a lo largo de todo el espectro de snapshots temporales calculados.

================================================================================
CONSIDERACIONES IMPORTANTES
================================================================================
* ESQUEMA DE AMORTIGUAMIENTO SEGUNDO ORDEN (ESTABILIDAD NUMÉRICA):
  La aproximación de la derivada de primer orden para el rozamiento (dy/dt) mediante 
  diferencias centradas unida al laplaciano exige despejar el término `y[j+1]` de 
  forma semi-implícita. La ecuación física balanceada debe estructurarse como:
    y[j+1] = [ 2*y[j] - (1 - gamma*dt/2)*y[j-1] + r*Lap(y) ] / (1 + gamma*dt/2)
  Utilizar un esquema explícito sin denominador acoplado induce una ganancia de 
  energía numérica artificial que hace divergir exponencialmente la amplitud.

* INICIALIZACIÓN COMPLETA DEL MOTOR DE ANIMACIÓN:
  Al declarar la línea con `ax.plot([], [])` vacía, es mandatorio que la función 
  `update` configure tanto los datos del eje Y como los del eje X (`line.set_xdata(x)`) 
  en la primera iteración. De lo contrario, el objeto gráfico carece de dominio 
  espacial y la pantalla permanece en blanco durante la simulación.
================================================================================
"""

# Consideracion de la fuerza de friccion en el movimiento de cuerdas

import numpy as np
import matplotlib.pyplot as plt
# Corregido: Se corrige la sintaxis de importación para traer FuncAnimation directamente
from matplotlib.animation import FuncAnimation

L = 1.0
c = 1.0

Nx = 200
dx = L / (Nx - 1)

Nt = 800
dt = 0.004

gamma = 1.0 # friccion

x = np.linspace(0, L, Nx)
y = np.zeros((Nt, Nx))

y[1, :] = np.exp(-100 * (x - 0.5)**2)
y[0, :] = y[1, :]

r = (c * dt / dx)**2

for j in range(1, Nt-1):
    # Corregido: Implementación de la ecuación de onda amortiguada
    # con esquema numérico centrado estable
    denominador = 1.0 + 0.5 * gamma * dt
    termino_central = 2.0 * y[j, 1:-1]
    termino_pasado = (1.0 - 0.5 * gamma * dt) * y[j-1, 1:-1]
    laplaciano = r * (y[j, 2:] - 2.0 * y[j, 1:-1] + y[j, 0:-2])
    
    y[j + 1, 1: -1] = (termino_central - termino_pasado + laplaciano) / denominador

# ANIMATION

fig, ax = plt.subplots()
# Corregido: Se pasa 'x' inicialmente para enlazar los puntos del dominio espacial
line, = ax.plot(x, np.zeros(Nx), lw=2)
ax.set_ylim(-1.2, 1.2)
ax.set_xlabel("Posición x")
ax.set_ylabel("Amplitud y")

def update(frame):
    line.set_ydata(y[frame, :])
    ax.set_title(f"Evolución de la onda con fricción - t = {frame * dt:.3f} s")
    return line,

# Corregido: Se invoca la clase FuncAnimation usando el módulo corregido de arriba
ani = FuncAnimation(fig, update, frames = Nt, interval = 20, blit=True)
plt.show()