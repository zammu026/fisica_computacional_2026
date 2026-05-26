"""
================================================================================
ENUNCIADO
================================================================================
Simular numéricamente la evolución temporal de una onda ideal en una cuerda unidimensional 
de longitud L con extremos fijos empleando el método de integración temporal Leapfrog. 
El sistema parte de una condición inicial dada por un pulso gaussiano y(x,0) = exp(-100(x-0.5)^2),
velocidad inicial nula.

================================================================================
RESOLVER
================================================================================
1. Configurar la malla espacial discreta y establecer el paso temporal dt que 
   garantice la condición de estabilidad de Courant (CFL <= 1).
2. Definir adecuadamente los dos primeros pasos de tiempo (t=0 y t=1) para modelar 
   fielmente la condición física de velocidad inicial cero mediante diferencias finitas.
3. Resolver la ecuación de onda discreta iterando en el espacio y en el tiempo 
   a través del esquema de Leapfrog centrado.
4. Imponer condiciones de frontera de Dirichlet homogéneas (extremos fijos, y=0) 
   en cada paso temporal.
5. Graficar instantáneas periódicas del perfil de la cuerda corrigiendo las referencias 
   de los índices para mostrar el tiempo físico correcto en las etiquetas.

================================================================================
CONSIDERACIONES IMPORTANTES
================================================================================
* REFERENCIA EXTRÍNSECA EN ETIQUETAS (BUG DE INDIZACIÓN):
  Al construir las leyendas de las gráficas con `f"t = {j * dt:.2f}"`, se estaba 
  empleando la variable global `j` (cuyo último valor remanente del bucle es `Nt-2`), 
  lo que provocaba que todas las curvas mostraran erróneamente el mismo tiempo final. 
  La expresión debe ligarse directamente al índice local del gráfico: `f"t = {i * dt:.2f}"`.

* APROXIMACIÓN DE LA VELOCIDAD INICIAL NULA:
  Para modelar una velocidad inicial nula (dy/dt = 0 en t=0) en diferencias finitas, 
  el pulso gaussiano debe asignarse al instante de partida `y[0, :]`. Aplicando el 
  esquema centrado, el primer paso físico de avance se deduce rigurosamente como:
    y[1, i] = y[0, i] + 0.5 * Courant² * (y[0, i+1] + y[0, i-1] - 2*y[0, i])
  Invertir la asignación o forzar la igualdad directa entre capas inyecta una 
  velocidad numérica artificial que deforma la simetría de la onda reflejada.
================================================================================
"""
# Solucion de la ecuacion de onda por el metodo leapfrog
import numpy as np
import matplotlib.pyplot as plt

# parametros fisicos
L = 1.0 # longitud
c = 1.0 # velocidad

# discretizacion
Nx = 100 # cantidad de puntos en la malla
dx = L / (Nx - 1) # paso espacial en la malla, delta x

dt = 0.005 # ajustar para cumplir Courant delta t
Nt = 300 # cantidad de pasos temporales
print((dt*c/dx)**2) # condicion de Courant

# estabilidad
courant = c * dt / dx
print(f"Courant: {courant}")

# malla espacial
x = np.linspace(0, L, Nx)

# inicializacion
y = np.zeros((Nt, Nx))

# Corregido: Ajuste riguroso para velocidad inicial cero (dy/dt = 0 en t=0)
# Condición inicial real en t = 0
y[0, :] = np.exp(-100 * (x - 0.5)**2)

# El paso t = 1 se calcula usando la aproximación de la derivada temporal para velocidad cero
for i in range(1, Nx - 1):
    y[1, i] = y[0, i] + 0.5 * (courant**2) * (y[0, i + 1] + y[0, i - 1] - 2*y[0, i])

# condiciones de frontera para el paso inicial, necesitas los vecinos izquierdo (i - 1) y derecho (i + 1) para calcular el cambio
y[0, 0] = y[0, -1] = 0 
y[1, 0] = y[1, -1] = 0

# evolucion temporal (leapfrog)
for  j in range(1, Nt -1):
    for i in range(1, Nx -1):
        y[j + 1, i] = (2*y[j, i] - y[j - 1, i] + (courant**2) * (y[j, i + 1] + y[j, i - 1]- 2*y[j, i] ))
    
    # condiciones de frontera o contorno
    y[j + 1, 0] = 0
    y[j + 1, - 1] = 0

# visualizacion
plt.figure(figsize=(10, 6))
for i in range(0, Nt, 20):
    # Corregido: Se cambió j * dt por i * dt para que la leyenda muestre el tiempo real de la curva graficada
    plt.plot(x, y[i, :], label = f"t = {i * dt:.3f} s")

plt.xlabel("x")
plt.ylabel("y (x, t)")
plt.title(f"Evolucion temporal de una cuerda con Metodo de Leapfrog (courant = {courant:.2f})")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') # Mover leyenda fuera para que no tape las curvas
plt.tight_layout()
plt.show()