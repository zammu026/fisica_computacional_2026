"""
================================================================================
ENUNCIADO
================================================================================
Simular numéricamente la evolución temporal de la ecuación de Korteweg-de Vries 
(KdV) a partir de una condición inicial ruidosa y suavizada, con el objetivo de 
observar el fenómeno físico de emergencia y desarrollo de solitones. El resultado 
debe almacenarse en el tiempo y visualizarse mediante una gráfica de superficie 3D.

================================================================================
RESOLVER
================================================================================
1. Inicializar un vector espacial de ruido aleatorio uniforme y aplicar un filtro 
   de suavizado mediante promedios de vecinos contiguos.
2. Implementar las aproximaciones por diferencias finitas para el término no lineal 
   y el término dispersivo de tercer orden de la ecuación KdV.
3. Resolver la integración temporal mediante un esquema explícito de pasos sucesivos, 
   respetando el acoplamiento entre los arreglos de estado (`u`, `u_old`, `u_new`).
4. Imponer condiciones de frontera homogéneas en los extremos del dominio espacial.
5. Construir una malla bidimensional (Meshgrid) para generar una superficie tridimensional 
   que muestre la amplitud `u(x, t)` en función del espacio y el tiempo.

================================================================================
CONSIDERACIONES IMPORTANTES
================================================================================
* OPERADORES REVENTADOS (MÉTODOS NATIVOS):
  La función inexistente `np.rol` debe reemplazarse por `np.roll` para realizar 
  los desplazamientos cíclicos necesarios en el algoritmo de suavizado espacial.

* ALGORITMO DE DISCRETIZACIÓN PARA KDV:
  El término no lineal de transporte se aproxima localmente promediando la amplitud 
  y multiplicándola por la derivada espacial primera:
    nonlinear ≈ ( u[i+1] + u[i] + u[i-1] ) * ( u[i] - u[i-1] ) / dx
  El término de dispersión de tercer orden (d³u/dx³) requiere una plantilla extendida 
  de 5 puntos centrada para mantener la consistencia del orden del error:
    dispersion ≈ ( u[i+2] - 2*u[i+1] + 2*u[i-1] - u[i-2] ) / (2 * dx³)

* DIMENSIONAMIENTO DE MATRICES (ALMACENAMIENTO):
  La matriz de historial temporal `U` debe inicializarse y llenarse con las dimensiones 
  correctas coordinadas con la malla de graficación: `U.shape = (Nt, N)`. Esto asegura 
  que el eje temporal coincida fila por fila con las iteraciones del bucle externo `n`.

* MANEJO DE HISTORIAL TEMPORAL:
  Para que el esquema numérico avance correctamente y no se estanque, la actualización 
  del estado anterior debe tomar el valor actual (`u_old = u.copy()`) justamente *antes* 
  de que el estado actual absorba los nuevos valores calculados (`u = u_new.copy()`).
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

N = 250
dx = 0.4
dt = 0.05
eps = 0.2
mu = 0.1
Nt = 200

x = np.arange(N) * dx
t_vals = np.arange(Nt) * dt

# condicion inicial = ruido
np.random.seed(0)
u = 0.2*np.random.rand(N)

# suavizado (Corregido: np.rol -> np.roll)
u = (np.roll(u,1) + u + np.roll(u,-1)) / 3

u_old = u.copy()
u_new = np.zeros_like(u)

# almacenamiento 3D (Corregido: (N, Nt) -> (Nt, N) para alinearse con las dimensiones de X y T)
U = np.zeros((Nt, N))

# evolucion kdV
for n in range(Nt):
    for i in range(2, N-2):
        nonlinear = (u[i+1] + u[i] + u[i-1]) * (u[i] - u[i-1])
        # Corregido: signos invertidos en la aproximación de la derivada de tercer orden
        dispersion = (u[i+2] - 2*u[i+1] + 2*u[i-1] - u[i-2])

        u_new[i] = (u_old[i] - eps*dt/(3*dx) * nonlinear - mu*dt/(2*dx**3)*dispersion)

    # condiciones de frontera simple
    u_new[0:2] = 0
    u_new[-2:] = 0

    # Corregido: El orden del intercambio temporal se invirtió para conservar la física del paso previo
    u_old = u.copy()
    u = u_new.copy()
    U[n, :] = u

# malla
X, T = np.meshgrid(x, t_vals)

# graficas (Corregido: plt.figure(10, 7) -> figsize=(10, 7))
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, T, U, cmap='viridis', linewidth=0, antialiased=True)
ax.set_xlabel('Posicion x')
ax.set_ylabel('Tiempo t')
ax.set_zlabel('u(x, t)')
ax.set_title('Emergencia de solitones desde ruido (Kdv)')
fig.colorbar(surf, shrink=0.5, aspect=10, label='Amplitud u(x, t)')
plt.show()