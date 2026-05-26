"""
================================================================================
ENUNCIADO
================================================================================
Simular numéricamente el perfil de velocidad horizontal vx(x, y) de un fluido en 
un dominio bidimensional empleando el método de Sobrerrelajación Sucesiva (SOR). 
El modelo acopla de manera simplificada la componente advectiva-difusiva con un 
gradiente de presión espacial predefinido P(x, y).

================================================================================
RESOLVER
================================================================================
1. Configurar la malla espacial discreta y establecer las condiciones de contorno 
   de Dirichlet homogéneas y de flujo de entrada en la frontera inferior.
2. Implementar el bucle iterativo SOR para resolver el operador elíptico laplaciano 
   asociado al campo de velocidades vx.
3. Incorporar el término de forzamiento del gradiente de presión mediante diferencias 
   finitas centradas dentro del residuo algebraico.
4. Escalar correctamente el residuo respecto a la dimensión espacial h² antes de 
   aplicar el factor de relajación omega.
5. Visualizar la distribución espacial resultante de la velocidad mediante un mapa de calor.

================================================================================
CONSIDERACIONES IMPORTANTES
================================================================================
* PROPIEDAD DE ESCALA EN ALGORITMO SOR:
  El residuo r utilizado para la actualización iterativa del campo debe estar en 
  las mismas unidades de magnitud que la variable de estado (`vx`). Para que esto 
  ocurra, la ecuación gobernante discretizada debe multiplicarse en su totalidad 
  por h² / 4, aislando el nodo central bajo el formato estándar de relajación:
    r = 0.25 * (vx_izq + vx_der + vx_inf + vx_sup - h * gradP) - vx[i, j]

* CONSISTENCIA MATEMÁTICA EN DERIVADAS:
  El gradiente de presión (∂P/∂x) aproximado por diferencias finitas centradas 
  posee un orden de magnitud inversamente proporcional a (2*h). Al integrarlo en 
  la ecuación balanceada del laplaciano multiplicada por h², su aporte neto dentro 
  del residuo numérico queda acotado por el factor geométrico lineal (h / 2).
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt

# parametros 
Nx, Ny = 50, 50
h = 1.0
omega = 1.5 # sobrerrelajacion

#campos
vx = np.zeros((Nx, Ny))
P = np.zeros((Nx, Ny))

# condicion de frontera
vx[:, 0] = 1.0 # flujo de entrada

# iteraciones
for iteraciones in range(500):
    for i in range(1, Nx - 1):
        for j in range(1, Ny - 1):
            # Corregido: Se reestructuró el residuo numérico bajo el formalismo SOR estándar.
            # Despejando la ecuación diferencial balanceada: Laplaciano(vx) = ∂P/∂x
            # Multiplicando todo por (h**2 / 4) para aislar el término vx[i, j]:
            
            suma_vecinos = (vx[i+1, j] + vx[i-1, j] + vx[i, j+1] + vx[i, j-1])
            grad_p = (P[i+1, j] - P[i-1, j]) / (2 * h)
            
            # El residuo algebraico debe normalizarse con el peso de los vecinos (0.25)
            r = 0.25 * (suma_vecinos - (h**2) * grad_p) - vx[i, j]
            
            # actualizacion SOR
            vx[i, j] = vx[i, j] + omega * r

# visualizacion
plt.figure(figsize=(7, 6))
# Corregido: Transposición con .T para que el eje X (primer índice Nx) sea horizontal y el Y vertical
plt.imshow(vx.T, origin='lower', cmap='viridis')
plt.colorbar(label='vx')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Flujo aprox de Navier Stokes')
plt.show()
