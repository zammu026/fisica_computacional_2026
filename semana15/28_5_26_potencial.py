'''
=====================================================================
ENUNCIADO 
=====================================================================
El presente código aborda la simulación numérica del potencial 
eléctrico V(x,y) y el campo eléctrico E(x,y) en una región bidimensional 
donde se encuentra inmerso un cilindro dieléctrico infinito de radio R 
y permitividad relativa eps_r. El sistema se encuentra sometido a un 
campo eléctrico externo originalmente uniforme E0 orientado en el eje x.

La solución numérica requiere resolver la ecuación de Laplace generalizada 
para medios no homogéneos, dada por Nabla * (eps * Nabla V) = 0.

1. Implementar el método de relajación (Jacobi/Gauss-Seidel) para 
   encontrar la distribución estacionaria del potencial V.
2. Incorporar las condiciones de continuidad en la frontera dieléctrica 
   promediando las propiedades del medio (eps) entre nodos vecinos.
3. Garantizar la estabilidad de las condiciones de frontera lejanas 
   para preservar el comportamiento asintótico del campo uniforme.
4. Calcular el gradiente numérico del potencial para obtener las 
   componentes del campo eléctrico Ex y Ey.
5. Visualizar los resultados mediante líneas equipotenciales y líneas 
    de campo eléctrico (streamplot), identificando la distorsión geométrica 
   provocada por la polarización del cilindro.

CONSIDERACIONES
- El dominio espacial está discretizado en una malla Nx x Ny de -1 a 1.
- El cilindro se modela mediante una máscara booleana ('inside') basada 
  en la ecuación analítica del círculo.
- Para optimizar el rendimiento, el bucle iterativo de diferencias finitas 
  se encuentra completamente vectorizado utilizando arreglos de NumPy.
=====================================================================
'''

import numpy as np
import matplotlib.pyplot as plt

# Tama o d ela malla
Nx= 200
Ny= 200

# Dominio
x= np.linspace(-1,1,Nx)
y= np.linspace(-1,1,Ny)

X,Y= np.meshgrid(x,y)

# Campos uniformes
E0=1.0

# Potencial inicial (Condición de campo uniforme lejano: V = -E0 * x)
V = -E0 * X

# Radio del cilindro
R= 0.3

# permitividad relativa
eps_r= 5.0

# Mascara del cilindro
inside = X**2 + Y**2 <= R**2
eps = np.ones((Ny, Nx))
eps[inside] = eps_r

iterations= 5000

# Matrices de conductancias efectivas entre nodos vecinos (Diferencias finitas para eps variable)
eps_w = 0.5 * (eps[:, 1:-1] + eps[:, :-2])    # Lado Oeste (j-1)
eps_e = 0.5 * (eps[:, 1:-1] + eps[:, 2:])     # Lado Este (j+1)
eps_s = 0.5 * (eps[1:-1, :] + eps[:-2, :])    # Lado Sur (i-1)
eps_n = 0.5 * (eps[1:-1, :] + eps[2:, :])     # Lado Norte (i+1)

# Factor de normalización para cada nodo central (i, j)
denom = eps_w[1:-1, :] + eps_e[1:-1, :] + eps_s[:, 1:-1] + eps_n[:, 1:-1]

for n in range(iterations):
	V_old= V.copy()
	
	# VECTORIZACIÓN: Reemplaza por completo los bucles for i/j anidados manteniendo tu lógica
	V[1:-1, 1:-1] = (eps_w[1:-1, :] * V_old[1:-1, :-2] +   # V[i, j-1]
	                 eps_e[1:-1, :] * V_old[1:-1, 2:] +    # V[i, j+1]
	                 eps_s[:, 1:-1] * V_old[:-2, 1:-1] +   # V[i-1, j]
	                 eps_n[:, 1:-1] * V_old[2:, 1:-1])     # V[i+1, j]
	V[1:-1, 1:-1] /= denom

	# Condiciones de frontera (Manteniendo tu nomenclatura y corrigiendo el potencial real)
	V[:, 0] = E0              # Izquierda (x = -1)
	V[:, -1] = -E0            # Derecha (x = 1)
	V[0, :] = -E0 * x         # Superior deforma con el campo original
	V[-1, :] = -E0 * x        # Inferior deforma con el campo original

# Gradiente numerico
Ey,Ex= np.gradient(-V)

fig,ax= plt.subplots(figsize=(8,8))

# Equipotemciales
cont= ax.contour(X,Y,V, levels=30, cmap= 'coolwarm')

# Campo electrico
ax.streamplot(X,Y,Ex,Ey,color='black', density=1.5)

# Cilindro
circle= plt.Circle((0,0),R,color='gray',alpha=0.4)

ax.add_patch(circle)

ax.set_xlabel('x')
ax.set_ylabel('y')

ax.set_title('Campos electricos y potenciales')

plt.show()
