"""
================================================================================
ENUNCIADO
================================================================================
Simular numéricamente el flujo bidimensional e incompresible de un fluido que 
atraviesa un canal y choca contra un obstáculo rectangular (viga). El modelo recurre 
a las ecuaciones acopladas de Navier-Stokes en su formulación Función de Corriente 
(u) - Vorticidad (w) resolviéndose mediante el método de Sobrerrelajación Sucesiva (SOR).

================================================================================
RESOLVER
================================================================================
1. Inicializar las matrices de estado e imponer la geometría del obstáculo rígido 
   dentro del dominio computacional.
2. Definir e imponer iterativamente las condiciones de frontera de Dirichlet y Neumann 
   para el flujo de entrada, paredes sólidas libres, salida convectiva y contorno de la viga.
3. Corregir el algoritmo iterativo SOR en la función `relax()` para garantizar que 
   el barrido de la malla calcule todos los nodos internos antes de retornar el residuo.
4. Evaluar el criterio de convergencia global mediante la tolerancia predefinida.
5. Derivar el campo cinemático de velocidades (vx, vy) y estructurar los bloques 
   gráficos para proyectar el campo vectorial y las líneas de corriente.

================================================================================
CONSIDERACIONES IMPORTANTES
================================================================================
* RETORNO ANTICIPADO EN BUCLE (BUG DE RETORNO):
  Colocar la sentencia `return max_residual` dentro del bucle anidado `for` de la 
  función `relax()` provoca que la iteración se detenga prematuramente en el primer 
  nodo calculado `(i=1, j=1)`. El retorno del residuo máximo debe indentarse 
  estrictamente al final de la función, fuera de los ciclos espaciales.

* MODELO DE CONDUCCIÓN FÍSICA EN FRONTERA VIGA:
  Para simular correctamente que la viga actúa como una pared sólida impermeable, 
  la función de corriente `u` en su superficie e interior debe asignarse a un 
  valor escalar constante (el potencial de su línea de estancamiento). 

* ESQUEMA DE DERIVACIÓN COMPATIBLE:
  El campo vectorial se obtiene aproximando numéricamente las derivadas primeras 
  mediante diferencias finitas centradas bidimensionales:
    vx =  ∂u / ∂y  ≈ ( u[i, j+1] - u[i, j-1] ) / 2
    vy = -∂u / ∂x  ≈ -( u[i+1, j] - u[i-1, j] ) / 2
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt

Nx = 70
Ny = 24

L_beam = 8 # longitud de la viga
H_beam = 4 # altura de la viga

v0 = 1.0
R = 0.1

omega = 0.3

# parametros de convergencia
max_iter = 5000
tol = 1e-3

# matrices
u = np.zeros((Nx, Ny))
w = np.zeros((Nx, Ny))

# posicion viga
x0 = 20
x1 = x0 + L_beam

y0 = (Ny // 2) - (H_beam // 2)
y1 = y0 + H_beam

def boundary_conditions(): # funciones de rutina
    # entradas 
    for j in range(Ny):
        u[0, j] = v0 * j
        
    #salida
    u[-1, :] = u[-2, :]
    w[-1, :] = w[-2, :]

    # superficie superior
    u[:, -1] = v0 * (Ny - 1)

    # inferior
    u[:, 0] = 0

    #viga (Corregido: Se asegura que el obstáculo físico actúe como frontera sólida impermeable)
    u[x0:x1, y0:y1] = u[x0, y0]
    w[x0:x1, y0:y1] = 0.0 

# Algoritmo SOR
def relax():
    max_residual = 0
    for i in range(1, Nx-1):
        for j in range(1, Ny-1):

            # saltar interior de la viga
            if x0 <= i < x1 and y0 <= j < y1:
                continue 

            # funcion u(vx, vy) y w(vx, vy)
            u_new = 0.25 * (
                u[i+1, j] + u[i-1, j] +
                u[i, j+1] + u[i, j-1] +
                w[i, j]
            )        
            ru = u_new - u[i, j]
            u[i, j] += omega * ru

            a1 = w[i+1, j] + w[i-1, j]
            a2 = w[i, j+1] + w[i, j-1]

            a3 = (R/4.0) * ((u[i, j+1] - u[i, j-1]) *
                    (w[i+1, j] - w[i-1, j]) - (u[i+1, j] - u[i-1, j]) *
                    (w[i, j+1] - w[i, j-1]))
            w_new = 0.25 * (a1 + a2 + a3)
            rw = w_new - w[i, j]
            w[i, j] += omega * rw

            max_residual = max(
                max_residual, 
                abs(ru),
                abs(rw)
            )
            
    return max_residual # Corregido: Se sacó de los bucles for para permitir que recorra toda la malla

for it in range(max_iter):
    boundary_conditions()
    residual = relax()

    # Corregido: Se añade control para evitar errores si 'residual' es NoneType por fallos de convergencia
    if residual is None:
        residual = 1.0

    if it % 100 == 0:
        print(f"Iteracion {it}")
        print(
            "u upstream = ",
            u[x0 - 5, Ny // 2]
        )
        print(
            "u arriba = ",
            u[x0 + L_beam // 2, y1 + 2]
        )
        print(
            "u downstream = ",
            u[x1 + 5, Ny // 2]
        )
        print("Residual = ", residual)
        print("----------------------------------------")

    if residual < tol:
        print(f"Convergencia alcanzada en la iteración {it}")
        break

# visualizacion
vx = np.zeros_like(u)
vy = np.zeros_like(u)

# Corregido: Ajuste de dimensiones espaciales para el cálculo correcto de derivadas centradas internas
vx[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / 2
vy[1:-1, :] = -(u[2:, :] - u[:-2, :]) / 2

# Forzar velocidad cero dentro de la viga para propósitos de visualización física
vx[x0:x1, y0:y1] = 0
vy[x0:x1, y0:y1] = 0

X, Y = np.meshgrid(np.arange(Nx), np.arange(Ny))

# Corregido: Completada la sintaxis faltante en las figuras y agregadas las gráficas finales solicitadas
plt.figure(figsize=(12, 5))

# 1. Campos vectoriales (Quiver)
plt.quiver(X, Y, vx.T, vy.T, scale=20, cmap='viridis')
# Dibujar el contorno del obstáculo rígido
plt.gca().add_patch(plt.Rectangle((x0, y0), L_beam, H_beam, color='red', alpha=0.7, label='Viga'))
plt.title('Campo Vectorial de Velocidades')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

# 2. Líneas de corriente (Contour)
plt.figure(figsize=(12, 5))
plt.contourf(X, Y, u.T, levels=20, cmap='viridis')
plt.colorbar(label='Líneas de corriente u(x,y)')
plt.gca().add_patch(plt.Rectangle((x0, y0), L_beam, H_beam, color='white', edgecolor='black', hatch='/'))
plt.title('Distribución de las Líneas de Corriente')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


