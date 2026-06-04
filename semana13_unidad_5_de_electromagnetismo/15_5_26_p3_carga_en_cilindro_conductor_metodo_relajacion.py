# Carga puntual 'q' colocada dentro de un cillindro conductor.
r'''
ENUNCIADO:
Considere una carga puntual 'q' colocada dentro de un cillindro conductor de radio a y longitud L.
Las paredes laterales y las tapas del cilindro se encuentran conectadas
a tierra V = 0 sobre toda superficie conductora. La carga se ubica en (rho_0, phi_0, z_0).

RECORDANDO:
El potencial electrostatico dentro del cilindro sale de la ecuacion de Poisson:
delta^2 V = -rho/epsilon_0
Con condiciones de frontera:
V(rho = a, phi, z) = 0
V(rho, phi, z = L) = 0
V(rho, phi, z = 0) = 0
 
La solucion exacta puede escribirsse como una expresion infinita en funciones de Bessel:
V(r, r_0) = q/(4pi epsilon_0) * sum_{m = - inf}^{inf} * sum_{n = 1}^{inf} * (Bessel terms)

RESOLVER:
a) Construya un programa en python que evalue numericamente las sumas de las soluciones
b) Grafique el potencial V(rho, phi) utilizando visualizacion 3D
c) Verifique que si rho_0 = 0, el sistema posee simetria rotacional y
si rho_0 ¬= 0, la simetria rotacional desaparece
'''

import numpy as np
import matplotlib.pyplot as plt

# Parametros
epsilon0 = 8.854e-12

a = 20.0 # radio
L = 20.0 # longitud

# malla numerico
Nr = 80 # puntos radiales
Nz = 80 # puntos en z

dr = a/(Nr - 1)
dz = L/(Nz - 1)

rho = np.linspace(0, a, Nr)
z = np.linspace(0, L, Nz)

R, Z = np.meshgrid(rho, z)

# potencial
V = np.zeros((Nz, Nr))

# Densidad de carga
rho_charge = np.zeros((Nz, Nr))

# Carga puntual aproximada
q = 1e-9

rho0 = 8.0
z0 = 10.0

ir = int(rho0/dr)
iz = int(z0 / dz)  # CORREGIDO: Se cambió la coma por operador división /

# Aproximacion discreta de delta de dirac
rho_charge[iz, ir] = q / (2 * np.pi * rho0 * dr * dz) # CORREGIDO: Se usó 'ir' en vez de 'dr' como índice, y se corrigió el factor de volumen cilíndrico

# parametros iterativos
max_iterations = 5000
tolerance = 1e-5

# Metodo de Gauss Seidel
print("Iterando...")
for iteration in range(max_iterations): # CORREGIDO: Corregido error de dedo 'iteretion' por 'iteration'
    Vold = V.copy()
    # recorrer puntos interiores
    for i in range(1, Nz-1):
        for j in range(1, Nr-1):
            rj = rho[j]
            # termino radial cilindrico
            term_r = ((1 + dr/(2*rj))*V[i, j+1] + (1 - dr/(2*rj))*V[i, j-1])/(dr**2)
            # termino axial
            term_z = (V[i+1, j] + V[i-1, j])/(dz**2)
            # denominador
            denom = 2/dr**2 + 2/dz**2
            # actualizar Gauss-Seidel
            V[i, j] = (term_r + term_z + rho_charge[i, j]/epsilon0)/denom

    # condicion espacial en rho = 0
    # simetria axial: dV/drho = 0
    V[:, 0] = V[:, 1]
    
    #error maximo
    error =np.max(np.abs(V - Vold))
    if iteration % 100 == 0:
        print(f"Iteration {iteration} Error: {error:.3e}")
    if error < tolerance:
        print(f'Convergencia alcanzada en {iteration} iteraciones')
        break

# conversion a coordenadas cartesianas
theta = np.linspace(0, 2*np.pi, 100)
Rplot, Tplot = np.meshgrid(rho, theta)

Xplot = Rplot * np.cos(Tplot)
Yplot = Rplot * np.sin(Tplot)

# seleccionar corte z = L/2
mid = Nz // 2
Vslice = np.tile(V[mid, :], (100, 1)) # CORREGIDO: Cambiado np.title por np.tile para replicar el arreglo

# graficacion 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection = "3d")

ax.plot_surface(Xplot, Yplot, Vslice, cmap = "viridis", edgecolor = 'none') # CORREGIDO: Se cambiaron 'X' e 'Y' por 'Xplot' e 'Yplot'

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("V")

ax.set_title('Potencial numerico dentro del cilindro')

plt.show()
