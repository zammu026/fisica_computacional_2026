# Carga puntual q colocada dentro de un cillindro conductor
r'''
ENUNCIADO:
Considere una carga puntual q colocada dentro de un cillindro conductor de radio a y longitud L.
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

from scipy.special import jn, jn_zeros
from mpl_toolkits.mplot3d import Axes3D

# Parametros fisicos y geometricos
epsilon0 = 8.854e-12

q = 1.0  # carga puntual
a = 20.0  # radio del cilindro
L = 20.0  # longitud del cilindro

rho0 = 0.0  # coordenada radial de la carga
phi0 = 0.0  # coordenada angular de la carga
z0 = 10.0  # coordenada z de la carga

# parametros numericos
Nm = 8 # numero de modos angulares
Nm = 12 # numero de ceros de Bessel

Nrho = 60 # puntos radiales
Nphi = 80 # puntos angulares


# funcion potencial
def potencial(rho, phi, z):
    suma_total = np.zeros_like(rho_dtype = np.float64)
    # suma sobre m
    for m in range(-Nm, Nm + 1):
        # orden absoluto para funciones de Bessel
        mabs = abs(m)
        # ceros de bessel Jm
        zeros = jn_zeros(mabs, Nm)
        # suma sobre n
        for n in range(Nm):
            xmn = zeros[n]
            # parte radial
            J1 = jn(mabs, xmn * rho/a)
            J2 = jn(mabs, xmn * rho0/a)
        
        # parte axial
        zmenor = np.minimum(z, z0)
        zmayor = np.maximum(z, z0)

        # evitar overflow
        sh1 = np.sinh(xmn * zmenor / a)
        sh2 = np.sinh(xmn * (L - zmayor) / a)
        sh3 = np.sinh(xmn * L / a)

        # parte angular
        angular = np.exp(1j * m * (phi - phi0))

        # denominador
        denom = (xmn * (jn(mabs + 1, xmn))**2 * sh3)

        # termino completo
        termino = (angular + J1 * J2 * sh1 * sh2 / denom)

        # toma de parte real
        # FALTAAAA SE ENCUENTRA EN EL LIBRO DE LANDAU

        