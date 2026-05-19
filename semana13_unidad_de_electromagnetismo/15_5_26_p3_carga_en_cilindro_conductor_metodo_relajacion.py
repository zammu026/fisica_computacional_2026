import numpy as np
import matplotlib.pyplot as plt

# Parametros
epsilon0 = 8.854e-12

a = 20.0 # radio
L = 20.0 # longitud

# parametros numericos
Nr = 80
Nz = 80

dr = a/(Nr - 1)
dz = L/(Nz - 1)

rho = np.linspace(0, a, Nr)
z = np.linspace(0, L, Nz)

# falta
# Densidad de carga
rho_charge = np.zeros((Nz, Nr))

# Carga puntual aproximada
q = 1e-9

rho0 = 8.0
z0 = 10.0

ir = int(rho0/dr)
iz = int(z0, dz)

# Aproximacion discreta de delta de dirac
rho_charge[iz, dr] = #falta

# parametros iterativos




Nrho = 60
Nphi = 80

# funcion potencial
def potencial(rho, phi, z):
    suma_total = np.zeros_like(rho_dtype =)