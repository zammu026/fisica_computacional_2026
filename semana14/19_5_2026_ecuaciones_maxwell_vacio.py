# Ecuaciones de maxwell en el vacio
r'''
RECORDANDO:
nabla.E = 0
nabla.H = 0
nabla × E = - mu_0 dH/dt
nabla × H = epsilon_0 dE/dt 

Los campos electricos variabless generan campos magneticos
Los campos magneticos variabless generan campos electricos 
===> El acoplamiento forma ondas electromagneticas

# Ecuciones de onda en el espacio vacio:
En la direccion z: 
dEx/dt = -1/epsilon_0 mu_0 dHz/dt
dHy/dt = -dEx/dt
Las variaciones espaciales de H producen...

APROXIMACIONES POR DIFERENCIAS FINITAS
# Derivada temporal:
d/dt E(z,t) aprox (E(z,t + delta_t/2) - E(z,t - delta_t/2))/delta_t

# Derivada espacial:
d/dz E(z,t) aprox (E(z + delta_z/2, t) - E(z - delta_z/2, t))/delta_x


MALLA DE YEE
Ex y Hz se calculan en mallas desplazadas
Existe un desface de delta Z/2 y delta t/2

NOTACION DISCRETA
Se discretiza 
z = h delta z
t = n delta t

entonces, Ex(z,t) ==> E_y^{k, n} k= indice espacial, n = indice temporal
y Hy(z,t) ==> H_y^{k, n}

ALGORITMO
Iniciar los campos de Ex y Hy
Actualizar los campos de Ex y Hy
Aplicar condiciones de frontera
Graficar
Repetir

RESULTADO:
La onda electromagnetica se propaga automaticamente como consecuencia de las ecuaiones de maxwell
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# parametros fisicos
c = 1.0 # velocidad de la luz normalizada
Nz = 300 # puntos espaciales
dz = 1.0 # paso espacial
beta = 0.4 # condicion de courant (factor multiplicativo de FDTD)
dt = beta * dz / c # paso temporal

steps = 600

# campos
Ex = np.zeros(Nz)
Hy = np.zeros(Nz)

# pulso inicial gaussiana (Viajando hacia la derecha)
z = np.arange(Nz)
Ex = np.exp(-((z-80)/15)**2)
Hy = Ex * (beta / dz) * dz  # Relación de impedancia para que viaje a la derecha

# figura
fig, ax = plt.subplots(figsize=(10, 5))
lineE, = ax.plot(z, Ex, label = 'E_x')
lineH, = ax.plot(z, Hy, label = 'H_y')

ax.set_xlim(0, Nz)
ax.set_ylim(-1.2, 1.2)

ax.set_xlabel('z')
ax.set_ylabel('Campo')
ax.legend()

# actualizacion de FDTD
def update(frame):
    global Ex, Hy
    
    # 1. Actualizar Ex usando el valor anterior de Ex y las diferencias de Hy
    # E(t + dt) = E(t) - dt/(epsilon*dz) * dH/dz
    Ex[0:-1] = Ex[0:-1] - beta * (Hy[1:] - Hy[0:-1])
    
    # 2. Actualizar Hy usando la NUEVA Ex
    # H(t + dt/2) = H(t - dt/2) - dt/(mu*dz) * dE/dz
    Hy[1:] = Hy[1:] - beta * (Ex[1:] - Ex[0:-1])
    
    # condiciones de frontera simple (nodos absorbentes o reflejantes de cero)
    Ex[0] = Ex[-1] = 0.0
    Hy[0] = Hy[-1] = 0.0

    lineE.set_data(z, Ex)
    lineH.set_data(z, Hy)
    return lineE, lineH

# animacion
ani = FuncAnimation(fig, update, frames=steps, interval = 20, blit=True)
plt.show()
