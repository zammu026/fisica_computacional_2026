import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# simulacion 1D de ondas electromagneticas usando el metodo FDTD
c = 3e8
mu0 = 4 * np.pi * 1e-7
eps0 = 1 / (c**2 * mu0)

# parametros de la malla
Nz = 400 
dz = 1e-3

# condicion de courant
dt = dz/(2*c)

# numero de pasos temporales
Nt = 1200
z = np.arange(Nz)

Ex = np.zeros(Nz)
Hy = np.zeros(Nz)
z0 = Nz//2
sigma = 25

# Ajuste de longitud de onda en pasos de malla
lambda0 = 120
k = 2*np.pi / lambda0

phase = 0

# CORRECCIÓN: Sintaxis de la multiplicación de arreglos
Ex = np.exp(-((z-z0)**2) / (2*sigma**2)) * np.sin(k*z)
# CORRECCIÓN: Relación física correcta para la amplitud inicial de Hy (Z0 = E/H)
Z0 = np.sqrt(mu0 / eps0)
Hy = (1 / Z0) * np.exp(-((z-z0)**2) / (2*sigma**2)) * np.sin(k*z + phase)

def update(frame):
    global Ex, Hy
    
    # CORRECCIÓN: Signo menos (-) en la ley de Faraday
    for i in range(Nz - 1):
        Hy[i] = Hy[i] - (dt / (mu0 * dz)) * (Ex[i + 1] - Ex[i])
        
    # CORRECCIÓN: Índices correctos para Hy (Hy[i] - Hy[i-1]) en la ley de Ampère
    for i in range(1, Nz):
        Ex[i] = Ex[i] - (dt / (eps0 * dz)) * (Hy[i] - Hy[i - 1])
        
    # Condiciones de frontera
    Ex[0] = 0
    Ex[-1] = 0

    line1.set_ydata(Ex)
    line2.set_ydata(Hy)
    return line1, line2

fig, ax = plt.subplots(figsize=(10, 5))
line1, = ax.plot(z, Ex, label = 'E_x')
# CORRECCIÓN: Se multiplica Hy por Z0 en el gráfico para poder visualizar ambos campos a la misma escala
line2, = ax.plot(z, Hy * Z0, label = 'H_y (escalado)')

ax.set_xlim(0, Nz)
ax.set_ylim(-1.2, 1.2)
ax.grid()
ax.legend()

# CORRECCIÓN: Se añade el parámetro 'frames' para limitar la animación a Nt pasos
ani = FuncAnimation(fig, update, frames=Nt, interval=10, blit=True)
plt.show()
