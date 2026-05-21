import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# simulacion 1D de ondas electromagneticas usando el metodo FDTD
c = 3e8
mu0 = 4 * np.pi * 1e-7
eps0 = 1 / (c**2 * mu0)

# parametros de la malla
Nz = 400 
z1 = 100
z2 = 150

eps_r = eps0 * np.ones(Nz)

dz = 1e-3

# condicion de courant
dt = dz/(2*c)

# numero de pasos temporales
Nt = 1500

Ex = np.zeros(Nz)
Hy = np.zeros(Nz)

n = 2.0

eps_r[z1:z2] = (n**2)*eps0

z = np.arange(Nz)

z0 = 80
sigma = 20
lambda0 = 40

k = 2*np.pi / lambda0
Ex = np.exp(-(z-z0)**2/(2*sigma**2)) * np.sin(k*z)

fig, ax = plt.subplots(figsize=(10, 5))
line1, = ax.plot(z, Ex, label = 'E_x')
line2, = ax.plot(z, Hy, label = 'H_y')

ax.axvspan(z1, z2, alpha=0.2) # nuevo

ax.set_xlim(0, Nz)
ax.set_ylim(-1.2, 1.2)
ax.grid()
ax.legend()

def update(frame):
    global Ex, Hy
    for i in range(Nz - 1):
        Hy[i] = Hy[i] + (dt / (mu0 * dz)) * (Ex[i + 1] - Ex[i])
        
    for i in range(1, Nz):
        Ex[i] = Ex[i] + (dt / (eps_r[i] * dz)) * (Hy[i] - Hy[i - 1]) # aqui hay un campo porque el dielectrico modifica la onda
        
    Ex[0] = 0
    Ex[-1] = 0

    line1.set_ydata(Ex)
    line2.set_ydata(Hy)
    return line1, line2

ani = FuncAnimation(fig, update, frames=Nt, interval=10, blit=True)
plt.show()