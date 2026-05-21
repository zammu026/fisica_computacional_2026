import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# simulacion 1D de ondas electromagneticas usando el metodo FDTD
c = 3e8
mu0 = 4 * np.pi * 1e-7
eps0 = 1 / (c**2 * mu0)

# parametros de la malla
Nz = 200 
dz = 1e-3

# condicion de courant
dt = dz / (2 * c)

# numero de pasos temporales
Nt = 500

# campos
Ex = np.zeros(Nz)
Hy = np.zeros(Nz)

# fuentes iniciales
x0 = 50
sigma = 10

for k in range(Nz):
    Ex[k] = np.exp(-((k-x0)**2)/(2*sigma**2))

# configuracion grafica
fig, ax = plt.subplots(figsize=(10, 5))

# Se inicializan los datos del eje X para que la animación funcione correctamente
line1, = ax.plot(np.arange(Nz), Ex, label = 'E_x')
line2, = ax.plot(np.arange(Nz), Hy, label = 'H_y')

ax.set_xlim(0, Nz)
ax.set_ylim(-1.2, 1.2)

ax.set_xlabel('Posicion de la red')
ax.set_ylabel('Amplitud')

ax.set_title('Propagacion de ondas electromagneticas usando el metodo FDTD')
ax.legend()
ax.grid()

def update(frame):
    global Ex, Hy
    
    # 1. Actualizar campo H (usa los valores actuales de E)
    for k in range(Nz - 1):
        Hy[k] = Hy[k] - (dt / (mu0 * dz)) * (Ex[k + 1] - Ex[k])
        
    # 2. Actualizar campo E (usa los nuevos valores de H)
    for k in range(1, Nz):
        Ex[k] = Ex[k] - (dt / (eps0 * dz)) * (Hy[k] - Hy[k - 1])
        
    # 3. Condiciones de frontera (nodos extremos a cero)
    Ex[0] = 0
    Ex[-1] = 0
    
    # 4. Actualizar los datos de las líneas en el gráfico
    line1.set_ydata(Ex)
    line2.set_ydata(Hy)
    
    return line1, line2

# animacion
ani = FuncAnimation(fig, update, frames=Nt, interval = 30, blit=True)
plt.show()
