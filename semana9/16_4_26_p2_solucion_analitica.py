# Metodo de leapfrog, modos normales (numerico vs analitico)
import numpy as np
import matplotlib.pyplot as plt

# parametros fisicos
L = 1.0
c = 1.0

# discretizacion
Nx = 200
dx = L / (Nx - 1)
dt = 0.004 # ajustar para cumplir Courant
Nt = 600
print((dt*c/dx)**2) # condicion de Courant

# estabilidad courant
courant = c * dt / dx
print(f"Courant: {courant}")

# malla espacial
x = np.linspace(0, L, Nx)

# inicializacion
y = np.zeros((Nt, Nx))

# condicion inicial
y[0, :] = np.exp(-100 * (x - 0.5)**2)
# para velocidad inicial cero
for i in range(1, Nx - 1):
    y[1, i] = y[0, i] + 0.5 * (courant**2) * (y[0, i+1] + y[0, i-1] - 2*y[0, i])

# evolucion temporal (leapfrog)
for j in range(1, Nt - 1):
    for i in range(1, Nx - 1):
        y[j + 1, i] = (2*y[j, i] - y[j - 1, i] + (courant**2) * (y[j, i + 1] + y[j, i - 1] - 2*y[j, i]))
    
    # condiciones de frontera
    y[j + 1, 0] = 0
    y[j + 1, -1] = 0

def fourier_modes(x, t, L, N):
    y_analitica = np.zeros_like(x)
    for n in range(1, N + 1):
        # CAMBIO: np.trapezoid en lugar de np.trapz para versiones nuevas de NumPy
        Bn = 2 * np.trapezoid(np.exp(-100 * (x - 0.5)**2) * np.sin(n * np.pi * x / L), x) / L
        wn = n * np.pi * c / L
        y_analitica += Bn * np.sin(n * np.pi * x / L) * np.cos(wn * t)
    return y_analitica

t_index = 200
t_val = t_index * dt
y_fourier = fourier_modes(x, t_val, L, 50)

plt.plot(x, y[t_index, :], label = f"Leapfrog (numerico)")
plt.plot(x, y_fourier, '--', label = f"Modos (analiticos)")
plt.xlabel("x")
plt.ylabel("y (x, t)")
plt.title(f"Evolucion temporal de una cuerda con Metodo de Leapfrog (courant = {courant:.2f})")
plt.legend()
plt.show()

# Hacerlo para diferentes Modos,por ejemplo N = 50, 100, 200, 500
n_mode = 3
y[1, :] = np.sin(n_mode * np.pi * x / L)
y[0, :] = y[1, :]

plt.plot(x, y[t_index, :])
plt.plot(x, np.sin(n_mode * np.pi * x / L) * np.cos(n_mode * np.pi * c * t_val / L), '--')
plt.show()