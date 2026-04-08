import os
# Silenciar advertencias de Qt/Wayland en Hyprland
os.environ["QT_LOGGING_RULES"] = "qt.qpa.wayland=false"
os.environ["QT_QPA_PLATFORM"] = "xcb"

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# parametros del sistema
a, b, c = 0.2, 0.2, 5.7

# sistema de Rossler
def rossler(t, state):
    x, y, z = state
    dxdt = -y - z
    dydt = x + a * y
    dzdt = b + z * (x - c)
    return np.array([dxdt, dydt, dzdt]) 

def rk4(f, t0, y0, h, N):
    t = np.zeros(N)
    y = np.zeros((N, len(y0)))

    t[0] = t0
    y[0] = y0
    for i in range(N - 1):
        k1 = f(t[i],           y[i])
        k2 = f(t[i] + h/2,     y[i] + h/2*k1)
        k3 = f(t[i] + h/2,     y[i] + h/2*k2)
        k4 = f(t[i] + h,       y[i] + h*k3)
        y[i+1] = y[i] + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t[i+1] = t[i] + h
    return t, y

# parametros de integracion
t0 = 0.0
tf = 200.0 # Aumentado para ver mejor el atractor
y0 = np.array([1.0, 1.0, 1.0])
h = 0.01
N = int((tf - t0) / h)

# resolver
t, sol = rk4(rossler, t0, y0, h, N)
x = sol[:, 0]
y = sol[:, 1]
z = sol[:, 2]

# Evaluacion temporal
plt.figure()
plt.plot(t, x, label='x(t)')
plt.plot(t, y, label='y(t)')
plt.plot(t, z, label='z(t)')
plt.xlabel('t')
plt.ylabel('x, y, z')
plt.title('Evolución Temporal de Rössler')
plt.legend()    
plt.show()

# Diagramas de fase 2D
plt.figure()
plt.plot(x, y, lw=0.5)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Diagrama de fase (x-y)')
plt.show()

# --- Gráfico de xdot vs x ---
# Calculamos la derivada de x según la ecuación del sistema: x_dot = -y - z
x_dot = -y - z

plt.figure(figsize=(8, 6))
plt.plot(x, x_dot, lw=0.5, color='darkgreen')
plt.xlabel('x')
plt.ylabel('dx/dt (x_dot)')
plt.title('Diagrama de Fase: $dx/dt$ vs $x$')
plt.grid(True, alpha=0.3)
plt.show()

# Atractor 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z, lw = 0.5, color='darkblue')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.title('Atractor de Rössler 3D')
plt.show()

# c) Sección de Poincaré
# Definimos la sección cuando x cruza 0 en una dirección (y + z = 0 aprox)
# Calculamos dxdt para todos los puntos para encontrar los cruces
dxdt_all = -y - z
# Buscamos índices donde dxdt cambia de signo y x es cercano a un valor (ej. x=0)
# Usaremos el cruce por el plano y = 0, que es más común en Rössler
indices = np.where((np.diff(np.sign(y)) > 0))[0] 

x_p = x[indices]
z_p = z[indices]

plt.figure()
plt.scatter(x_p, z_p, s=5, color='red')
plt.xlabel('x')
plt.ylabel('z')
plt.title('Sección de Poincaré (Plano y = 0)')
plt.show()

# --- VENTANA ÚNICA CON MÚLTIPLES PLOTS (2 filas x 3 columnas) ---
fig = plt.figure(figsize=(16, 9))
fig.suptitle('Sistema de Rössler', fontsize=16)

# 1. Evaluación Temporal
ax1 = fig.add_subplot(2, 3, 1)
ax1.plot(t, x, label='x(t)')
ax1.plot(t, y, label='y(t)')
ax1.plot(t, z, label='z(t)')
ax1.set_title('Series Temporales')
ax1.legend(fontsize='small')

# 2. Diagrama de Fase X-Y
ax2 = fig.add_subplot(2, 3, 2)
ax2.plot(x, y, lw=0.5)
ax2.set_title('Plano de Fase X-Y')

# 3. Gráfico xdot vs x
ax3 = fig.add_subplot(2, 3, 3)
ax3.plot(x, x_dot, lw=0.5, color='darkgreen')
ax3.set_title('Retrato de Fase $dx/dt$ vs $x$')

# 4. Atractor 3D
ax4 = fig.add_subplot(2, 3, 4, projection='3d')
ax4.plot(x, y, z, lw=0.5, color='purple')
ax4.set_title('Atractor de Rössler 3D')

# 5. Sección de Poincaré (Plano y=0)
indices = np.where((np.diff(np.sign(y)) > 0)) 
ax5 = fig.add_subplot(2, 3, 5)
ax5.scatter(x[indices], z[indices], s=3, color='red')
ax5.set_title('Sección de Poincaré ($y=0$)')

# 6. Diagrama de Fase Y-Z
ax6 = fig.add_subplot(2, 3, 6)
ax6.plot(y, z, lw=0.5, color='orange')
ax6.set_title('Plano de Fase Y-Z')

# Ajustar diseño para que no se traslapen
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()