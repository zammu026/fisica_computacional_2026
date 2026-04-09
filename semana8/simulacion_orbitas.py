import numpy as np
import matplotlib.pyplot as plt

# RK4 manteniendo tus argumentos (t, h, N, y, f)
def rk4Algor(t, h, N, y, f):
    # k1, k2... se crean automáticamente como arrays al operar
    k1 = h * f(t, y)
    k2 = h * f(t + h/2, y + k1/2)
    k3 = h * f(t + h/2, y + k2/2)
    k4 = h * f(t + h, y + k3)
    
    y = y + (k1 + 2 * (k2 + k3) + k4)/6
    return y

# CORRECCIÓN: El orden de argumentos debe ser (t, y) para coincidir con la llamada
def f(t, y):
    # y[0]=x, y[1]=vx, y[2]=y_pos, y[3]=vy
    r = np.sqrt(y[0]**2 + y[2]**2)
    Fx = -y[0]/r**3
    Fy = -y[2]/r**3
    return np.array([y[1], Fx, y[3], Fy])

# Condiciones iniciales
y = np.array([0.5, 0.0, 0.0, np.sqrt(2.0)]) 
h = 0.001
t = 0.0
tmax = 20.0
N = 4

# Arrays para guardar
Nt = int(tmax/h)
x = np.zeros(Nt)
y_pos = np.zeros(Nt)

# Integración
for i in range(Nt):
    x[i] = y[0]
    y_pos[i] = y[2]
    y = rk4Algor(t, h, N, y, f)
    t += h

# Graficar
plt.plot(x, y_pos)
plt.xlabel("x")
plt.ylabel("y")
plt.axis('equal') # Recomendado para ver la forma real de la órbita
plt.show()
