import numpy as np
import matplotlib.pyplot as plt

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
tf = 100.0
y0 = np.array([1.0, 1.0, 1.0])
h = 0.01
N = int((tf - t0) / h)

# resolver
t, sol = rk4(rossler, t0, y0, h, N)
x = sol[:, 0]
y = sol[:, 1]
z = sol[:, 2]

# derivada de x
#dxdt = (x[1:] - x[:-1]) / h
print("x:", x)
print("y:", y)
print("z:", z)
# Evaluacion temporal
plt.figure()
plt.plot(t, x, label='x(t)')
plt.plot(t, y, label='y(t)')
plt.plot(t, z, label='z(t)')
plt.xlabel('t')
plt.ylabel('x, y, z')
plt.legend()    
#plt.show()

#diagramas de fase
plt.figure()
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Diagrama de fase')
#plt.show()

# Plotear dxdt VS x
#plt.figure()
#plt.plot(dxdt, x)
#plt.xlabel('x')
#plt.ylabel('dxdt')    
#plt.show()    

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z, lw = 0.5)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.title('Atractor de Rossler')
plt.show()

# c) Mapa de poincare
indices = np.where(np.diff(np.sign(dxdt)))[0]
x_p = x[indices]
y_p = y[indices]

plt.figure()
plt.plot(x_p, y_p, s=5)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Seccion de Poincare')
plt.show()