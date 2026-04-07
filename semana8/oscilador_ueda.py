# Osciladpor de Duffing, condiciones iniciales: x0 = 1.0, v0 = 0.0, y0 = 0.5, mu = 1.0
import numpy as np
import matplotlib.pyplot as plt
# parametros

# RK4
def rk4_step(f, t, y, h, params):
    k1 = f(t, y, params)
    k2 = f(t + h/2, y + h/2 * k1, params)
    k3 = f(t + h/2, y + h/2 * k2, params)
    k4 = f(t + h, y + h * k3, params)
    return y + (h/6) * (k1 + 2*k2 + 2*k3 + k4)

def duffing(t, y, p):
    x, v = y
    dxdy = v
    dvdt = - 2 * p['gamma'] * v - p['alpha'] * x - p['beta'] * x**3 + p['F'] * np.cos(p['omega'] * t) # diccionario de parametros
    return np.array([dxdy, dvdt])
def simulate(f, y0, params, t_max, h):
    N = int(t_max / h)
    t = np.zeros(N)
    Y = np.zeros((N, len(y0)))
    y = y0.copy() # evitar modificar el vector original
    for i in range(N):
        t[i] = i * h
        Y[i] = y
        y = rk4_step(f, t[i], y, h, params)
    return t, Y
# Duffing
p1 = {'alpha': 0.0, 'beta': 1.0, 'gamma': 0.04, 'F': 5.0, 'omega': 1.0}
T = 2 * np.pi / p1['omega']
t_trans = 100 * T
t_total = t_trans + 50 * T

t, Y = simulate(duffing, np.array([0.009, 0.0]), p1, t_total, 0.01)
# eliminar transiente
mask = t > t_trans
x = Y[mask, 0]
v = Y[mask, 1]

plt.figure(figsize=(12, 6))
plt.plot(t[mask], x, label='x(t)')
plt.plot(t[mask], v, label='v(t)')
plt.xlabel('t')
plt.ylabel('x, v')
plt.legend()
plt.title('Oscilador de Duffing')
plt.grid(True)
plt.show()

plt.figure(figsize=(6, 6))
plt.plot(x, v)
plt.xlabel('x')
plt.ylabel('v')
plt.title('Diagrama de fase del oscilador de Duffing')
plt.grid(True)
plt.show()