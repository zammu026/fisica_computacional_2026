# Van der Pol oscillator, condiciones iniciales: x0 = 1.0, v0 = 0.0, y0 = 0.5, mu = 1.0
import numpy as np
import matplotlib.pyplot as plt
# parametros
omega0 = 1.0
# sistema de ecuaciones
def f(t, y, mu, x0):
    x, v = y
    dxdt = v
    dvdt = - mu * (x**2 - x0**2) * v - omega0**2 * x # dv/dt = - mu * (x^2 - x0^2) * v - omega0^2 * x
    return np.array([dxdt, dvdt])
# RK4
def rk4_step(t, y, h, mu, x0):
    k1 = f(t, y, mu, x0)
    k2 = f(t + h/2, y + h/2 * k1, mu, x0)
    k3 = f(t + h/2, y + h/2 * k2, mu, x0)
    k4 = f(t + h, y + h * k3, mu, x0)
    return y + (h/6) * (k1 + 2*k2 + 2*k3 + k4)

def simulate(mu, x0, t_max = 50, h = 0.01): # simula el sistema de Van der Pol para un valor dado de mu y x0
    N = int(t_max / h)
    t_values = np.zeros(N)
    xs = np.zeros(N)
    vs = np.zeros(N)
    y = np.array([0.5, 0.0])
    for i in range(N):
        t = i * h
        t_values[i] = i * h
        xs[i] = y[0]
        vs[i] = y[1]
        y = rk4_step(t, y, h, mu, x0)
    return t_values, xs, vs
plt.figure(figsize=(12, 6))
# mu > 0: mu = 1.0, x0 = 1.0, oscilaciones autosostenidas
# mu < 0: mu = -1.0, x0 = 1.0, amortiguamiento positivo
# mu = 0: mu = 0.0, x0 = 1.0, oscilador armónico simple
# mu >> 1: mu = 10.0, x0 = 1.0, oscilaciones no lineales fuertes, regimen fuertemente no lineal
# mu << 1: mu = 0.1, x0 = 1.0, oscilaciones casi armónicas, regimen bajamente no lineal
mu = 5 
# |x| < x0: amortiguamiento positivo (pierde energia)
# |x| > x0: amortiguamiento negativo (gana energia)
# |x| = x0: amortiguamiento nulo
x0 = 0.1
t_values, xs, vs = simulate(mu, x0)
plt.plot(t_values, xs, label='x(t)')
plt.plot(t_values, vs, label='v(t)')    
plt.xlabel('t')
plt.ylabel('x, v')
plt.legend()
plt.title('Van der Pol Oscillator')
plt.grid(True)
plt.show()

# Grafica de fase
plt.figure(figsize=(6, 6))
plt.plot(xs, vs)
plt.xlabel('x')
plt.ylabel('v')
plt.title('Diagrama de fase del oscilador de Van der Pol')
plt.grid(True)
plt.show()