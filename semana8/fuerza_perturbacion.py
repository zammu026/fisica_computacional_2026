# sistema vec_F = (-k/r^6 + C/r^3) r_unitario
# casos: |C| < l^2/m, |C| = l^2/m, |C| > l^2/m
# Resolver numericamente, identificar estados ligandos y analizar precesion
# C es parametro que mide la intensidad de la perturbacion, C = 0 gravedad de Newton pura, C distino de 0 desviacion del comportamiento lineal

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


# COMIENZA EL CODE DE FUERZAS DE PERTURBACION
def f_grav_C(C):
    def f(t, y):
        r = np.sqrt(y[0]**2 + y[2]**2)
        Fx = - y[0]/ r**3 + C * y[0]/ r**4
        Fy = - y[2]/ r**3 + C * y[2]/ r**4
        return np.array([y[1], Fx, y[3], Fy])
    return f

def simular_C(y0, C, h = 0.001, tmax = 100):
    y = y0.copy()
    t = 0.0
    N = 4
    Nt = int(tmax/h)
    x = np.zeros(Nt)
    y_pos = np.zeros(Nt)
    f = f_grav_C(C)

    for i in range(Nt):
        x[i] = y[0]
        y_pos[i] = y[2]
        y = rk4Algor(t, h, N, y, f)
        t += h
    return x, y_pos
# condiciones iniciales
y0 = np.array([0.5, 0.0, 0.0, 1.5])
#valores de C
Cs = [0.008, 0.03, 0.0]
labels = ['C=0.008', 'C=0.03', 'C=0.0']

# Graficar
plt.figure()
for i in range(len(Cs)):
    C = Cs[i]
    lab = labels[i]
    x, y = simular_C(y0, C)
    plt.plot(x, y, linewidth = 1, label = lab)

plt.xlabel("x")
plt.ylabel("y")
plt.title('Orbitas con perturaciones 1/r³(presecion)')
plt.axis('equal') # Recomendado para ver la forma real de la órbita
plt.legend()
plt.show()