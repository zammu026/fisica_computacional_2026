# Este código resuelve un sistema de ecuaciones diferenciales ordinarias (EDO) utilizando el método de Runge-Kutta de cuarto orden (RK4).
# Define una función vectorial f(r, t) que representa el sistema de EDO.
# Integra numéricamente desde t=0 hasta t=10 con condiciones iniciales r=[1.0, 1.0].
# Grafica las soluciones x(t) y y(t) en función del tiempo t.

import matplotlib.pyplot as plt
import numpy as np
def  f(r,t):
	x = r[0]
	y = r[1]
	fx = x*y - x
	fy = y - x*y + np.sin(t)**2
	return np.array([fx, fy], float)

a = 0.0
b = 10.0
N = 1000
h = (b - a)/N

tpoints = np.arange(a,b,h)
xpoints = []
ypoints = []

r = np.array([1.0, 1.0], float)

for t in tpoints:
    xpoints.append(r[0])
    ypoints.append(r[1])
    
    k1 = h*f(r, t)
    k2 = h*f(r + k1/2, t + h/2)
    k3 = h*f(r + k2/2, t + h/2)
    k4 = h*f(r + k3, t + h)
    r += (k1 + 2*k2 + 2*k3 + k4)/6

plt.plot(tpoints, xpoints)
plt.plot(tpoints, ypoints)
plt.xlabel("t")
plt.show()

