# ODE: y'' = -100y - 2y' + 100 sin(3t)
import numpy as np
from rk4Algor import rk4Algor
import matplotlib.pyplot as plt
# parametros
Tstart = 0.0
Tend = 10.0
N = 1000
h = (Tend - Tstart)/N #paso temporal

# Condiciones iniciales
y = np.array([0.0, 0.0]) #y[0]=posicion   y[1]=velocidad
t = Tstart

# Funcion del lado derecho
def f(t,y):
	dydt = np.zeros(2)
	dydt[0] = y[1]
	dydt[1] = -100*y[0] - 2*y[1] + 100*np.sin(3*t)
	return dydt

# Integracion temporal
tt = []
yy = []

while t < Tend:
	tt.append (t)
	yy.append(y[0])

	y = rk4Algor(t,h,2,y,f) 	#y_new = y_old + delta
	t = t + h

# Graficar
plt.plot(tt,yy)
plt.xlabel("t")
plt.ylabel("y(t)")
plt.title("Solucion con rk4")
plt.show()
