# ALGORITMOS DE ECUACIONES DIFERENCIALES ORDINARIAS (EDO): RK4 y Euler
import numpy as np
import matplotlib.pyplot as plt

def f(x,t):
	return -x**15 + np.sin(t**2) # funcion a integrar f(x,t) = dx/dt = -x^15 + sin(t^2)
#Intervalo
a = 0.0
b = 10.0
N = 1000
h = (b-a)/N

#Arrays
t = np.linspace(a, b, N)
x = np.zeros (N)

# condiciones 
x[0] = 0.0

# Metodo RK4
for n in range(N-1):
	k1 = f(x[n], t[n])
	k2 = f(x[n] + 0.5*h*k1, t[n] + 0.5*h)
	k3 = f(x[n] + 0.5*h*k2, t[n] + 0.5*h)
	k4 = f(x[n] + h*k3, t[n] + h)

	x[n+1] = x[n] + (h/6.0)*(k1 + 2*k2 + 2*k3 +k4)

# Metodo Euler
x_euler = np.zeros(N)
x_euler[0] = 0.0
for n in range(N-1):
    x_euler[n+1] = x_euler[n] + h * f(x_euler[n], t[n])

# Graficar
plt.plot(t,x, label="RK4")
plt.plot(t,x_euler, label="Euler")
plt.xlabel("t")
plt.ylabel("x(t)")
plt.legend()
plt.show()