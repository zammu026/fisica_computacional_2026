import numpy as np
import matplotlib.pyplot as plt

def f(x,t):
	return -x**3 + np.sin(t)

#Intervalo
a = 0.0
b = 10.0
N = 1000
h = (b-a)/N

#Arrays
t = np.linspace(a, b, N)
x = np.zeros (N)

# condiciones inicial
x[0] = 0.0
# Metodo de euler
for n in range(N-1):
	x[n+1] = x[n] + h *f(x[n],t[n])
# Graficar
plt.plot(t,x)
plt.xlabel("t")
plt.ylabel("x(t)")
plt.show()
