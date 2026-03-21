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

# condiciones 
x[0] = 0.0

# Metodo RK4
for n in range(N-1):
	k1 = f(x[n], t[n])
	k2 = f(x[n] + 0.5*h*k1, t[n] + 0.5*h)
	k3 = f(x[n] + 0.5*h*k2, t[n] + 0.5*h)
	k4 = f(x[n] + h*k3, t[n] + h)

	x[n+1] = x[n] + (h/6.0)*(k1 + 2*k2 + 2*k3 +k4)
# Graficar
plt.plot(t,x)
plt.xlabel("t")
plt.ylabel("x(t)")
plt.show()
