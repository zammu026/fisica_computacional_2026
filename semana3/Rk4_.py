import numpy as np
import matplotlib.pyplot as plt

def rk4(t, h, y, f):
        k1 = h*f(t,y)
        k2 = h*f(t+h/2,y+k1/2)
        k3 = h*f(t+h/2,y+k2/2)
        k4 = h*f(t+h,y+k3)

        return y+(k1 + 2*(k2+k3) + k4)/6

def f(t, y):
	x = y[0]
	v = y[1]
	dxdt = v
	dvdt = -(k/m)*x - (mu/m)*v
	return np.array([dxdt, dvdt], float)


# Parametros fisicos
m = 1.0 #masa
k = 10.0 #constante del resorte
mu = 0.5 # coef de friccion

#tiempo
t0 = 0.0
tf = 20.0
N = 2000
h = (tf - t0)/N

# condiciones inicial
y = np.array([0.2, 0.0], float) #x(0) = 0.2, v(0) = 0

t = t0
tpoints = []
xpoints = []

for i in range(N):
	tpoints.append(t)
	xpoints.append(y[0])
	y = rk4(t, h, y, f)
	t += h

plt.plot(tpoints, xpoints)
plt.xlabel("t")
plt.ylabel("x(t)")
plt.title("Sistema maso resorte con friccion RK4")
plt.grid()
plt.show()
