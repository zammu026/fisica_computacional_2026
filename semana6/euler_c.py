# implementacion del método de Euler Cromer para simular un oscilador no lineal 
import numpy as np
import matplotlib.pyplot as plt
# parametros
k = 1.0
m = 1.0
alpha = 1.5
tmax = 20

x0 = 0.5
v0 = 0.0

N = 2000
t_values = np.linspace(0, tmax, N)

x = np.zeros(N)
v = np.zeros(N)

x[0], v[0] = x0, v0

# paso temporal
h = t_values[1] - t_values[0]
def fv(x):
    return -k*x/m + k*alpha*(x**2)/m

def fx(v):
    return v

# Integracian de euler c.
for i in range(N-1):
    v[i+1] = v[i] + h*fv(x[i])
    x[i+1] = x[i] + h*fx(v[i+1])

plt.plot(t_values, x, label='x(t)')
plt.plot(t_values, v, label='v(t)')
plt.xlabel('t')
plt.ylabel('x, v')
plt.legend()
plt.show()

plt.plot(x, v, label='Phase Space')
plt.xlabel('x')
plt.ylabel('v') 
plt.legend()
plt.show()
