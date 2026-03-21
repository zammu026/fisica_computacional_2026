import numpy as np
import matplotlib.pyplot as plt
# parametros
# en el modelo 2 ya no hay alpha, cambia por p
k = 1.0
m = 1.0
p = 2.0
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
    return -k * np.sign(x) * np.abs(x)**(p - 1)
#    return -k*x**(p - 1)

def fx(v):
    return v

# Integracian de euler c.
for i in range(N-1):
    v[i+1] = v[i] + h*fv(x[i])
    x[i+1] = x[i] + h*fx(v[i+1])

plt.plot(t_values, x)
plt.plot(t_values, v)
plt.show()

plt.plot(x, v)
plt.show()
