import numpy as np
import matplotlib.pyplot as plt
# parametros
k = 1.0
m = 1.0
alpha = 0.5
alpha1 = 1.5
tmax = 20

mu_k = 0.05
N_force = 1.0 # normal
F0 = mu_k * N_force

x0 = 0.5
v0 = 0.0

N = 2000
t_values = np.linspace(0, tmax, N)

x = np.zeros(N)
v = np.zeros(N)

x1 = np.zeros(N)
v1 = np.zeros(N)

x[0], v[0], x1[0], v1[0] = x0, v0, x0, v0

# paso temporal
h = t_values[1] - t_values[0]

def friction(v):
    return -F0 *np.sign(v)

def fv(x, v):
    return (-k*x + k*alpha*(x**2) + friction(v))/m

def fv1(x, v):
    return (-k*x + k*alpha1*(x**2)+ friction(v))/m

# Integracian de euler c.
for i in range(N-1):
    v[i+1] = v[i] + h*fv(x[i], v[i])
    x[i+1] = x[i] + h*v[i+1]

    v1[i+1] = v[i] + h*fv1(x1[i], v1[i])
    x1[i+1] = x[i] + h*v1[i+1]

plt.plot(t_values, x)
plt.plot(t_values, v)
plt.show()

plt.plot(x, v)
plt.show()
