import numpy as np
import matplotlib.pyplot as plt
from Rk4_ import rk4

# tiempos de vida media
T_Sr = 28.78 * 365.25 # dias
T_Y = 2.67 # dias

# constantes de decaimiento
lambda_Sr = np.log(2) / T_Sr
lambda_I = np.log(2) / T_Y

def f(t, y):
	N_Sr, N_Y = y
	dN_Sr = -lambda_Sr* N_Sr
	dN_Y = lambda_Sr * N_Sr - lambda_Y * N_Y
	return np.array([dN_Sr, dN_Y])

# intervalo temporal
t0, tmax = 1, 1e6 # dias
N = 20000
h = (tmax - t0)/N

t_vals = np.zeros(N)
y_vals = np.zeros((N, 2))

# condiciones inicial
y = np.array([1.0, 0.0]) # todo Sr, nada I
t = t0

for i in range(N):
	t_vals[i] = t
	y_vals[i] = y
	y = rk4(t, h, y, f)
	t += h

activity_Sr = lambda_Sr * y_vals[:,0]

plt.semilogx(t_vals, activity_Sr / activity_Sr[0])
plt.xlabel("Tiempo (dias)")
plt.ylabel("Actividad relativa de $^{90}$Sr")
plt.grid()
plt.show()

