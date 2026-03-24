#OSCILADOR ARMONICO NO LINEAL AMARTIGUADA FORZADA CON RK4, BATIDOS cuando wf es cercano a omega_0
import numpy as np
import matplotlib.pyplot as plt
# parametros
k = 1.0
m = 1.0
omega_0 = 1.0

mu = 0.001 #friccion viscosa
F_drive = 1.0
t_max = 100
N = 5000
t_values = np.linspace(0, t_max, N)
h = t_values[1] - t_values[0]

# condiciones iniciales
x0 = 0.0
v0 = 0.0

def acceleration(t, x, v, alpha, omega_f):
    return -k*x/m + k*alpha*(x**2)/m - mu*v/m + F_drive*np.cos(omega_f*t)/m

def rk4(alpha, omega_f):
    x = np.zeros(N)
    v = np.zeros(N)
    x[0] = 0.0
    v[0] = 0.0
    for i in range(N-1):
        t = t_values[i]
        k1_x = v[i]
        k1_v = acceleration(t, x[i], v[i], alpha, omega_f)
        k2_x = v[i] + 0.5*h*k1_v
        k2_v = acceleration(t + 0.5*h, x[i] + 0.5*h*k1_x, v[i] + 0.5*h*k1_v, alpha, omega_f)
        k3_x = v[i] + 0.5*h*k2_v
        k3_v = acceleration(t + 0.5*h, x[i] + 0.5*h*k2_x, v[i] + 0.5*h*k2_v, alpha, omega_f)
        k4_x = v[i] + h*k3_v
        k4_v = acceleration(t + h, x[i] + h*k3_x, v[i] + h*k3_v, alpha, omega_f)
        x[i+1] = x[i] + (h/6.0)*(k1_x + 2*k2_x + 2*k3_x + k4_x)
        v[i+1] = v[i] + (h/6.0)*(k1_v + 2*k2_v + 2*k3_v + k4_v)
    return x, v

# Graficar
omega_f = 1.005 # frecuencia de la fuerza externa oscilatoria
F_drive = 5.0 # amplitud de la fuerza de externa
x_ml, v_ml = rk4(alpha=0.00, omega_f=omega_f)
plt.plot(t_values, x_ml)
plt.xlabel("Tiempo (s)")
plt.ylabel("Posición (m)")
plt.title("Oscilador Armónico No Lineal Amortiguada Forzada con RK4, mode Locking wf = 2, fuerza grande")
plt.grid()
plt.show()

# Diagrama de fase
plt.plot(x_ml, v_ml)
plt.xlabel("Posición (m)")
plt.ylabel("Velocidad (m/s)")
plt.title("Oscilador Armónico No Lineal Amortiguada Forzada con RK4, mode Locking wf = 2, fuerza grande")
plt.grid()
plt.show()

# Batidos wf  aprox omega_0