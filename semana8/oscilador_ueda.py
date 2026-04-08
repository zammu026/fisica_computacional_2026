import numpy as np
import matplotlib.pyplot as plt

# RK4 - Runge-Kutta de 4to orden
def rk4_step(f, t, y, h, params):
    k1 = f(t, y, params)
    k2 = f(t + h/2, y + h/2 * k1, params)
    k3 = f(t + h/2, y + h/2 * k2, params)
    k4 = f(t + h, y + h * k3, params)
    return y + (h/6) * (k1 + 2*k2 + 2*k3 + k4)

# Definición del sistema (Oscilador de Ueda/Duffing)
def duffing(t, y, p):
    x, v = y
    dxdy = v
    # Ueda: alpha = 0, beta = 1
    dvdt = - 2 * p['gamma'] * v - p['alpha'] * x - p['beta'] * x**3 + p['F'] * np.cos(p['omega'] * t)
    return np.array([dxdy, dvdt])

# Función de simulación
def simulate(f, y0, params, t_max, h):
    N = int(t_max / h)
    t = np.linspace(0, t_max, N) # Mejor usar linspace para consistencia temporal
    Y = np.zeros((N, len(y0)))
    y = y0.copy() # evitar modificar el vector original
    for i in range(N):
        Y[i] = y
        y = rk4_step(f, t[i], y, h, params)
    return t, Y

# Parámetros específicos del Oscilador de Ueda para Caos
p1 = {
    'alpha': 0.0,   # Sin rigidez lineal (Ueda)
    'beta': 1.0,    # Rigidez cúbica
    'gamma': 0.05,  # Amortiguamiento (k=0.1 en 2*gamma)
    'F': 7.5,       # Fuerza de excitación
    'omega': 1.0    # Frecuencia
}

# Configuración de tiempo
T = 2 * np.pi / p1['omega']
t_trans = 100 * T # Tiempo para eliminar el transitorio
t_total = t_trans + 300 * T # Simulación larga para ver la estructura
h = 0.01 # Paso de integración

# Ejecución
t, Y = simulate(duffing, np.array([1.0, 0.0]), p1, t_total, h)

# Eliminar transiente
mask = t > t_trans
x = Y[mask, 0]
v = Y[mask, 1]

# --- Visualización ---

# Gráfico de series temporales
plt.figure(figsize=(12, 5))
plt.plot(t[mask], x, lw=0.7, label='x(t)')
plt.title('Series Temporales del Oscilador de Ueda (Estado Estacionario)')
plt.xlabel('t')
plt.ylabel('x')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

# Diagrama de fase
plt.figure(figsize=(8, 8))
plt.plot(x, v, lw=0.4, color='black')
plt.title('Diagrama de Fase: Atractor de Ueda')
plt.xlabel('x')
plt.ylabel('v')
plt.grid(True, alpha=0.3)
plt.show()

# Sección de Poincaré (Opcional: solo puntos en fase con el forzamiento)
poincare_mask = (t[mask] % T) < h
plt.figure(figsize=(8, 8))
plt.scatter(x[poincare_mask], v[poincare_mask], s=2, color='red')
plt.title('Sección de Poincaré (Atractor Extraño)')
plt.xlabel('x')
plt.ylabel('v')
plt.grid(True, alpha=0.3)
plt.show()