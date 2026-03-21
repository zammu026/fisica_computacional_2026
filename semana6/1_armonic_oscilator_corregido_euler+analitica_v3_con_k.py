import numpy as np
import matplotlib.pyplot as plt

def oscilador_analitico(t, x0, v0, k, m):
    """Solución teórica exacta."""
    omega = np.sqrt(k / m)
    # Asumiendo v0 = 0 para simplificar la amplitud y fase
    return x0 * np.cos(omega * t)

def oscilador_euler_simple(t, x0, v0, k, m, dt, **kwargs): # **kwargs allows the function to accept any additional arguments without error in output
    """Método de Euler estándar (Inestable)."""
    x = np.zeros(len(t))
    v = np.zeros(len(t))
    x[0], v[0] = x0, v0
    
    for i in range(len(t) - 1):
        # Derivadas actuales
        dxdt = v[i]
        dvdt = -(k / m) * x[i]
        
        # Actualización Euler Simple
        x[i+1] = x[i] + dxdt * dt
        v[i+1] = v[i] + dvdt * dt
    return x

# --- Configuración ---
params = {'x0': 1.0, 'v0': 0.0, 'k': 10.0, 'm': 1.0, 'dt': 0.05, 't_max': 10.0}
t = np.arange(0, params['t_max'], params['dt'])

# Cálculo de ambas soluciones
x_teorica = oscilador_analitico(t, **{k: params[k] for k in ('x0', 'v0', 'k', 'm')})
x_euler = oscilador_euler_simple(t, **params)

# --- Visualización ---
plt.figure(figsize=(12, 5))

# Gráfica Comparativa
plt.subplot(1, 2, 1)
plt.plot(t, x_teorica, 'k-', label='Analítica (Exacta)', lw=2)
plt.plot(t, x_euler, 'r--', label='Euler Simple', alpha=0.8)
plt.title('Analítico vs Euler')
plt.legend()
plt.grid(True)

# Gráfica del Error
plt.subplot(1, 2, 2)
error = np.abs(x_teorica - x_euler)
# plt.fill_between(t, error, color='red', alpha=0.3)
plt.plot(t, error, color='red')
plt.title('Error Absoluto (Crece con el tiempo)')
plt.ylabel('Error (m)')
plt.grid(True)

plt.tight_layout()
plt.show()
