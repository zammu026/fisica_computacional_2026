# Energia potencial de un oscilador no lineal con fuerza de tipo k*|x|^(p-1)*sign(x)
import numpy as np
import matplotlib.pyplot as plt

# Parámetros
k, m, p = 1.0, 1.0, 3.0  # p=2 sería armónico simple, p=3 es no lineal
tmax, N = 20, 2000       # Más puntos para mejor precisión
x0, v0 = 1.0, 0.0

t_values = np.linspace(0, tmax, N)
h = t_values[1] - t_values[0]
x, v = np.zeros(N), np.zeros(N)
x[0], v[0] = x0, v0

# Integración Euler-Cromer (mantiene la energía mejor que Euler simple)
for i in range(N-1):
    accel = -(k/m) * np.abs(x[i])**(p-1) * np.sign(x[i])
    v[i+1] = v[i] + h * accel
    x[i+1] = x[i] + h * v[i+1]

# Cálculo de Energías
kinetic = 0.5 * m * v**2          # K = 1/2 * m * v^2
potential = (k/p) * np.abs(x)**p  # U = k/p * |x|^p
total_energy = kinetic + potential

# Gráficas
plt.figure(figsize=(12, 5))

# Panel 1: Energías vs Tiempo
plt.subplot(1, 2, 1)
plt.plot(t_values, kinetic, label='Cinética (K)', alpha=0.7)
plt.plot(t_values, potential, label='Potencial (U)', alpha=0.7)
plt.plot(t_values, total_energy, '--', label='Total (E)', color='black', linewidth=2)
plt.title('Conservación de la Energía')
plt.xlabel('Tiempo')
plt.ylabel('Energía')
plt.legend()
plt.grid(True)

# Panel 2: Espacio de Fase
plt.subplot(1, 2, 2)
plt.plot(x, v, color='purple')
plt.title('Espacio de Fase')
plt.xlabel('Posición (x)')
plt.ylabel('Velocidad (v)')
plt.grid(True)
plt.tight_layout()
plt.show()