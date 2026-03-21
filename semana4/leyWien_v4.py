import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return 5 * np.exp(-x) + x - 5

def encontrar_raices():
    # 1. Configuración de búsqueda
    puntos_inicio = [-1.0, 7.0]  # Uno cerca de cada sospecha de raíz
    raices = []
    dx = 1e-5
    eps = 1e-7

    # 2. Algoritmo de Newton-Raphson
    for x in puntos_inicio:
        for _ in range(100):
            f_val = f(x)
            if abs(f_val) < eps:
                raices.append(x)
                break
            # Derivada numérica
            df = (f(x + dx/2) - f(x - dx/2)) / dx
            x = x - f_val / df
    
    return raices

# --- Generación de la Gráfica ---
raices_halladas = encontrar_raices()
x_range = np.linspace(-1, 8, 400)
y_range = f(x_range)

plt.figure(figsize=(10, 6))
plt.plot(x_range, y_range, label='$f(x) = 5e^{-x} + x - 5$', color='blue', lw=2)
plt.axhline(0, color='black', linestyle='-', alpha=0.3) # Eje X
plt.axvline(0, color='black', linestyle='-', alpha=0.3) # Eje Y

# Dibujar las raíces encontradas
for i, r in enumerate(raices_halladas):
    plt.plot(r, f(r), 'ro', markersize=8)
    plt.annotate(f'Raíz {i+1}: {r:.4f}', (r, 0.5), textcoords="offset points", 
                 xytext=(0,10), ha='center', fontweight='bold', color='red')

plt.title('Búsqueda de Raíces mediante Newton-Raphson', fontsize=14)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()

print(f"Raíces encontradas: {raices_halladas}")
