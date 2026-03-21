import numpy as np
import matplotlib.pyplot as plt

# Datos de ejemplo
x = np.array([1, 2, 3, 4, 5])
y = np.array([2.2, 3.8, 6.1, 7.9, 10.3])

# Least squares: y = mx + b
# Resuelve el sistema usando la ecuación normal: (X^T X) a = X^T y
X = np.column_stack([x, np.ones(len(x))])  # matriz de diseño
m, b = np.linalg.lstsq(X, y, rcond=None)[0]

print(f"Pendiente (m): {m:.4f}")
print(f"Intercepto (b): {b:.4f}")
print(f"Línea ajustada: y = {m:.4f}x + {b:.4f}")

# Graficar
plt.scatter(x, y, label="Datos", color="blue")
plt.plot(x, m * x + b, label=f"Ajuste: y={m:.2f}x+{b:.2f}", color="red")
plt.legend()
plt.title("Least Squares Fitting")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()
