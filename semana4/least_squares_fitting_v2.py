import matplotlib.pyplot as plt

# Datos de ejemplo
x = [1, 2, 3, 4, 5]
y = [2.2, 3.8, 6.1, 7.9, 10.3]

n = len(x)

# Sumas necesarias para la ecuación normal
sum_x  = 0
sum_y  = 0
sum_xx = 0
sum_xy = 0

for i in range(n):
    sum_x  += x[i]
    sum_y  += y[i]
    sum_xx += x[i] * x[i]
    sum_xy += x[i] * y[i]

# Ecuación normal resuelta a mano:
# m = (n·Σxy - Σx·Σy) / (n·Σx² - (Σx)²)
# b = (Σy - m·Σx) / n
m = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x ** 2)
b = (sum_y - m * sum_x) / n

print(f"Pendiente (m): {m:.4f}")
print(f"Intercepto (b): {b:.4f}")
print(f"Línea ajustada: y = {m:.4f}x + {b:.4f}")

# Graficar
y_fit = [m * xi + b for xi in x]

plt.scatter(x, y, label="Datos", color="blue")
plt.plot(x, y_fit, label=f"Ajuste: y={m:.2f}x+{b:.2f}", color="red")
plt.legend()
plt.title("Least Squares Fitting (con for)")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()
