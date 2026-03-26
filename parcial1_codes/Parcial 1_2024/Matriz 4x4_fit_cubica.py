import numpy as np
import matplotlib.pyplot as plt
from google.colab import drive

drive.mount('/content/drive')

# Cargar datos
x, y = np.genfromtxt('/content/drive/My Drive/Datos python/datos_parcial.dat', unpack=True, skip_header=1)
x = np.array(x)
y = np.array(y)

# Definir función para sistema de tercer grado (Cúbica)
def generar_matriz_sistema_cubico(x, y):
    # Sumatorias necesarias para el sistema 4x4
    n = len(x)
    sx = np.sum(x)
    sx2 = np.sum(x**2)
    sx3 = np.sum(x**3)
    sx4 = np.sum(x**4)
    sx5 = np.sum(x**5)
    sx6 = np.sum(x**6)
    
    sy = np.sum(y)
    sxy = np.sum(x * y)
    sx2y = np.sum(x**2 * y)
    sx3y = np.sum(x**3 * y)

    # Matriz del sistema (Normal equations)
    A = np.array([
        [sx6, sx5, sx4, sx3],
        [sx5, sx4, sx3, sx2],
        [sx4, sx3, sx2, sx],
        [sx3, sx2, sx,  n]
    ])

    # Vector de resultados
    b = np.array([sx3y, sx2y, sxy, sy])
    
    return A, b

def eliminacion_gaussiana(A, b):
    n = len(b)
    A = A.astype(float)
    b = b.astype(float)
    for i in range(n):
        for j in range(i+1, n):
            factor = A[j, i] / A[i, i]
            A[j, i:] -= factor * A[i, i:]
            b[j] -= factor * b[i]
    
    res = np.zeros(n)
    for i in range(n-1, -1, -1):
        res[i] = (b[i] - np.dot(A[i, i+1:], res[i+1:])) / A[i, i]
    return res

# Ejecución
A, b_vec = generar_matriz_sistema_cubico(x, y)
coef = eliminacion_gaussiana(A, b_vec)
a, b, c, d = coef

print(f"Coeficientes: a={a}, b={b}, c={c}, d={d}")

# Graficar
x_fit = np.linspace(min(x), max(x), 100)
y_fit = a*x_fit**3 + b*x_fit**2 + c*x_fit + d

plt.scatter(x, y, label='Datos originales')
plt.plot(x_fit, y_fit, color='red', label='Ajuste Cúbico')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
