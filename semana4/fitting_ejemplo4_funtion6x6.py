import numpy as np
from numpy import array, zeros, linspace, dot
from numpy.linalg import inv, solve
import matplotlib.pyplot as plt

# Datos originales
Nd = 24
x = array([0.0032, 0.9, 0.9, 0.275, 1.1, 0.5, 2.0, 0.8, 0.9, 0.214, 0.263, 1.1, 0.45, 1.7, 0.63, 2.0, 0.034, 2.0, 1.0, 0.275, 1.4, 0.5, 2.0, 0.9])
y = array([170, 150, 500, -185, 500, 290, 500, 300, 650, -130, -70, 450, 200, 960, 200, 800, 290, 1090, 920, -220, 500, 270, 850, -30])
sig = array([300]*24) # Incertidumbre constante

# Para orden 5, la matriz es de (n+1)x(n+1) = 6x6
A = zeros((6, 6), float)
bvec = zeros((6, 1), float)

# Sumatorias para extender la idea matricial
# Necesitamos potencias de x hasta x^10 para la matriz A (2*orden)
# y potencias de x^k * y hasta x^5 para el vector bvec
for i in range(Nd):
    sig2 = sig[i]**2
    for fila in range(6):
        # Llenar vector bvec: sum(x^fila * y / sig^2)
        bvec[fila] += (x[i]**fila * y[i]) / sig2
        for columna in range(6):
            # Llenar matriz A: sum(x^(fila+columna) / sig^2)
            A[fila, columna] += (x[i]**(fila + columna)) / sig2

# solve via matrix inverse
# Nota: xvec = inv(A) @ bvec es la forma correcta de multiplicar matrices en numpy
xvec_inv = dot(inv(A), bvec)
print('\n x via Inverse A \n', xvec_inv)

# solve via gaussian elimination (más estable numéricamente)
xvec = solve(A, bvec)
print('\n x via Elimination \n', xvec)

print('\n Fit to Polynomial Order 5 \n')
print(f'y(x) = {xvec[0][0]:.2f} + {xvec[1][0]:.2f}x + {xvec[2][0]:.2f}x^2 + {xvec[3][0]:.2f}x^3 + {xvec[4][0]:.2f}x^4 + {xvec[5][0]:.2f}x^5')

# Graficación
xt = linspace(min(x), max(x), 100)
# Evaluar polinomio: sum(a_i * x^i)
g = sum(xvec[i] * xt**i for i in range(6))

plt.errorbar(x, y, yerr=sig, fmt='bo', label='Datos')
plt.plot(xt, g, 'r-', label='Ajuste Polinomial (Orden 5)')
plt.legend()
plt.xlabel('Distancia (r)')
plt.ylabel('Velocidad (v)')
plt.title('Ajuste de Polinomio de Orden 5')
plt.show()
