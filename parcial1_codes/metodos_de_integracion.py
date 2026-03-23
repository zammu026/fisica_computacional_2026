import numpy as np

def trapecio(f, a, b, n):
    x = np.linspace(a, b, n+1)
    h = (b - a) / n
    y = f(x)
    I = h * (0.5*y[0] + np.sum(y[1:n]) + 0.5*y[n])
    return I

def simpson(f, a, b, n):
    if n % 2 == 1:
        n += 1
    x = np.linspace(a, b, n+1)
    h = (b - a) / n
    y = f(x)
    I = h/3 * (y[0] + y[n] + 4*np.sum(y[1:n:2]) + 2*np.sum(y[2:n-1:2]))
    return I

def montecarlo(f, a, b, N):
    x = np.random.uniform(a, b, N)
    I = (b - a) * np.mean(f(x))
    return I

def integrar_gauss(f, a, b, N_gauss):
    # 1. Obtener puntos (x) y pesos (w) para el intervalo estándar [-1, 1]
    x, w = np.polynomial.legendre.leggauss(N_gauss)
    # 2. Mapear los puntos del intervalo [-1, 1] al intervalo [a, b]
    # Fórmula: xp = 0.5 * (b - a) * x + 0.5 * (b + a)
    xp = 0.5 * (b - a) * x + 0.5 * (b + a)
    # 3. Ajustar los pesos al nuevo intervalo
    # Fórmula: wp = 0.5 * (b - a) * w
    wp = 0.5 * (b - a) * w
    # 4. Calcular la sumatoria (Integral ≈ Σ w_i * f(x_i))
    I = np.sum(wp * f(xp))
    return I

# Definir función e intervalo
f = lambda x: 2*x**4 + 3*x**3 + 4*x**2 - 5*x # f(x) = 2x^4 + 3x^3 + 4x^2 - 5x, def f(x): return 2*x**4 + 3*x**3 + 4*x**2 - 5*x
a, b = 0, 1
n = 1000
N = 100000
N_gauss = 100 # puntos para la Cuadratura de Gauss, deben de ser pocos sino la memoria colapsa

print(f"Trapecio:   {trapecio(f, a, b, n):.16f}")
print(f"Simpson:    {simpson(f, a, b, n):.16f}")
print(f"Montecarlo: {montecarlo(f, a, b, N):.16f}")
print(f"Gauss:      {integrar_gauss(f, a, b, N_gauss):.16f}")
# print(f"Exacto:     {1/5 + 1/2:.6f}")  # integral exacta de 2x^4 + 3x^3 + 4x^2 - 5x entre 0 y 1