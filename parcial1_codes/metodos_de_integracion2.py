import numpy as np

# --- FUNCIONES DE INTEGRACIÓN ---

def trapecio(f, a, b, n):
    x = np.linspace(a, b, n + 1)
    h = (b - a) / n
    y = f(x)
    return h * (0.5 * y[0] + np.sum(y[1:n]) + 0.5 * y[n])

def simpson(f, a, b, n):
    if n % 2 != 0: n += 1  # Simpson requiere n par
    x = np.linspace(a, b, n + 1)
    h = (b - a) / n
    y = f(x)
    return (h/3) * (y[0] + y[n] + 4*np.sum(y[1:n:2]) + 2*np.sum(y[2:n-1:2]))

def montecarlo(f, a, b, N_pts):
    x = np.random.uniform(a, b, int(N_pts)) # Forzamos entero para evitar errores
    return (b - a) * np.mean(f(x))

def romberg(f, a, b, n_iter):
    R = np.zeros((n_iter, n_iter))
    for k in range(n_iter):
        R[k, 0] = trapecio(f, a, b, 2**k) # 2**k subintervalos
        for j in range(1, k + 1):
            R[k, j] = (4**j * R[k, j-1] - R[k-1, j-1]) / (4**j - 1)
    return R[n_iter-1, n_iter-1]

def integrar_gauss(f, a, b, n_gauss):
    x, w = np.polynomial.legendre.leggauss(n_gauss)
    xp = 0.5 * (b - a) * x + 0.5 * (b + a)
    wp = 0.5 * (b - a) * w
    return np.sum(wp * f(xp))

# --- CONFIGURACIÓN Y EJECUCIÓN ---

f = lambda x: 2*x**4 + 3*x**3 + 4*x**2 - 5*x
a, b = 0, 1

# Parámetros específicos para evitar saturación de memoria
n_trapecio_simpson = 1000
n_montecarlo = 100000
n_gauss = 10     # Con 10 puntos es exacto para polinomios de grado 19
n_romberg = 10   # NO subir de 15-20 para evitar colapso

print(f"Trapecio:   {trapecio(f, a, b, n_trapecio_simpson):.16f}")
print(f"Simpson:    {simpson(f, a, b, n_trapecio_simpson):.16f}")
print(f"Montecarlo: {montecarlo(f, a, b, n_montecarlo):.16f}")
print(f"Gauss:      {integrar_gauss(f, a, b, n_gauss):.16f}")
print(f"Romberg:    {romberg(f, a, b, n_romberg):.16f}")
