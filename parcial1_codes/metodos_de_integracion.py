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

# Definir función e intervalo
f = lambda x: x**2 # f(x) = x^2, def f(x): return x**2
a, b = 0, 1
n = 1000
N = 100000

print(f"Trapecio:   {trapecio(f, a, b, n):.6f}")
print(f"Simpson:    {simpson(f, a, b, n):.6f}")
print(f"Montecarlo: {montecarlo(f, a, b, N):.6f}")
print(f"Exacto:     {1/3:.6f}")  # integral exacta de x^2 entre 0 y 1