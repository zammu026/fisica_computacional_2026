import math

def f(x):
    return 5 * math.exp(-x) + x - 5

def newton_raphson(x_start, eps=1e-7, dx=1e-5, n_max=100):
    x = x_start
    for i in range(n_max):
        f_val = f(x)
        
        # Condición de éxito
        if abs(f_val) < eps:
            return x
        
        # Derivada numérica (Diferencia central)
        df = (f(x + dx/2) - f(x - dx/2)) / dx
        
        if df == 0: break # Evitar división por cero
        
        x = x - f_val / df
    return None

# 1. Buscamos la raíz cerca de 0 (Valor inicial: -0.5 o 0.5)
raiz1 = newton_raphson(x_start=0.5)

# 2. Buscamos la raíz cerca de 5 (Valor inicial: 10 o 5)
raiz2 = newton_raphson(x_start=10.0)

print(f"Raíz 1 encontrada: {raiz1:.8f}")
print(f"Raíz 2 encontrada: {raiz2:.8f}")
