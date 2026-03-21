from math import cos

x0 = 1.111; dx_step = 3.e-4; eps = 0.002; Nmax = 100;

def f(x):
    return 2*cos(x) - x

def NewtonR(x, dx, eps, Nmax):
    for it in range(0, Nmax + 1):
        F = f(x)
        print(f"Iteración {it}: x = {x:.6f}, f(x) = {F:.6f}")
        
        if (abs(F) <= eps):
            print("\nRaíz encontrada, f(root) =", F)
            return x
        
        # Derivada numérica (f'(x))
        df = (f(x + dx/2) - f(x - dx/2)) / dx
        
        # Incremento de Newton: delta_x = -f(x)/f'(x)
        incremento = -F / df
        x += incremento

    print("\nNewton falló para Nmax =", Nmax)
    return x

root = NewtonR(x0, dx_step, eps, Nmax)
