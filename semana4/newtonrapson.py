from math import cos

# Parámetros (Ajustamos x0 a un valor cercano a la raíz para asegurar convergencia)
x0 = 1.5; dx = 3.e-4; eps = 0.002; Nmax = 100;

def f(x):
    return 2*cos(x) - x

def NewtonR(x, dx_num, eps, Nmax):
    for it in range(0, Nmax + 1):
        F = f(x)
        
        # Condición de parada (Éxito)
        if (abs(F) <= eps):
            print(f"\n Root found at iteration {it}")
            print(f" x = {x:.6f}, f(x) = {F:.6f}, eps = {eps}")
            return x
            
        print(f"Iteration = {it:2} | x = {x:10.6f} | f(x) = {F:10.6f}")

        # Derivada numérica centrada (f'(x))
        df = (f(x + dx_num/2) - f(x - dx_num/2)) / dx_num
        
        # Salto de Newton: x_{n+1} = x_n - f(x_n)/f'(x_n)
        if df == 0: 
            print("Error: Derivada nula.")
            return None
            
        salto = -F / df 
        x += salto

        # Condición de fallo
        if it == Nmax:
            print(f"\n Newton Failed for Nmax = {Nmax}")
            
    return x

# Ejecutar e imprimir resultado
root = NewtonR(x0, dx, eps, Nmax)
print(f"\n Final root approximation: x = {root:.6f}, f(x) = {f(root):.6f}")


