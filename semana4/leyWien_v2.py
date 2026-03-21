x0 = 1.111  # Nota: 1111 estaba muy lejos de la raíz para este tipo de función
dx_step = 3.e-4 
eps = 0.002 
Nmax = 100

def f(x):
    # Uso de la función exponencial de math para mayor precisión
    import math
    return 5 * math.exp(-x) + x - 5

def NewtonR(x, dx, eps, Nmax):
    for it in range(Nmax):
        F = f(x)
        
        # Condición de parada (convergencia)
        if abs(F) <= eps:
            print(f"\nRoot found! x = {x:.6f}, f(x) = {F:.6e}")
            return x
            
        print(f"Iteration {it:2d}: x = {x:10.6f}, f(x) = {F:10.6e}")

        # Diferencia finita central para la derivada
        df = (f(x + dx/2) - f(x - dx/2)) / dx
        
        # EL ERROR ESTABA AQUÍ: 
        # No debes sobrescribir 'dx', usa una variable nueva para el paso
        delta_x = -F / df
        x += delta_x

    print(f"\nNewton failed to converge after {Nmax} iterations.")
    return x

root = NewtonR(x0, dx_step, eps, Nmax)
