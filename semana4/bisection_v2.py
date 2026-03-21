import math

eps = 1e-3; Nmax = 100; a = 0.0; b = 7.0 

def f(x):
    return 2*math.cos(x) - x 

def bisection(Xminus, Xplus, Nmax, eps): 
    # Validar si el signo cambia en el intervalo inicial
    if f(Xminus) * f(Xplus) > 0:
        print("Error: f(a) y f(b) deben tener signos opuestos.")
        return None

    for it in range(0, Nmax):
        x = (Xplus + Xminus) / 2
        print(f" it = {it}, x = {x:.6f}, f(x) = {f(x):.6f}")

        if abs(f(x)) < eps:   
            print("\n Root found with precision eps =", eps)
            return x

        # Actualizar límites correctamente
        if f(Xplus) * f(x) > 0:
            Xplus = x   
        else:
            Xminus = x  

    print("\n No root after N iterations \n")
    return x

root = bisection(a, b, Nmax, eps)
