# Bisection.py: Matplotlib, 0 of f(x) via  Bisection algorithm
import math
from numpy import *
eps = 1e-3; Nmax = 100; a = 0.0; b = 7.0 # Presision, [a,b]

def f(x):
    return 2*math.cos(x) - x # your function here

def bisection(x_minus, x_plus, Nmax, eps): # dont change
    if f(x_minus) * f(x_plus) >= 0:
        raise ValueError("La función debe tener signos opuestos en los extremos del intervalo")

    for it in range(Nmax):
        x = (x_plus + x_minus) / 2
        print(" it =", it, " x =", x, " f(x) =", f(x))

        if f(x) == 0 or abs(f(x)) < eps:
            print("\n Root found with precision eps =", eps)
            return x

        if f(x_minus) * f(x) < 0:
            x_plus = x
        else:
            x_minus = x

    print("\n No root after N iterations \n")
    return x
root = bisection(a, b, Nmax, eps)
print(f"The root is approximately: {root}")