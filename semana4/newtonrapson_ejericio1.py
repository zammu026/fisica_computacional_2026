#from math import cos
# Graficar ademas, ejercicio  6.15
from matplotlib.pyplot as plt
x0 = 1111; dx = 3.e-4; eps = 0.002; Nmax = 100;
def f(x):
    return 924*x**6 - 2772*x**5 +  3150*x**4 - 1680*x**3 + 420*x - 42*x + 1

def NewtonR(x, dx, eps, Nmax):
    for it in range(0, Nmax + 1):
        F = f(x)
        if (abs(F) <= eps):
            print("\n Root found, f(root) =", F, ", eps =", eps)
            break
        print("Interation  =", it, "x = ", x, "f(x) =", F)

        df = (f(x + dx/2) - f(x - dx/2))/dx
        dx = -F/dx
        x += dx

        if it == Nmax + 1:
            print("\n Newton Failed for Nmax =", Nmax)
    return x

NewtonR(x0, dx, eps, Nmax)
