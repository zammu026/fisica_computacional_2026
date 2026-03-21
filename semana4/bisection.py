# Bisection.py: Matplotlib, 0 of f(x) via  Bisection algorithm
import math
from numpy import *
eps = 1e-3; Nmax = 100; a = 0.0; b = 7.0 # Presision, [a,b]

def f(x):
    return 2*math.cos(x) - x # your function here

def bisection(Xmimus, Xplus, Nmax, eps): # dont change
    for it in range(0, Nmax):
        x = (Xplus + Xmimus)/2
        print(" it =", it, " x  =", x, " f(x) =", f(x))

        if (f(Xplus)*f(x) > 0):
            xplus = x   # change x+ to x
        else:
            Xminus = x  # change x- to x


        if (abs(f(x)) < eps):   # converge?
            print("\n Root found with precision eps =", eps)
            break

        if it == Nmax - 1:
            print("\n No root after N iterations \n")
    return x
root = bisection(a, b, Nmax, eps)
