'''
ESTADOS LIGADOS
Determinar numericamente las energias permitidas de una particula confinada en un pozo cuadrado
unidimencional utilizando un algoritmo de busqueda de raices. Este problema uede abrodado con algunos
de los siguientes metodos numericos:
- Estados ligados
- ecuaciones transcendentales
- metodos numericos de busqueda de raices
- preparacion de metodos de disparo

POZO CUADRADO FINITO

'''

# Bisection.py: Matplotlib, 0 of f(x) via  Bisection algorithm
import math
from numpy import *
import matplotlib.pyplot as plt

eps = 1e-3; Nmax = 100; a = 0.0; b = 7.0 # Presision, [a,b]

v0 = 15

# funcion par
def f(x):
    return sqrt(v0-x)*tan(sqrt(v0-x)) - sqrt(x) # your function here

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
print(f"The root PAR of Eb is approximately: {root}")

if root is not None:
    # 1. Crear datos para la curva
    x_vals = linspace(a, b, 400)
    y_vals = f(x_vals)

    # 2. Configurar la gráfica con Matplotlib
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_vals, label='f(Eb)', color='blue') # Curva
    plt.axhline(0, color='black', linestyle='--', linewidth=1)         # Eje X
    
    # 3. Dibujar la raíz encontrada
    plt.plot(root, f(root), 'ro', label=f'Raíz approx: {root:.4f}') 
    
    # Detalles estéticos
    plt.title('Método de Bisección')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid(True)
    
    # Mostrar resultado
    plt.show()

# funcion impar
def f(x):
    return sqrt(v0-x)*(1/tan(sqrt(v0-x))) - sqrt(x) # your function here

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
print(f"The root IMPAR of Eb is approximately: {root}")

if root is not None:
    # 1. Crear datos para la curva
    x_vals = linspace(a, b, 400)
    y_vals = f(x_vals)

    # 2. Configurar la gráfica con Matplotlib
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_vals, label='f(Eb)', color='blue') # Curva
    plt.axhline(0, color='black', linestyle='--', linewidth=1)         # Eje X
    
    # 3. Dibujar la raíz encontrada
    plt.plot(root, f(root), 'ro', label=f'Raíz approx: {root:.4f}') 
    
    # Detalles estéticos
    plt.title('Método de Bisección')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid(True)
    
    # Mostrar resultado
    plt.show()
