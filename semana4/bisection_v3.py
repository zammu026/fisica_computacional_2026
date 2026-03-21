import math
import numpy as np
import matplotlib.pyplot as plt

# Parámetros iniciales
eps = 1e-3; Nmax = 100; a = 0.0; b = 7.0 

def f(x):
    # Usamos np.cos para que acepte arreglos de numpy al graficar
    return 2 * np.cos(x) - x 

def bisection(Xminus, Xplus, Nmax, eps): 
    if f(Xminus) * f(Xplus) > 0:
        print("Error: f(a) y f(b) deben tener signos opuestos.")
        return None

    for it in range(0, Nmax):
        x = (Xplus + Xminus) / 2
        if abs(f(x)) < eps:   
            print(f"\nRaíz encontrada: {x:.6f} en {it} iteraciones.")
            return x
        if f(Xplus) * f(x) > 0:
            Xplus = x   
        else:
            Xminus = x  
    return x

# Ejecutar algoritmo
root = bisection(a, b, Nmax, eps)

if root is not None:
    # 1. Crear datos para la curva
    x_vals = np.linspace(a, b, 400)
    y_vals = f(x_vals)

    # 2. Configurar la gráfica con Matplotlib
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_vals, label='f(x) = 2*cos(x) - x', color='blue') # Curva
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
