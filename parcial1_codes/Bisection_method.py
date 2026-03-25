#Root Finding: Bisection Method_ vscode
#El método de bisección es un método numérico para encontrar raíces de funciones continuas
import numpy as np
import matplotlib.pyplot as plt
def f(x):
    return 2*np.cos(x) - x        

# Bisection Method
def bisection(a, b, tol):
    if f(a) * f(b) >= 0:
        print("The function must have different signs at the endpoints.")
        return None

    while (b - a) / 2 > tol:
        c = (a + b) / 2
        if f(c) == 0:
            return c
        elif f(a) * f(c) < 0:
            b = c
        else:
            a = c

    return (a + b) / 2
# Parameters
a = 1  # Start of interval
b = 2  # End of interval (cambio para signo diferente)
tol = 1e-5  # Tolerance for convergence
# Find root using bisection method
root = bisection(a, b, tol)
print(f"The root is approximately: {root}")
# Plotting the function
x = np.linspace(1, 4, 400)
y = f(x)
plt.plot(x, y, label='f(x) = 2cos(x) - x')
plt.axhline(0, color='red', lw=0.5, ls='--')
if root is not None:
    plt.scatter(root, f(root), color='green', label=f'Root ≈ {root:.5f}')
plt.title('Bisection Method')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()   
plt.grid()
plt.show()   