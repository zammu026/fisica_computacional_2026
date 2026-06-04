# metodo biseccion para estados ligados
import numpy as np
import matplotlib.pyplot as plt
import math

V0 = 25.0

def f_even(E):
    return np.sqrt(V0 - E) * np.tan(np.sqrt(V0 - E)) - np.sqrt(E)

def f_odd(E):
    return - np.sqrt(V0 - E) * (1/(np.tan(np.sqrt(V0 - E)))) - np.sqrt(E)

def Bisection(fun, Xminus, Xplus, Nmax = 100, eps = 1e-8):
    for it in range(Nmax):
        x = (Xplus + Xminus)/ 2.0
        if (fun(Xplus) * fun(x) > 0.0):
            Xplus = x
        else:
            Xminus = x
        if (abs(fun(x)) < eps):
            break
    return x

N = 10000
E = np.linspace(1.0e-5, V0 - 1.0e-5, N)

Feven = np.zeros(N)
Fodd = np.zeros(N)

for i in range(N):
    Feven[i] = f_even(E[i])
    Fodd[i] = f_odd(E[i])

roots_even = np.array([])
roots_odd = np.array([])

for i in range(N-1):
    if np.isfinite(Feven[i]) and np.isfinite(Feven[i+1]):
        if Feven[i] * Feven[i+1] < 0.0:
            r = Bisection(f_even, E[i], E[i+1])
            roots_even = np.append(roots_odd, r)

for i in range(N-1):
    if np.isfinite(Fodd[i]) and np.isfinite(Fodd[i+1]):
        if Fodd[i] * Fodd[i+1] < 0.0:
            r = Bisection(f_odd, E[i], E[i+1])
            roots_odd = np.append(roots_odd, r)

print(roots_even)
print(roots_odd)

# plotear
plt.figure(figsize = (9, 6))
plt.plot(E, Feven, label = 'Par')
plt.plot(E, Fodd, label = 'Impar')
plt.axhline(0, color = 'k', linestyle = '--')
plt.ylim(-15, 15)
plt.xlim(0, 25)
plt.plot(roots_even, np.zeros(len(roots_even)), 'ro')
plt.plot(roots_odd, np.zeros(len(roots_odd)), 'bo')
plt.legend()
plt.grid()
plt.show()

# CORREGIDO
# Metodo biseccion para estados ligados (Bound states)
import numpy as np
import matplotlib.pyplot as plt

V0 = 25.0

def f_even(E):
    # Se agrega un valor diminuto al argumento de la tangente para evitar división por cero
    arg = np.sqrt(V0 - E)
    return arg * np.tan(arg + 1e-15) - np.sqrt(E)

def f_odd(E):
    arg = np.sqrt(V0 - E)
    # Evitamos dividir entre 0 en la cotangente
    return - arg * (1 / (np.tan(arg) + 1e-15)) - np.sqrt(E)

def Bisection(fun, Xminus, Xplus, Nmax = 100, eps = 1e-8):
    for it in range(Nmax):
        x = (Xplus + Xminus)/ 2.0
        # Criterio de paro estándar
        if abs(Xplus - Xminus) < eps:
            break
        if (fun(Xplus) * fun(x) > 0.0):
            Xplus = x
        else:
            Xminus = x
    return x

N = 10000
# Reducimos el dominio un poco más en los bordes para evitar la raíz cuadrada de cero exactamente
E = np.linspace(1.0e-5, V0 - 1.0e-5, N)

Feven = np.array([f_even(e) for e in E])
Fodd = np.array([f_odd(e) for e in E])

roots_even = np.array([])
roots_odd = np.array([])

# Busqueda de raices: Par
for i in range(N-1):
    if np.isfinite(Feven[i]) and np.isfinite(Feven[i+1]):
        if Feven[i] * Feven[i+1] < 0.0:
            r = Bisection(f_even, E[i], E[i+1])
            # Se corrigió la variable append
            roots_even = np.append(roots_even, r)

# Busqueda de raices: Impar
for i in range(N-1):
    if np.isfinite(Fodd[i]) and np.isfinite(Fodd[i+1]):
        if Fodd[i] * Fodd[i+1] < 0.0:
            r = Bisection(f_odd, E[i], E[i+1])
            roots_odd = np.append(roots_odd, r)

print("Raíces pares:", roots_even)
print("Raíces impares:", roots_odd)

# Graficar
plt.figure(figsize = (9, 6))
plt.plot(E, Feven, label = 'Par')
plt.plot(E, Fodd, label = 'Impar')
plt.axhline(0, color = 'k', linestyle = '--')
plt.ylim(-15, 15)
plt.xlim(0, V0)

plt.plot(roots_even, np.zeros(len(roots_even)), 'ro', label='Raíces Par')
plt.plot(roots_odd, np.zeros(len(roots_odd)), 'bo', label='Raíces Impar')

plt.legend()
plt.grid()
plt.title("Estados ligados en pozo de potencial finito")
plt.xlabel("Energía E")
plt.ylabel("f(E)")
plt.show()
