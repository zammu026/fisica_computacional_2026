'''
Resuelve la ecuacion de Schrodinger estacionaria:
1. Integracion Runge-Kutta de cuarto orden
2. Empalme de funciones de onda izquierda y derecha
3. Metodo de biseccion para encontrar autovalores


from numpy import *
import numpy as np
import matplotlib.pyplot as plt

# m/(hbar*c)**2 = 0.4829
eps = 1e-1
Nsteps = 501
h = 0.04
Nmax = 100

E = -17
Emax = 1.1 * E
Emin = E / 1.1

def f(x, y):
    global E
    F = zeros((2), float)
    F[0] = y[1] # d/dx phi
    F[1] = -(0.4829) * (E - V(x)) * y[0] # d**2/dx**2 phi = -(0.4829) * (E - V(x)) * phi
    return F

def V(x):
    if (abs(x) < 10):
        return (-16.0)
    else:
        return 0

def rk4Algor(t, h, N, y, f):
    k1 = np.zeros(N)
    k2 = np.zeros(N)
    k3 = np.zeros(N)
    k4 = np.zeros(N)
    
    k1 = h*f(t,y)
    k2 = h*f(t+h/2,y+k1/2)
    k3 = h*f(t+h/2,y+k2/2)
    k4 = h*f(t+h,y+k3)

    y = y+(k1 + 2*(k2+k3) + k4)/6
    
    return y

def diff(h, E):
    y = zeros((2), float)

    i_match = Nsteps//3
    nL = i_match + 1

    y[0] = 1.E-15
    y[1] = y[0] * sqrt(- E * 0.4829)

    for ix in range(0, nL + 1):
        x = h * (ix - Nsteps/2)
        y = rk4Algor(x, h, 2, y, f)
    
    left = y[1]/y[0]

    y[0] = 1.E-15
    y[1] = - y[0] * sqrt(- E * 0.4829)

    for ix in range(nL + 1, Nsteps + 1):
        x = h * (ix - Nsteps/2)
        y = rk4Algor(x, - h, 2, y, f)
    
    right = y[1]/y[0]

    return (left - right)/(left + right)

def plot(h):
    global xL, xR, Lwf, Rwf
    x = 0
    Lwf = []
    Rwf = []
    xR = []
    xL = []

    Nsteps = 1501

    y = zeros((2), float)
    yL = zeros((2, 505), float)

    i_match = 500
    nL = i_match + 1

    y[0] = 1.E-40
    y[1] = -sqrt(- E * 0.4829) * y[0]

    for ix in range(0, nL + 1):
        yL[0] [ix] = y[0]
        yL[1] [ix] = y[1]
        x = h * (ix - Nsteps/2)
        y = rk4Algor(x, h, 2, y, f)
    
    y[0] = -1.E-15
    y[1] = -sqrt(- E * 0.4829) * y[0]

    for ix in range(Nsteps - 1, nL + 2, -1):
        x = h * (ix + 1 - Nsteps/2)
        y = rk4Algor(x, - h, 2, y, f)
        xR.append(x)
        Rwf.append(y[0])
    x = x - h
    normL = y[0]/yL[0][nL]

    for ix in range(0, nL + 1):
        x = h*(ix - Nsteps/2 + 1)
        y[0] = yL[0][ix] * normL
        y[1] = yL[1][ix] * normL
        xL.append(x)
        Lwf.append(y[0])

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111)
ax.grid()

for count in range(0, Nmax):
    E = (Emax - Emin)/2
    Diff = diff(h, E)

    Etemp = E
    E = Emax
    diffMax = diff(h, E)
    E = Etemp

    if diffMax * Diff > 0:
        Emax = E
    else:
        Emin = E
    print('Iteration, E = ', count, E)

    if abs(Diff) < eps:
        break

    if count > 3:
        fig.clear()

        plot(h)

        plt.plot(xL, Lwf)
        plt.plot(xR, Rwf)
        plt.text(3, -200, 'Energy = %10.4f' % E, fontsize=14)
        plt.pause(0.8)

plt.xlabel('x')
plt.ylabel(r'$\psi(x)$', fontsize=18)
plt.title('R & L wavefunctions matched at x = 0')

print('Final eigenvalue E = ', E)
print('Iterations = ', count, ' max = ', Nmax)

plt.show()
'''

# CORREGIDO
from numpy import *
import numpy as np
import matplotlib.pyplot as plt

# m/(hbar*c)**2 = 0.4829
eps = 1e-1
Nsteps = 501
h = 0.04
Nmax = 100

E = -17
Emax = 1.1 * E
Emin = E / 1.1

def f(x, y):
    global E
    F = zeros((2), float)
    F[0] = y[1] # d/dx phi
    F[1] = -(0.4829) * (E - V(x)) * y[0] # d**2/dx**2 phi = -(0.4829) * (E - V(x)) * phi
    return F

def V(x):
    if (abs(x) < 10):
        return (-16.0)
    else:
        return 0

def rk4Algor(t, h, N, y, f):
    k1 = np.zeros(N)
    k2 = np.zeros(N)
    k3 = np.zeros(N)
    k4 = np.zeros(N)
        
    k1 = h*f(t,y)
    k2 = h*f(t+h/2,y+k1/2)
    k3 = h*f(t+h/2,y+k2/2)
    k4 = h*f(t+h,y+k3)

    y = y+(k1 + 2*(k2+k3) + k4)/6
        
    return y

def diff(h, E):
    y = zeros((2), float)

    i_match = Nsteps//3
    nL = i_match + 1

    y[0] = 1.E-15
    y[1] = y[0] * sqrt(- E * 0.4829)

    for ix in range(0, nL + 1):
        x = h * (ix - Nsteps/2)
        y = rk4Algor(x, h, 2, y, f)
        
    left = y[1]/y[0]

    y[0] = 1.E-15
    y[1] = - y[0] * sqrt(- E * 0.4829)

    # C1: El bucle debe ir de derecha a centro en reversa (de Nsteps hacia nL+1)
    for ix in range(Nsteps, nL, -1):
        x = h * (ix + 1- Nsteps/2)
        y = rk4Algor(x, - h, 2, y, f)
        
    right = y[1]/y[0]

    return (left - right)/(left + right)

def plot(h):
    global xL, xR, Lwf, Rwf
    x = 0
    Lwf = []
    Rwf = []
    xR = []
    xL = []

    Nsteps = 1501

    y = zeros((2), float)
    yL = zeros((2, 505), float)

    i_match = 500
    nL = i_match + 1

    y[0] = 1.E-40
    y[1] = -sqrt(- E * 0.4829) * y[0]

    for ix in range(0, nL + 1):
        yL[0] [ix] = y[0]
        yL[1] [ix] = y[1]
        x = h * (ix - Nsteps/2)
        y = rk4Algor(x, h, 2, y, f)
        
    y[0] = -1.E-15
    y[1] = -sqrt(- E * 0.4829) * y[0]

    for ix in range(Nsteps - 1, nL + 2, -1):
        x = h * (ix + 1 - Nsteps/2)
        y = rk4Algor(x, - h, 2, y, f)
        xR.append(x)
        Rwf.append(y[0])
    x = x - h
    normL = y[0]/yL[0][nL]

    for ix in range(0, nL + 1):
        x = h*(ix - Nsteps/2 + 1)
        y[0] = yL[0][ix] * normL
        y[1] = yL[1][ix] * normL
        xL.append(x)
        Lwf.append(y[0])

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111)
ax.grid()

for count in range(0, Nmax):
    # C2: El punto medio de la bisección es con suma (+), no con resta (-)
    E = (Emax + Emin)/2
    Diff = diff(h, E)

    Etemp = E
    E = Emax
    diffMax = diff(h, E)
    E = Etemp

    if diffMax * Diff > 0:
        Emax = E
    else:
        Emin = E
    print('Iteration, E = ', count, E)

    if abs(Diff) < eps:
        break

    if count > 3:
        fig.clear()
        
        plot(h)
        
        plt.plot(xL, Lwf)
        plt.plot(xR, Rwf)
        plt.text(3, -200, 'Energy = %10.4f' % E, fontsize=14)
        plt.pause(0.8)

plt.xlabel('x')
plt.ylabel(r'$\psi(x)$', fontsize=18)
plt.title('R & L wavefunctions matched at x = 0')

print('Final eigenvalue E = ', E)
print('Iterations = ', count, ' max = ', Nmax)

plt.show()
