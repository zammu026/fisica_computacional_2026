import numpy as np
import matplotlib.pyplot as plt

Nx = 60
Ny = 60

h = 0.4
omega = 0.1

g = 9.8
v0 = 8e-4

nu = 0.5

R = v0*h/nu
Niter = 4000

# geometria
Nb = 15
Ndown = 20

# matrices vacias
u = np.zeros((Nx+1, Ny+1), float)
w = np.zeros((Nx+1, Ny+1), float)

# condiciones de frontera
def BelowHole():
    for i in range(Nb+1, Nx+1):
        u[i, 0] = u[i-1, 1]
        w[i-1, 0] = w[i-1, 1]
        for j in range(0, Ndown+1):
            if i == Nb:
                vy = 0
            elif i == Nx:
                vy = -np.sqrt(2*g*h*(Ny+Nb-j))
            elif i == Nx-1:
                vy = -np.sqrt(2*g*h*(Ny+Nb-j))/2
            else:
                vy = 0
            u[i, j] = u[i-1, j] - vy*h

# condiciones de frontera
def BorderRight():
    for j in range(1, Ny+1):
        vy = -np.sqrt(2*g*h*(Ny-j))
        u[Nx, j] = u[Nx-1, j] + vy*h
        u[Nx, j] = u[Nx, j-1]
        w[Nx, j] = -2*(u[Nx, j] - u[Nx, j-1]) / h**2

def BottonBefore():
    for i in range(1, Nb+1):
        u[i, Ndown] = u[i, Ndown-1]
        w[i, down]

def Left():
    for j in range (Ndown, Ny):
        w[0, j] = -2*(u[0, j] - u[1, j])

def Relax():
    Borders()

    #resolver u
    for i in range(1, Nx):
        for j in range(1, Ny):
            r1 = omega*

# resolver w
for i in range(1, Nx):
    for j in range(1, Ny):
        a1 = (w[i+1, j]
              + w[i-1, j]
              + w[i, j+1]
              + w[i, j-1])
        a2 = ((u[i, j+1] - u[i, j-1])*
              )
        

# iteraciones
for it in range(Niter):
    Relax()
    if it % 100 == 0:
        print("Iteraciones =", it)

# velocidades
vx = np.zeros_like(u)
vy = np.zeros_like(u)

for i in range(1, )

# malla
x = np.arange(0, Nx+1)*h
y = np.arange(0, Ny+1)*h


# funcion corriente
plt.figure(figsize=(8,6))

plt.imshow(
    u.T, 
    origin = "lower"
    extent = [0, Nx*h, 0, Ny*h],
    aspect = "auto"

)