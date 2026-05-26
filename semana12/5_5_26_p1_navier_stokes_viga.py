import numpy as np
import matplotlib.pyplot as plt

Nx,Ny= 70,20
h= 1.0
nu= 1.0
omega= 0.1
V0= 1.0

u= np.zeros((Nx+1,Ny+1))
w= np.zeros((Nx+1,Ny+1))

# condiciones de frontera
def boundary():
    # Inicializacion
    for i in range(Nx+1):
        for j in range(Ny+1):
            w[i,j]=0.0
            u[i,j]=0.0
    
    # Superficie 
    for i in range(1, Nx):
        u[i,Ny]=u[i,Ny-1]+V0*h
        w[i,Ny]=0.0

    # Entrada
    for j in range(1,Ny):
        u[0,j]=u[1,j]
        w[0,j]=0.0
    
    # Salida
    for j in range(1,Ny):
        u[Nx,j]=u[Nx-1,j]
        w[Nx,j]=w[Nx-1,j]


def relax():
    for i in range(1,Nx):
        for j in range(1,Ny):
            r1= omega*((u[i+1,j]+u[i-1,j]+u[i,j+1]+u[i,j-1]-4*u[i,j])+h*h*w[i,j])
            u[i,j]+= r1

    for i in range(1,Nx):
        for j in range(1,Ny):
            a1= w[i+1,j]+ w[i-1,j]+w[i,j+1]+w[i,j-1]
            w[i,j]= 0.25*a1


# Iteraciones 
boundary()
for it in range(100):
    relax()
    if it%10==0:
        print("Iteracion:", it)

# Visualizacion
X,Y= np.meshgrid(range(Nx+1), range(Ny+1))
plt.contourf(X,Y,u.T)
plt.colorbar(label='Funcion de corriente')
plt.title('Flujo sobre la viga')
plt.xlabel('x')
plt.ylabel('y')
plt.show()