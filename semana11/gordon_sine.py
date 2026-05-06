import numpy as np
import matplotlib.pyplot as plt

# parametros 
Nx, Ny = 50, 50
h = 1.0
omega = 1.5 # sobrerrelajacion

#campos
vx = np.zeros((Nx, Ny))
P = np.zeros((Nx, Ny))

# condicion de frontera
vx[:, 0] = 1.0 # flujo de entrada

# iteraciones
for iteraciones in range(500):
    for i in range(1, Nx - 1):
        for j in range(1, Ny - 1):
            # laplaciano directo
            laplace = (
                vx[i+1, j] + vx[i-1, j] +
                vx[i, j+1] + vx[i, j-1]
                - 4*vx[i, j]
            ) / h**2
            # residual simple
            r = laplace - (P[i+1,j] - P[i-1, j]) / (2*h)
            #actualizacion SOR
            vx[i, j] = vx[i, j] + omega*r

#visusalizacion
plt.imshow(vx, origin= 'lower')
plt.colorbar(label = 'vx')
plt.title('Flujo aprox de Navier Stokes')
plt.show()