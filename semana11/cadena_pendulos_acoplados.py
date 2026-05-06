# ecuacion de gordon-sine
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
Vx[:, 0] = 4