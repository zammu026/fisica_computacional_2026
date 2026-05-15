import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

N = 50
V = np.zaros((N, N))
V[0, :] = 100.0

tolerance = 10e-4
max_iterations = 10000

for iteration in range(max_iterations):
    Vold = V.copy()
    for i in range(1, N-1):
        for j in range(1, N-1):

            V[i, j] = 0.25*(
                V[i+1, j] +
                V[i-1, j] +
                V[i, j+1] +
                V[i, j-1]
            )
