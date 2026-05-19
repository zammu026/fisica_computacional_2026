# Paquetes necesarios
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#parameters
N = 50
V = np.zeros((N, N))
V[0, :] = 100.0

tolerance = 10e-4
max_iterations = 10000

# codigo de gauss seidel
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

    #error maximo
    error =np.max(np.abs(V - Vold))
    if iteration % 100 == 0:
        print("Iteration: ", iteration, "Error: ", error)
    if error < tolerance:
        print("\nConvergencia alcanzada")
        break

x = np.arange(0, N, 1)
y = np.arange(0, N, 1)
X, Y = np.meshgrid(x, y)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection = '3d')
ax.plot_surface(X, Y, V, cmap = 'plasma')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('V(x, y)')
plt.show()
