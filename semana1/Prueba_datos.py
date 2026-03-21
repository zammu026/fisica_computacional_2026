import numpy as np
import matplotlib.pyplot as plt
D = np.loadtxt("datos_1.dat")
x,y = D[:, 0], D[:, 1]
plt.plot(x, y)
plt.savefig("fig1.png", dpi = 500)
plt.show()

