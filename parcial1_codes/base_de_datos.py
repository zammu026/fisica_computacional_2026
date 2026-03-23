# LLAMAR A BASE DE DATOS Y PLOTEARLOS
import numpy as np
import matplotlib.pyplot as plt
D = np.loadtxt("semana1/datos_1.dat") #especificar en que carpeta se encuentra el archivo .dat
x,y = D[:, 0], D[:, 1]
plt.plot(x, y, label = "Datos")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Gráfica de Datos")
plt.legend()
# plt.savefig("parcial1_codes/fig1.png", dpi = 100)
plt.show()