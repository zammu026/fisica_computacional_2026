from numpy import zeros
from cmath import exp, pi

import numpy as np
import matplotlib.pyplot as plt
D = np.loadtxt("curva_luz_estrella.dat")
x,y = D[:, 0], D[:, 1]
plt.plot(x, y)
plt.savefig("fig2.png", dpi = 500)
#plt.show()

import numpy as np
import matplotlib.pyplot as plt
from TDF import dft
# n de puntos
N = 1000
t = np.linspace(0,1,N)

# transformada de Furier
#c1 = np.fft.fft(y)
# frecuencia
f = np.fft.fftfreq(N, t[1] - t[0])
c2 = dft(y)
# el del script de TDF
# c2 = TDF.dft

#plt.plot(f, np.abs(c1))
plt.plot(f, np.abs(c2))
plt.xlim(0, 10)
# plt.ylim(0, 500)
plt.xlabel('Frecuencia')
plt.ylabel('Amplitud')
plt.show()
