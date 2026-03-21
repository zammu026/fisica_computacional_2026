from TDF import dft
import numpy as np
import matplotlib.pyplot as plt

# n de puntos
N = 1000
t = np.linspace(0,1,N)

# se;al compuesta por dos frecuencias
y = np.sin(2*np.pi*5*t) + 0.5*np.cos(2*np.pi*20*t)
# transformada de Furier
c1 = np.fft.fft(y)
# frecuencia
f = np.fft.fftfreq(N, t[1] - t[0])

# el del script de TDF
c2 = TDF.dft

plt.plot(f, np.abs(c1))
plt.plot(f, np.abs(c2))
plt.xlim(0, 30)
plt.ylim(0, 500)
plt.xlabel('Frecuencia')
plt.ylabel('Amplitud')
plt.show()
