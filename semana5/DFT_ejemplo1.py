from DFT import dft
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# n de puntos
N = 1000
t = np.linspace(0,1,N)

# señal compuesta por dos frecuencias
y = np.sin(2*np.pi*5*t) + 0.5*np.cos(2*np.pi*20*t)
# transformada de Fourier
c1 = np.fft.fft(y)
# frecuencia
f = np.fft.fftfreq(N, t[1] - t[0])

# el del script de DFT
c2 = dft(y)

# Plot only the positive frequencies for comparison
n_freq = len(c2)
f_positive = f[:n_freq]

plt.figure(figsize=(10, 6))
plt.plot(f_positive, np.abs(c1[:n_freq]), label='NumPy FFT', linewidth=2)
plt.plot(f_positive, np.abs(c2), label='Custom DFT', linestyle='--', linewidth=2)
plt.xlim(0, 30)
plt.ylim(0, 500)
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Amplitud')
plt.legend()
plt.title('Comparación DFT: NumPy vs Custom')
plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.savefig('dft_ejemplo1_result.png', dpi=100)
# print("Plot saved as 'dft_ejemplo1_result.png'")
plt.show()
