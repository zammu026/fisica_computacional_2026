import numpy as np
from numpy import zeros
from cmath import exp, pi
#import matplotlib
#matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

def dft(y):
    N = len(y)
    # para señales reales, solo necesitamos calcular la mitad de los coeficientes
    c = zeros(N//2 + 1, complex)

    for k in range(N//2 + 1):
        for n in range(N):
            c[k] += y[n] * exp(-2j * pi * k * n/N)
    return c

def idft(c):
    # N original basado en la longitud de c (asumiendo señal real)
    N = (len(c) - 1) * 2
    y = zeros(N, complex)
    for n in range(N):
        # Componente DC
        y[n] += c[0]
        # Componentes intermedias (se multiplican por 2 para compensar la mitad faltante)
        for k in range(1, N // 2):
            y[n] += 2 * (c[k] * exp(2j * pi * k * n / N)).real
        # Componente Nyquist
        y[n] += c[N // 2] * exp(2j * pi * (N // 2) * n / N)
        y[n] /= N
    return y


# Test data
N = 1000
t = np.linspace(0, 1, N)

# Signal composed of two frequencies
y = np.sin(2*np.pi*5*t) + 0.5*np.cos(2*np.pi*20*t)

# Apply DFT
c1 = np.fft.rfft(y)
c2 = dft(y)

# Frequencies
f = np.fft.fftfreq(N, t[1] - t[0])

# Apply inverse DFT to verify
y_reconstructed = idft(c2)

# Plotting
# plt.figure(figsize=(12, 5))

plt.plot(1, 2, 1)
# Plot only the positive frequencies for comparison
n_freq = len(c2)
f_positive = f[:n_freq]
plt.plot(f_positive, np.abs(c1[:n_freq]), label='NumPy FFT')
plt.plot(f_positive, np.abs(c2), label='Custom DFT', linestyle='--')
plt.xlim(0, 30)
plt.ylim(0, 500)
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Amplitud')
plt.legend()
plt.title('DFT Comparison')
plt.show()

plt.plot(1, 2, 2)
plt.plot(t, y, label='Original Signal')
plt.plot(t, np.abs(y_reconstructed), label='Reconstructed Signal', linestyle='--')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.legend()
plt.title('Signal Reconstruction via IDFT')
plt.show()
