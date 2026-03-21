from numpy import zeros
from cmath import exp, pi

def idft(c):
    N = (len(c) - 1)*2
    y = zeros(N, complex)

    for n in range(N):
        for k in range(N//2 + 1):
            y[n] += c[k] * exp(2j*pi*k*n/N)
        for k in range(1, N//2):
            y[n] += c[k].conjugate()*exp(-2j*pi*k*n/N)
        y[n] /= N
    return y


plt.plot(f, np.abs(c1))
plt.plot(f, c2)
plt.xlim(0, 30)
plt.ylim(0, 500)
plt.xlabel('Frecuencia')
plt.ylabel('Amplitud')
plt.show()
