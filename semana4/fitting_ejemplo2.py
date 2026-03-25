import pylab as p
import numpy as np
from numpy import *
from numpy.linalg import inv, solve
import matplotlib.pyplot as plt

Nd = 9

A = zeros((2,2), float)
bvec = zeros((2,1), float)

ss = sx = sxx = sy = sxxx = sxxxx = sxy = sxxy = 0.0
x = array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0]) # cm
y = array([7.3, 9.3, 18.3, 15.4, 30, 30, 31.1, 39.7, 49.8]) # Temperatura

sig = array([1, 1, 1, 1, 1, 1, 1, 1, 1]) # sig = incertidumbre

xRange = arange(1.0, 2.0, 0.1) # que hace eso

#p.plot(x, y, 'bo')
#p.errorbar(x, y, sig)
# p.show()

for i in range(0, Nd):
    sig2 = sig[i] * sig[i]

    ss += 1./sig2
    sx += x[i]/sig2
    sy += y[i]/sig2

    rh1 = x[i] * x[i]

    sxx +=  rh1/sig2
   # sxxy += rh1*y[i]/sig2
    sxy += x[i]*y[i]/sig2
   # sxxx += rh1*x[i]/sig2
   # sxxxx += rh1*rh1/sig2

A = array ([[ss, sx], [sx, sxx]])
bvec = array([sy, sxy])

# solve via matrix inverse
xvec = np.dot(inv(A), bvec)
print('\n x via Inverse A \n', xvec, '\n')

# solve via gaussian elimination
xvec = solve(A, bvec)
print('\n x via Elimination \n', xvec)
print('\n Fit to Parabola \n')
print('y(x) = a0 + a1 x')

print('a0 = ', xvec[0])
print('a1 = ', xvec[1])
#print('a2 = ', xvec[2])
 
# Agregar
xt = np.linspace(x[0], x[-1], 100)
g = xvec[0] + xt* xvec[1] 

# Calcular chi2
y_fit = xvec[0] + xvec[1]*x

chi2 = np.sum(((y - y_fit) / sig)**2)
dof = Nd - 2
chi2_red = chi2 / dof
print("chi2 =", chi2)
print("chi2/dof =", chi2_red)

# Grafica
plt.scatter(x, y, color='blue', label='Datos')
plt.errorbar(x, y, sig, fmt='none', ecolor='blue', capsize=3, label='Errores')
plt.plot(xt, g, color='red', label='Ajuste Lineal')
plt.legend()
plt.show()