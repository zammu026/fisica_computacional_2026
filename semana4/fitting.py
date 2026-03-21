import pylab as p
import numpy as np
from numpy import *
from numpy.linalg import inv, solve
import matplotlib.pyplot as plt

Nd = 7

A = zeros((3,3), float)
bvec = zeros((3,1), float)

ss = sx = sxx = sy = sxxx = sxxxx = sxy = sxxy = 0.0
x = array([1.0, 1.1, 1.24, 1.35, 1.451, 1.5, 1.92])
y = array([0.52, 0.8, 0.7, 1.8, 2.9, 2.9, 3.6])
sig = array([0.1, 0.1, 0.2, 0.3, 0.2, 0.1, 0.1])

xRange = arange(1.0, 2.0, 0.1)

p.plot(x, y, 'bo')
p.errorbar(x, y, sig)
#p.show()

for i in range(0, Nd):
    sig2 = sig[i] * sig[i]

    ss += 1./sig2
    sx += x[i]/sig2
    sy += y[i]/sig2

    rh1 = x[i] * x[i]

    sxx +=  rh1/sig2
    sxxy += rh1*y[i]/sig2
    sxy += x[i]*y[i]/sig2
    sxxx += rh1*x[i]/sig2
    sxxxx += rh1*rh1/sig2

A = array ([[ss, sx, sxx], [sx, sxx, sxxx], [sxx, sxxx, sxxxx]])
bvec = array([sy, sxy, sxxy])

# solve via matrix inverse
xvec = multiply(inv(A), bvec)
print('\n x via Inverse A \n', xvec, '\n')

# solve via gaussian elimination
xvec = solve(A, bvec)
print('\n x via Elimination \n', xvec)
print('\n Fit to Parabola \n')
print('y(x) = a0 + a1 x + a2 x a la 2')

print('a0 = ', xvec[0])
print('a1 = ', xvec[1])
print('a2 = ', xvec[2])
 
# Agregar
xt = np.linspace(x[0], x[-1], 100)
g = xvec[0] + xt* xvec[1] +  xvec[2]*(xt**2)

plt.scatter(x,y)
plt.plot(xt, g)
plt.errorbar(x,y,sig)
plt.show()
