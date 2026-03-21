import pylab as p
import numpy as np
from numpy import *
from numpy.linalg import inv, solve
import matplotlib.pyplot as plt

Nd = 24
# v(r) = Hr + b
A = zeros((2,2), float)
bvec = zeros((2,1), float)

ss = sx = sxx = sy = sxxx = sxxxx = sxy = sxxy = 0.0
x = array([0.0032, 0.9, 0.9, 0.275, 1.1, 0.5, 2.0, 0.8, 0.9, 0.214, 0.263, 1.1, 0.45, 1.7, 0.63, 2.0, 0.034, 2.0, 1.0, 0.275, 1.4, 0.5, 2.0, 0.9 ]) # r distancia
y = array([170, 150, 500, -185, 500, 290, 500, 300, 650, -130, -70, 450, 200, 960, 200, 800, 290, 1090, 920, -220, 500, 270, 850, -30]) # v velocidad

sig = array([300,300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300]) # sig = incertidumbre

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
xvec = multiply(inv(A), bvec)
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

plt.scatter(x,y)
plt.plot(xt, g)
plt.errorbar(x,y,sig)
plt.show()
#-----------------------------------------------------------

#for i in range(0, Nd):
#    chi = (g[i]-xt[i])**2/(sig2[i])**2
#print('El chi cuadrado es ', chi)
