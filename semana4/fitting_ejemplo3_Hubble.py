# Fitting a line to the Hubble data
import pylab as p
import numpy as np
from numpy import *
from numpy.linalg import inv, solve
import matplotlib.pyplot as plt
# para analizar el chi ^2
from scipy import stats 

Nd = 24
# v(r) = Hr + b
A = zeros((2,2), float)
bvec = zeros((2,1), float)

ss = sx = sxx = sy = sxy = 0.0
x = array([0.0032, 0.9, 0.9, 0.275, 1.1, 0.5, 2.0, 0.8, 0.9, 0.214, 0.263, 1.1, 0.45, 1.7, 0.63, 2.0, 0.034, 2.0, 1.0, 0.275, 1.4, 0.5, 2.0, 0.9 ]) # r distancia
y = array([170, 150, 500, -185, 500, 290, 500, 300, 650, -130, -70, 450, 200, 960, 200, 800, 290, 1090, 920, -220, 500, 270, 850, -30]) # v velocidad

sig = array([300]*24) # sig = incertidumbre

# Ajuste por minimos cuadrados
for i in range(0, Nd):
    sig2 = sig[i] * sig[i]

    ss += 1./sig2
    sx += x[i]/sig2
    sy += y[i]/sig2
    sxx += x[i]*x[i]/sig2
    sxy += x[i]*y[i]/sig2

A = array ([[ss, sx], [sx, sxx]])
bvec = array([sy, sxy])

# solve via matrix inverse
xvec = multiply(inv(A), bvec)
print('\n x via Inverse A \n', xvec, '\n')

# solve via gaussian elimination
xvec = solve(A, bvec)
a0 = xvec[0]
a1 = xvec[1]

print('a0 = ', a0)
print('a1 = ', a1)

# Visualización corregida
xt = np.linspace(min(x), max(x), 100) 
g = a0 + a1 * xt 

plt.errorbar(x, y, yerr=sig, fmt='bo', label='Datos')
plt.plot(xt, g, 'r', label='Ajuste lineal')
plt.xlabel('r (Distancia)')
plt.ylabel('v (Velocidad)')
plt.legend()
plt.show()
#-----------------------------------------------------------
# Cálculo de Chi cuadrado corregido
chi = 0.0
for i in range(0, Nd):
    # El chi cuadrado se mide sobre los puntos originales (observado - modelo)
    y_modelo = a0 + a1 * x[i]
    chi += (y[i] - y_modelo)**2 / (sig[i]**2)

print('El chi cuadrado es ', chi)
# --- NUEVA SECCIÓN ESTADÍSTICA ---

# 1. Grados de libertad (n_datos - n_parametros)
nu = Nd - 2 

# 2. Chi cuadrado reducido
chi_red = chi / nu

# 3. Valor p (Probabilidad de excedencia)
# Indica la probabilidad de obtener un chi cuadrado mayor al calculado si el modelo es correcto
p_valor = 1 - stats.chi2.cdf(chi, nu)

print('\n--- Resultados Estadísticos ---')
print('Chi cuadrado total: ', chi)
print('Grados de libertad (nu): ', nu)
print('Chi cuadrado reducido: ', chi_red)
print('Valor p (p-value): ', p_valor)

if p_valor > 0.05:
    print('\nInterpretación: El modelo lineal es estadísticamente aceptable.')
else:
    print('\nInterpretación: El modelo lineal podría no representar bien los datos.')