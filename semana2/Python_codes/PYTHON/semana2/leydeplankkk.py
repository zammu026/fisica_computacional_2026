import sys
sys.path.append('/media/samuel/SAMU_2025/pythoncodes')
import integrales_impropias as ii
import numpy as np
#import cero_inf as ci
#import trapecio as trape

h = 6.626e-34  # Constante de Planck (J*s)
c = 3.0e8      # Velocidad de la luz (m/s)
k = 1.38e-23   # Constante de Boltzmann (J/K)
T = 5800      #k 

#f(z) =(1/(1-z)**2)((z/(1-z)**3)/(np.exp(z/(1-z))-1) 
C = 8*np.pi*(k*T)**4/(h*c)**3

resultado, error = ii.integral_0_inf(
    f = lambda x:C* x**3/(np.exp(x)-1)
)
#print(f"Energy: {ley_planck(frecuencia):.2e} W/(sr m^3)")
