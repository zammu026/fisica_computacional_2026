'''
=====================================================================
ENUNCIADO DEL PROBLEMA Y CONTEXTO FÍSICO-MATEMÁTICO
=====================================================================
El presente código aborda el cálculo y la simulación del potencial 
eléctrico Phi(z) generado por una distribución de carga en forma de 
anillo plano, o una configuración geométrica equivalente que involucra 
simetría axial. 

La solución analítica de este sistema físico requiere el uso de la 
Integral Elíptica Completa de Primera Especie, denotada como K(m).

1. Implementar el cálculo de la integral elíptica K(m) por dos vías:
   a) Solución numérica directa: Utilizando integración por el método 
      de los trapecios en una discretización de la variable theta.
   b) Aproximación polinomial: Empleando un desarrollo truncado de 
      alta precisión basado en los coeficientes de Hastings.
2. Evaluar el error numérico máximo entre ambos métodos para validar 
   la precisión del algoritmo y la aproximación matemática.
3. Modelar el potencial eléctrico Phi(z) a lo largo del eje 'z'.
4. Comparar gráficamente el comportamiento del potencial Phi(z) 
   frente a la aproximación de una carga puntual (1/r) en el límite 
   asintótico (grandes distancias).

CONSIDERACIONES:
- El parámetro de entrada 'm' de la integral elíptica se encuentra 
  restringido estrictamente al intervalo abierto (0, 1). El código 
  evalúa un rango seguro entre 0.001 y 0.999 para evitar divergencias.
- Las variables físicas 'V' (potencial de referencia) y 'a' (radio de 
  la geometría) están normalizadas de forma predeterminada con valor 1.
- La aproximación polinomial utiliza el complementario m1 = 1 - m, e 
  incorpora un término logarítmico para modelar la singularidad en m=1.
=====================================================================
'''

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------
# Integral eliptica comppleta K(m)
# -----------------------------------

def elliptic_K(m, N= 10000):
	theta = np.linspace(0,np.pi/2,N)
	integrand= 1.0/np.sqrt(1-m*np.sin(theta)**2)
	K= np.trapezoid(integrand, theta)

	return K

# ------------------------------------
# Aproximacion polinomial
# -----------------------------------

def K_approx(m):
	m1=1-m

	a0=1.3862944
	a1=0.1119723
	a2= 0.0725296
	b0=0.5
	b1=0.1213478
	b2=0.0288729

	# CORREGIDO: Se agregó np.log(1/m1) o -np.log(m1), y la potencia correcta m1**2 al final
	return (a0 + a1*m1 + a2*m1**2 - (b0 + b1*m1 + b2*m1**2) * np.log(m1))

# Valores de m

m_vals= np.linspace(0.001,0.999,200)

K_num= []
K_poly=[]

for m in m_vals:
	K_num.append(elliptic_K(m))
	K_poly.append(K_approx(m))

K_num= np.array(K_num)
K_poly=np.array(K_poly)

# Error absoluto
error= np.abs(K_num-K_poly)
print("Error maximo =", np.max(error))
'''
# Graficas
plt.figure(figsize=(8,5))

plt.plot(m_vals, K_poly, '--', label= 'Aprox. polinomial')

plt.xlabel('m')
plt.ylabel('K(m)')
plt.legend()

plt.grid()
plt.show()
'''

# Potencial Phi(z)
def Phi(z,V=1,a=1):
	k=2*a/np.sqrt(z**2 + 4*a**2)
	K= elliptic_K(k**2)
	return(V/2*(1-(k*z)/(np.pi*a)*K))

# Valores de z
z_vals= np.linspace(0.05,10,300)

phi_vals= []

for z in z_vals:
	phi_vals.append(Phi(z))
phi_vals=np.array(phi_vals)

# Comparacion con 1/r

plt.figure(figsize=(8,5))
plt.plot(z_vals, phi_vals, label=r'$\Phi(z)$') # Modificado ligeramente para que muestre el símbolo griego Phi en el gráfico

plt.plot(z_vals, 1/z_vals, '--', label=r'$1/r$')

plt.xlabel('z')
plt.ylabel('potencial')

plt.legend()
plt.grid()
plt.show()
