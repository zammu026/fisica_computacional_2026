# Resolver la ecuacion del oscilador No Lineal
# LLamar a Euler de un script que hiciste hace mucho, X: + k X. + w_o^2 X = 0
# Sea v = dx/dt, dv/dt = -kv - w_o^2 x, para mayor facilidad sea k = 0
# import numpy as np
import numpy as np
import matplotlib.pyplot as plt

def dvdt(x): return - wo**2*x
#    return -x**3 + np.sin(t) #dx/dt = -x^3 + sin(t)
# dx/dt = f(x,t) IMPORTANTEEEEE
def dxdt(v): return v

wo = 0.8
#Intervalo
a = 0.0
b = 10.0
N = 1000
h = (b-a)/N

#Arrays
t = np.linspace(a, b, N)
x = np.zeros (N)
v = np.zeros (N)
# condiciones inicial
x0 = 1.0
v0 = 0.0
x[0] = x0
v[0] = v0
# Metodo de euler
for n in range(N-1):
    x[n+1] = x[n] + h * dxdt(v[n])
    v[n+1] = v[n] + h * dvdt(x[n])
#return x,v
# Graficar
plt.plot(t, x, label = 'Posicion (x)')
plt.plot(t, v, label = 'Velocidad (v)')
plt.title('Oscilador Armonico Simple - Metodo de Euler')
plt.xlabel("t")
plt.ylabel("x(t) = Amplitud")
plt.legend()
plt.grid(True)
plt.show()

# Compararlo con la solucion analitica
# --- SOLUCIÓN ANALÍTICA --- si k=0, x0 y v0 = 0, la solucion es x(t)= x0 cos(w0t)
# 
x_analitica = x0 * np.cos(wo * t)

# Graficar
# plt.figure(figsize=(10, 6))
plt.plot(t, x, 'r--', label='Euler (Numérica)')
plt.plot(t, x_analitica, 'g', label='Analítica (Exacta)', alpha=0.6)
plt.title("Comparación: Euler vs Solución Analítica")
plt.xlabel("Tiempo (t)")
plt.ylabel("Posición (x)")
plt.legend()
plt.grid(True)
plt.show()

# Gráfica del Error
# plt.subplot(1, 2, 2)
error = np.abs(x_analitica - x)
plt.plot(t, error, color='red', alpha=0.9)
plt.title('Error Absoluto')
plt.ylabel('Error (m)')
plt.xlabel('t')
plt.grid(True)
plt.show()
