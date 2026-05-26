"""
================================================================================
ENUNCIADO
================================================================================
Comparar numéricamente la evolución temporal de una cuerda vibrante utilizando el 
método de Leapfrog contra su solución analítica exacta obtenida mediante la suma 
de modos normales de Fourier. La simulación debe evaluar tanto la evolución de un 
pulso gaussiano genérico como el comportamiento armónico puro de un modo normal 
específico ante diferentes resoluciones o números de componentes (N = 50, 100, 200, 500).

================================================================================
RESOLVER
================================================================================
1. Configurar la malla espacial regular y verificar la estabilidad del esquema 
   temporal mediante el cálculo del parámetro de Courant.
2. Resolver la dinámica para un pulso gaussiano integrando la función `np.trapezoid` 
   para evaluar numéricamente los coeficientes de Fourier Bn de la solución analítica.
3. Proyectar comparativamente la curva numérica de Leapfrog contra la solución analítica 
   en un instante de tiempo fijo `t_val`.
4. Reinicializar el sistema aislando un armónico puro (modo normal `n_mode`) aplicando 
   la aproximación correcta de velocidad inicial cero.
5. Ejecutar de forma secuencial un bucle que simule el avance temporal para el modo 
   específico y grafique su respuesta frente a la solución analítica exacta de coseno.

================================================================================
CONSIDERACIONES IMPORTANTES
================================================================================
* RECONSTRUCCIÓN DEL HISTORIAL TEMPORAL (BUG DE FLUJO LOGÍSTICO):
  Al final de tu propuesta original modificabas la condición inicial con `y[1, :] = np.sin(...)` 
  e inmediatamente intentabas graficar `y[t_index, :]` sin haber vuelto a ejecutar el 
  bucle `for j in range(1, Nt - 1):`. Esto provocaba que la matriz contuviera puros 
  ceros en instantes avanzados. Cada vez que cambies la condición inicial, es mandatorio 
  volver a correr el algoritmo evolutivo de Leapfrog.

* VELOCIDAD INICIAL NULA EN ARMÓNICOS PUROS:
  Para el análisis del modo normal aislado, la asignación `y[0, :] = y[1, :]` inyectaba 
  un error en la velocidad del primer paso. Al igual que con el pulso gaussiano, se 
  debe fijar primero la geometría real en `y[0, :]` y deducir la capa `y[1, :]` mediante 
  la mitad de la aceleración del operador de onda para respetar la condición dy/dt = 0.
================================================================================
"""

# Metodo de leapfrog, modos normales (numerico vs analitico)
import numpy as np
import matplotlib.pyplot as plt

# parametros fisicos
L = 1.0
c = 1.0

# discretizacion
Nx = 200
dx = L / (Nx - 1)
dt = 0.004 # ajustar para cumplir Courant
Nt = 600
print((dt*c/dx)**2) # condicion de Courant

# estabilidad courant
courant = c * dt / dx
print(f"Courant: {courant}")

# malla espacial
x = np.linspace(0, L, Nx)

# inicializacion
y = np.zeros((Nt, Nx))

# condicion inicial
y[0, :] = np.exp(-100 * (x - 0.5)**2)
# para velocidad inicial cero
for i in range(1, Nx - 1):
    y[1, i] = y[0, i] + 0.5 * (courant**2) * (y[0, i+1] + y[0, i-1] - 2*y[0, i])

# evolucion temporal (leapfrog)
for j in range(1, Nt - 1):
    for i in range(1, Nx - 1):
        y[j + 1, i] = (2*y[j, i] - y[j - 1, i] + (courant**2) * (y[j, i + 1] + y[j, i - 1] - 2*y[j, i]))
    
    # condiciones de frontera
    y[j + 1, 0] = 0
    y[j + 1, -1] = 0

def fourier_modes(x, t, L, N):
    y_analitica = np.zeros_like(x)
    for n in range(1, N + 1):
        # CAMBIO: np.trapezoid en lugar de np.trapz para versiones nuevas de NumPy
        Bn = 2 * np.trapezoid(np.exp(-100 * (x - 0.5)**2) * np.sin(n * np.pi * x / L), x) / L
        wn = n * np.pi * c / L
        y_analitica += Bn * np.sin(n * np.pi * x / L) * np.cos(wn * t)
    return y_analitica

t_index = 200
t_val = t_index * dt
y_fourier = fourier_modes(x, t_val, L, 50)

# --- PRIMERA GRÁFICA: PULSO GAUSSIANO ---
plt.figure(figsize=(10, 5))
plt.plot(x, y[t_index, :], label = f"Leapfrog (numerico)")
plt.plot(x, y_fourier, '--', label = f"Modos (analiticos N=50)")
plt.xlabel("x")
plt.ylabel("y (x, t)")
plt.title(f"Evolución temporal del pulso gaussiano (t = {t_val:.3f} s)")
plt.legend()
plt.grid(True)
plt.show()

# --- SEGUNDA SECCIÓN: ANÁLISIS DE MODOS PUROS (N = 50, 100, 200, 500) ---
# Corregido: Se itera a través de los diferentes números de modos o frecuencias solicitadas en tu comentario
modos_a_evaluar = [50, 100, 200, 500]

for n_mode in modos_a_evaluar:
    # Reiniciar la matriz a cero
    y = np.zeros((Nt, Nx))
    
    # Corregido: Inicialización física correcta para velocidad inicial cero con el armónico puro
    y[0, :] = np.sin(n_mode * np.pi * x / L)
    for i in range(1, Nx - 1):
        y[1, i] = y[0, i] + 0.5 * (courant**2) * (y[0, i+1] + y[0, i-1] - 2*y[0, i])
        
    # Corregido: Se vuelve a ejecutar el solver de Leapfrog para calcular el avance temporal del modo actual
    for j in range(1, Nt - 1):
        for i in range(1, Nx - 1):
            y[j + 1, i] = (2*y[j, i] - y[j - 1, i] + (courant**2) * (y[j, i + 1] + y[j, i - 1] - 2*y[j, i]))
        y[j + 1, 0] = 0
        y[j + 1, -1] = 0

    # Solución analítica para este modo puro
    y_analitica_modo = np.sin(n_mode * np.pi * x / L) * np.cos(n_mode * np.pi * c * t_val / L)

    # Graficar la comparación para el modo actual
    plt.figure(figsize=(10, 4))
    plt.plot(x, y[t_index, :], label=f"Leapfrog Numérico (Modo {n_mode})")
    plt.plot(x, y_analitica_modo, '--', label=f"Analítico Exacto (Coseno)")
    plt.xlabel("x")
    plt.ylabel("y (x, t)")
    plt.title(f"Comparativa de Modos Normales - Armónico N = {n_mode} (t = {t_val:.3f} s)")
    plt.legend()
    plt.grid(True)
    plt.show()
