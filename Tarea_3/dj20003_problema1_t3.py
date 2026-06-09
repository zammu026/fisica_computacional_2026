"""
Práctica Numérica 3 - Ejercicio 1: Ondas Sonoras
Universidad de El Salvador - Física Computacional 2026

Resuelve las ecuaciones de Euler 1D con el esquema de Lax-Friedrichs.
Condición inicial: perturbación acústica pequeña (A = 2.1e-4).
Se grafica rho'/rho0, p'/p0, v/cs0 a distintos tiempos y se compara
con la solución analítica de onda sonora lineal.
"""

import numpy as np
import matplotlib.pyplot as plt

# PARÁMETROS FÍSICOS
gamma = 5.0 / 3.0          # índice adiabático (gas monoatómico ideal)
rho0  = 1.0                # densidad de fondo (estado base)
p0    = 1.0                # presión de fondo (estado base)
v0    = 0.0                # velocidad de fondo (medio en reposo)
A     = 2.1e-4             # amplitud de la perturbación (régimen lineal)

cs0 = np.sqrt(gamma * p0 / rho0)   # velocidad del sonido de fondo (ec. 11 con estado base)
print(f"Velocidad del sonido cs0 = {cs0:.6f}")

# DOMINIO ESPACIAL Y TEMPORAL
N    = 400                 # número de celdas en la malla
x0   = 0.0                 # extremo izquierdo del dominio
xf   = 1.0                 # extremo derecho del dominio
dx   = (xf - x0) / N      # tamaño de cada celda
x    = x0 + (np.arange(N) + 0.5) * dx   # posición del centro de cada celda

CFL   = 0.45               # número de Courant; debe ser < 1 para estabilidad
t_end = 2.0               # tiempo total de simulación
t_out = [0.0, 0.5, 1.0, 1.5, 2.0]   # tiempos en que se guardan los resultados para graficar

# CONVERSIÓN ENTRE PRIMITIVAS Y CONSERVATIVAS
def prim2cons(rho, p, v):
    """(rho, p, v) → U = [rho, rho*v, E]
    Convierte variables físicas a variables conservativas (ec. 6).
    La energía total E = rho*(epsilon + v²/2), con epsilon = p/((gamma-1)*rho)."""
    epsilon = p / ((gamma - 1.0) * rho)          # energía interna específica
    E = rho * (epsilon + 0.5 * v**2)             # energía total (ec. 4)
    return np.array([rho, rho * v, E])

def cons2prim(U):
    """U = [rho, rho*v, E] → (rho, p, v)
    Recupera las variables físicas a partir de las conservativas.
    Se usa la ecuación de estado del gas ideal (ec. 5)."""
    rho     = U[0]
    v       = U[1] / rho                         # velocidad = momento / densidad
    E       = U[2]
    epsilon = E / rho - 0.5 * v**2              # energía interna específica
    p       = (gamma - 1.0) * rho * epsilon      # ec. de estado (ec. 5)
    return rho, p, v

# FLUJOS F(U)  (ec. 7)
def flujos(U):
    """Calcula el vector de flujos F(U) de las ecuaciones de Euler.
    F = [rho*v, rho*v² + p, (E+p)*v]  (ec. 7)"""
    rho, p, v = cons2prim(U)
    E = U[2]
    F = np.zeros_like(U)
    F[0] = rho * v            # flujo de masa
    F[1] = rho * v**2 + p    # flujo de momento (incluye presión)
    F[2] = (E + p) * v       # flujo de energía
    return F

# PASO DE TIEMPO ADAPTATIVO (condición CFL)
def dt_CFL(U):
    """Calcula el paso de tiempo máximo permitido por la condición CFL (ec. 10).
    La velocidad de señal máxima es |v| + cs, donde cs es la vel. del sonido local."""
    rho, p, v = cons2prim(U)
    cs = np.sqrt(gamma * p / rho)                # velocidad del sonido local (ec. 11)
    lambda_max = np.max(np.abs(v) + cs)          # máxima velocidad de información
    return CFL * dx / lambda_max

# ESQUEMA LAX-FRIEDRICHS  (ec. 9) con fronteras periódicas
def lax_friedrichs(U, dt):
    """Aplica un paso del esquema de Lax-Friedrichs (ec. 9).
    Usa np.roll para obtener los vecinos j+1 y j-1 con periodicidad."""
    F   = flujos(U)
    jp1 = np.roll(U, -1, axis=1)   # desplaza el arreglo para obtener U_{j+1}
    jm1 = np.roll(U,  1, axis=1)   # desplaza el arreglo para obtener U_{j-1}
    Fp1 = np.roll(F, -1, axis=1)   # flujo en j+1
    Fm1 = np.roll(F,  1, axis=1)   # flujo en j-1
    return 0.5 * (jp1 + jm1) - (dt / (2.0 * dx)) * (Fp1 - Fm1)

# PERTURBACIÓN ACÚSTICA INICIAL  (ec. 14-17)
# w(x) es la forma de onda coseno definida en ec. 14
w = np.cos(2.0 * np.pi * (x - x0) / (xf - x0) + 5.0 * np.pi / 8.0)
rho_init = rho0 + A * rho0 * w          # ec. 15: perturbación de densidad
p_init   = p0   + A * gamma * p0 * w   # ec. 16: perturbación de presión
v_init   = v0   + A * cs0 * w          # ec. 17: perturbación de velocidad (onda en +x)
U = prim2cons(rho_init, p_init, v_init) # convertir a variables conservativas

# SOLUCIÓN ANALÍTICA LINEAL: onda plana viajando en +x
def solucion_analitica(t):
    """Solución exacta en el límite lineal: onda que viaja a velocidad cs0 en +x.
    La fase se traslada en el tiempo como k*(x - cs0*t)."""
    k = 2.0 * np.pi / (xf - x0)
    w = np.cos(k * (x - x0) - cs0 * t * k + 5.0 * np.pi / 8.0)
    return rho0 + A * rho0 * w, p0 + A * gamma * p0 * w, v0 + A * cs0 * w

# INICIALIZACIÓN Y LOOP TEMPORAL
# Se guarda el estado en cada tiempo de t_out para comparar con la analítica
resultados = {}   # diccionario: tiempo → (rho, p, v)
t = 0.0
output_idx = 0

# Guardar condición inicial (t=0)
if abs(t - t_out[output_idx]) < 1e-12:
    resultados[t_out[output_idx]] = cons2prim(U)
    output_idx += 1

while t < t_end and output_idx < len(t_out):
    dt = dt_CFL(U)
    # Recortar dt para no sobrepasar el siguiente tiempo de salida
    if t + dt > t_out[output_idx]:
        dt = t_out[output_idx] - t
    U = lax_friedrichs(U, dt)
    t += dt
    # Si llegamos a un tiempo de salida, guardar el estado
    if abs(t - t_out[output_idx]) < 1e-10:
        resultados[t_out[output_idx]] = cons2prim(U)
        output_idx += 1

# DETERMINACIÓN NUMÉRICA DE LA VELOCIDAD: seguimos el pico de densidad
# El pico se desplaza una distancia Δx en Δt = 1.0, dando v = Δx/Δt
i0 = np.argmax(resultados[0.0][0] - rho0)   # índice del pico en t=0
i1 = np.argmax(resultados[1.0][0] - rho0)   # índice del pico en t=1
desplazamiento = (x[i1] - x[i0]) % (xf - x0)   # módulo por periodicidad
v_numerica = desplazamiento / 1.0               # Δt = 1.0

print(f"\nVelocidad de propagación numérica  = {v_numerica:.6f}")
print(f"Velocidad del sonido teórica  cs0  = {cs0:.6f}")
print(f"Error relativo                     = {abs(v_numerica - cs0)/cs0*100:.3f} %")

# GRÁFICAS
# Línea sólida: solución numérica; línea punteada: solución analítica
colores = ['navy', 'royalblue', 'seagreen', 'darkorange', 'crimson']
fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
fig.suptitle("Ejercicio 1: Ondas Sonoras  (A = 2.1×10⁻⁴)", fontsize=14)

for i, t_plot in enumerate(t_out):
    rho_p, p_p, v_p = resultados[t_plot]
    rho_a, p_a, v_a = solucion_analitica(t_plot)

    lbl = f"t = {t_plot:.1f}"
    axes[0].plot(x, (rho_p - rho0) / rho0, color=colores[i], lw=1.8, label=f"Num. {lbl}")
    axes[0].plot(x, (rho_a - rho0) / rho0, '--', color=colores[i], lw=1.0, alpha=0.7,
                 label=f"Anal. {lbl}")
    axes[1].plot(x, (p_p - p0) / p0, color=colores[i], lw=1.8)
    axes[1].plot(x, (p_a - p0) / p0, '--', color=colores[i], lw=1.0, alpha=0.7)
    axes[2].plot(x, v_p / cs0, color=colores[i], lw=1.8)
    axes[2].plot(x, v_a / cs0, '--', color=colores[i], lw=1.0, alpha=0.7)

axes[0].set_ylabel(r"$(ρ - ρ_0)/ρ_0$", fontsize=12)
axes[1].set_ylabel(r"$(p - p_0)/p_0$", fontsize=12)
axes[2].set_ylabel(r"$v / c_{s0}$", fontsize=12)
axes[2].set_xlabel("x", fontsize=12)
axes[0].legend(fontsize=8, ncol=2)
for ax in axes:
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()