"""
Práctica Numérica 3 - Ejercicio 2: Movimiento del Medio (Efecto Doppler)
v0 = 0,  -cs0/8,  +cs0/2
"""

import numpy as np
import matplotlib.pyplot as plt

gamma = 5/3
rho0, p0 = 1.0, 1.0
A = 2.1e-4
cs0 = np.sqrt(gamma * p0 / rho0)   # velocidad del sonido de fondo

N = 400
x0, xf = 0.0, 1.0
dx = (xf-x0)/N
x  = x0 + (np.arange(N)+0.5)*dx   # centros de celda
CFL = 0.45

def prim2cons(rho, p, v):
    """Convierte (rho, p, v) → U = [rho, rho*v, E]."""
    E = rho*(p/((gamma-1)*rho) + 0.5*v**2)
    return np.array([rho, rho*v, E])

def cons2prim(U):
    """Recupera (rho, p, v) desde las variables conservativas U."""
    rho = U[0]; v = U[1]/rho
    p = (gamma-1)*rho*(U[2]/rho - 0.5*v**2)
    return rho, p, v

def flujos(U):
    """Calcula el vector de flujos F(U) = [rho*v, rho*v²+p, (E+p)*v]."""
    rho, p, v = cons2prim(U)
    return np.array([rho*v, rho*v**2+p, (U[2]+p)*v])

def dt_cfl(U):
    """Paso de tiempo adaptativo por condición CFL: dt = CFL*dx / max(|v|+cs)."""
    rho, p, v = cons2prim(U)
    return CFL*dx / np.max(np.abs(v)+np.sqrt(gamma*p/rho))

def lax(U, dt):
    """Un paso de Lax-Friedrichs con fronteras periódicas (np.roll)."""
    F = flujos(U)
    return 0.5*(np.roll(U,-1,1)+np.roll(U,1,1)) - dt/(2*dx)*(np.roll(F,-1,1)-np.roll(F,1,1))

def simular(v0_medio, t_end=1.5):
    """Ejecuta la simulación con un fondo de velocidad v0_medio.
    La perturbación acústica se superpone sobre ese fondo.
    Devuelve snapshots en t = 0, 0.5, 1.0, 1.5."""
    w   = np.cos(2*np.pi*(x-x0)/(xf-x0) + 5*np.pi/8)
    # La velocidad inicial es el fondo más la perturbación acústica
    U   = prim2cons(rho0+A*rho0*w, p0+A*gamma*p0*w, v0_medio+A*cs0*w)
    t_sal = [0.0, 0.5, 1.0, 1.5]
    snaps = {}
    t, idx = 0.0, 0
    if abs(t-t_sal[0]) < 1e-12:
        snaps[t_sal[0]] = cons2prim(U); idx += 1
    while t < t_end and idx < len(t_sal):
        dt = min(dt_cfl(U), t_sal[idx]-t)   # no sobrepasar el tiempo de salida
        U = lax(U, dt); t += dt
        if abs(t-t_sal[idx]) < 1e-10:
            snaps[t_sal[idx]] = cons2prim(U); idx += 1
    return snaps

# Los tres casos: medio en reposo, moviéndose en -x y en +x
casos = {"v₀=0": 0.0, "v₀=-cs₀/8": -cs0/8, "v₀=+cs₀/2": cs0/2}
res   = {nombre: simular(v0) for nombre, v0 in casos.items()}

# Velocidad aparente: la onda se propaga a cs0 + v0 (Doppler)
print("Velocidades aparentes (seguimiento del pico, Δt=1):")
for nombre, v0 in casos.items():
    i0 = np.argmax(res[nombre][0.0][0] - rho0)
    i1 = np.argmax(res[nombre][1.0][0] - rho0)
    v_ap = ((x[i1]-x[i0]) % (xf-x0)) / 1.0   # desplazamiento del pico en Δt=1
    print(f"  {nombre}: v_ap={v_ap:.5f}  (teórica cs0+v0={cs0+v0:.5f})")

# Cuando v0 = -cs0 la onda parece estacionaria (velocidad neta = 0)
print(f"\nOnda estacionaria cuando v0 = -cs0 = {-cs0:.5f}")

colores = ['navy','seagreen','crimson']
t_vals  = [0.0, 0.5, 1.0, 1.5]

# Gráfica 1: densidad relativa en los 4 tiempos, comparando los 3 casos
fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
fig.suptitle("Ejercicio 2: Efecto Doppler — densidad relativa")
for idx_t, tv in enumerate(t_vals):
    ax = axes[idx_t//2][idx_t%2]
    ax.set_title(f"t = {tv:.1f}")
    for ci, nombre in enumerate(casos):
        ax.plot(x, (res[nombre][tv][0]-rho0)/rho0, color=colores[ci], lw=1.8, label=nombre)
    ax.set_ylabel(r"$(ρ-ρ_0)/ρ_0$"); ax.set_xlabel("x")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Gráfica 2: posición del pico en el tiempo para cada caso
# La pendiente de cada curva es la velocidad aparente de la onda
fig2, ax2 = plt.subplots(figsize=(7, 4))
ax2.set_title("Trayectoria del pico de densidad")
for ci, nombre in enumerate(casos):
    xp = [x[np.argmax(res[nombre][tv][0]-rho0)] for tv in t_vals]
    ax2.plot(t_vals, xp, 'o-', color=colores[ci], lw=1.8, label=nombre)
ax2.set_xlabel("t"); ax2.set_ylabel("x del pico")
ax2.legend(); ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()