"""
Práctica Numérica 3 - Ejercicio 6: Estabilidad Numérica
Prueba distintos CFL y determina empíricamente el límite de estabilidad.
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings

gamma = 5/3
rho0, p0, v0 = 1.0, 1.0, 0.0
A = 2.1e-4
cs0 = np.sqrt(gamma*p0/rho0)

N = 200
x0, xf = 0.0, 1.0
dx = (xf-x0)/N
x  = x0 + (np.arange(N)+0.5)*dx   # centros de celda
t_fin = 1.0

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

def lax(U, dt):
    """Un paso de Lax-Friedrichs con dt FIJO (no adaptativo).
    Así se puede forzar un CFL > 1 para estudiar la inestabilidad."""
    F = flujos(U)
    return 0.5*(np.roll(U,-1,1)+np.roll(U,1,1)) - dt/(2*dx)*(np.roll(F,-1,1)-np.roll(F,1,1))

def ci():
    """Genera la condición inicial (perturbación acústica sobre estado base)."""
    w = np.cos(2*np.pi*(x-x0)/(xf-x0) + 5*np.pi/8)
    return prim2cons(rho0+A*rho0*w, p0+A*gamma*p0*w, v0+A*cs0*w)

# Calcular la velocidad máxima de señal en t=0 para fijar dt por CFL
U0  = ci()
rho0_arr, p0_arr, v0_arr = cons2prim(U0)
lam_max = np.max(np.abs(v0_arr) + np.sqrt(gamma*p0_arr/rho0_arr))

CFL_probar = [0.3, 0.5, 0.7, 0.9, 0.95, 1.0, 1.05, 1.2]
resultados = {}

for CFL_t in CFL_probar:
    # dt fijo proporcional al CFL elegido (no se reajusta durante la simulación)
    dt = CFL_t * dx / lam_max
    U  = ci(); t = 0.0; estable = True
    while t < t_fin:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            try:
                Un = lax(U, dt)
                rho_c, p_c, _ = cons2prim(Un)
                # Detectar inestabilidad: NaN, Inf, densidad o presión negativa
                if not np.all(np.isfinite(Un)) or np.any(rho_c<=0) or np.any(p_c<=0):
                    estable = False; break
                U = Un
            except RuntimeWarning:
                estable = False; break
        t += dt
    resultados[CFL_t] = (estable, U.copy())
    print(f"CFL={CFL_t:.2f}  →  {'ESTABLE ✓' if estable else 'INESTABLE ✗'}")

estables   = [c for c in CFL_probar if resultados[c][0]]
inestables = [c for c in CFL_probar if not resultados[c][0]]
print(f"\nCFL crítico entre {max(estables,default=0):.2f} y {min(inestables,default=999):.2f}")

# Solución analítica de referencia en t=t_fin
k = 2*np.pi/(xf-x0)
rho_a = rho0 + A*rho0*np.cos(k*(x-x0) - cs0*k*t_fin + 5*np.pi/8)

# Gráfica 1: perfiles numéricos para todos los CFL estables vs analítica
cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(estables)))
fig, ax = plt.subplots(figsize=(9, 5))
ax.set_title("Ejercicio 6: Perfiles estables para distintos CFL (t = 1.0)")
for i, c in enumerate(estables):
    ax.plot(x, (resultados[c][1][0]-rho0)/rho0, lw=1.5, color=cmap[i], label=f"CFL={c:.2f}")
ax.plot(x, (rho_a-rho0)/rho0, 'k--', lw=1.5, label="Analítica")
ax.set_xlabel("x"); ax.set_ylabel(r"$(ρ-ρ_0)/ρ_0$")
ax.legend(fontsize=9, ncol=2); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Gráfica 2: evolución de la inestabilidad paso a paso
# Se muestran los primeros 5 pasos para ver el crecimiento explosivo
if inestables:
    CFL_i = inestables[0]   # primer CFL inestable
    dt_i  = CFL_i * dx / lam_max
    U_i   = ci()
    snaps_i = [(0, U_i[0].copy())]
    for paso in range(1, 6):
        try:
            U_i = lax(U_i, dt_i)
            snaps_i.append((paso, U_i[0].copy() if np.all(np.isfinite(U_i)) else np.full(N, np.nan)))
        except:
            snaps_i.append((paso, np.full(N, np.nan)))

    cmap2 = plt.cm.Reds(np.linspace(0.3, 0.9, len(snaps_i)))
    fig2, ax2 = plt.subplots(figsize=(9, 4))
    ax2.set_title(f"Ejercicio 6: Inestabilidad con CFL = {CFL_i:.2f}")
    for i, (paso, rho_s) in enumerate(snaps_i):
        ax2.plot(x, (rho_s-rho0)/rho0, color=cmap2[i], lw=1.5, label=f"paso {paso}")
    ax2.set_xlabel("x"); ax2.set_ylabel(r"$(ρ-ρ_0)/ρ_0$")
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()