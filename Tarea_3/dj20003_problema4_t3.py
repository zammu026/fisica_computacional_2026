"""
Práctica Numérica 3 - Ejercicio 4: Conservación
Calcula M(t), P(t), E(t) y los errores relativos δM, δP, δE.
"""

import numpy as np
import matplotlib.pyplot as plt

gamma = 5/3
rho0, p0, v0 = 1.0, 1.0, 0.0
A = 2.1e-4
cs0 = np.sqrt(gamma*p0/rho0)

N = 400
x0, xf = 0.0, 1.0
dx = (xf-x0)/N
x  = x0 + (np.arange(N)+0.5)*dx   # centros de celda
CFL = 0.45
t_end = 3.0   # tiempo largo para ver si hay deriva acumulada

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
    """Paso de tiempo adaptativo por condición CFL."""
    rho, p, v = cons2prim(U)
    return CFL*dx / np.max(np.abs(v)+np.sqrt(gamma*p/rho))

def lax(U, dt):
    """Un paso de Lax-Friedrichs con fronteras periódicas."""
    F = flujos(U)
    return 0.5*(np.roll(U,-1,1)+np.roll(U,1,1)) - dt/(2*dx)*(np.roll(F,-1,1)-np.roll(F,1,1))

# Condición inicial: perturbación acústica sobre el estado base
w = np.cos(2*np.pi*(x-x0)/(xf-x0) + 5*np.pi/8)
U = prim2cons(rho0+A*rho0*w, p0+A*gamma*p0*w, v0+A*cs0*w)

# Integrales iniciales: aproximación discreta de ∫ρ dx, ∫ρv dx, ∫E dx
# Con fronteras periódicas y Lax-Friedrichs estas cantidades deben conservarse
M0, P0, E0 = np.sum(U[0])*dx, np.sum(U[1])*dx, np.sum(U[2])*dx
print(f"t=0:  M0={M0:.10f}  P0={P0:.4e}  E0={E0:.10f}")

# Historial de integrales en el tiempo
t_hist, M_hist, P_hist, E_hist = [0.0],[M0],[P0],[E0]
t, paso = 0.0, 0

while t < t_end:
    dt = min(dt_cfl(U), t_end-t)
    U = lax(U, dt); t += dt; paso += 1
    # Registrar cada 10 pasos para no sobrecargar la memoria
    if paso % 10 == 0:
        M, P, E = np.sum(U[0])*dx, np.sum(U[1])*dx, np.sum(U[2])*dx
        t_hist.append(t); M_hist.append(M); P_hist.append(P); E_hist.append(E)

t_hist = np.array(t_hist)

# Errores relativos respecto al valor inicial (ec. 21)
dM = np.abs(np.array(M_hist)-M0) / abs(M0)
dP = np.abs(np.array(P_hist)-P0) / max(abs(P0), 1e-15)   # evitar división por cero
dE = np.abs(np.array(E_hist)-E0) / abs(E0)

print(f"Error máximo: δM={np.max(dM):.2e}  δP={np.max(dP):.2e}  δE={np.max(dE):.2e}")

# Gráfica en escala logarítmica para ver la magnitud de los errores
fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
fig.suptitle("Ejercicio 4: Errores relativos de conservación")
for ax, d, lbl, col in zip(axes, [dM,dP,dE],
                            [r"$\delta M$",r"$\delta P$",r"$\delta E$"],
                            ['navy','seagreen','crimson']):
    ax.semilogy(t_hist, d+1e-18, color=col, lw=1.5)   # +1e-18 evita log(0)
    ax.set_ylabel(lbl); ax.grid(True, alpha=0.3, which='both')
axes[-1].set_xlabel("t")
plt.tight_layout()
plt.show()