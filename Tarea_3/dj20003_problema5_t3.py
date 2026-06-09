"""
Práctica Numérica 3 - Ejercicio 5: Convergencia
Error L2 para N = 100, 200, 400, 800. Orden empírico de convergencia.
"""

import numpy as np
import matplotlib.pyplot as plt

gamma = 5/3
rho0, p0, v0 = 1.0, 1.0, 0.0
A = 2.1e-4
cs0 = np.sqrt(gamma*p0/rho0)
x0, xf = 0.0, 1.0
CFL = 0.45
t_fin = 1.0   # tiempo al que se mide el error

def prim2cons(rho, p, v):
    """Convierte (rho, p, v) -> U = [rho, rho*v, E]."""
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

def lax(U, dt, dx):
    """Un paso de Lax-Friedrichs. dx se pasa explícitamente porque varía con N."""
    F = flujos(U)
    return 0.5*(np.roll(U,-1,1)+np.roll(U,1,1)) - dt/(2*dx)*(np.roll(F,-1,1)-np.roll(F,1,1))

def simular_N(N):
    """Simula hasta t_fin con N celdas y devuelve (dx, error L2).
    El error L2 compara la densidad numérica con la solución analítica exacta."""
    dx = (xf-x0)/N
    x  = x0 + (np.arange(N)+0.5)*dx
    w  = np.cos(2*np.pi*(x-x0)/(xf-x0) + 5*np.pi/8)
    U  = prim2cons(rho0+A*rho0*w, p0+A*gamma*p0*w, v0+A*cs0*w)
    t  = 0.0
    while t < t_fin:
        rho, p, v = cons2prim(U)
        # dt se recorta para no sobrepasar t_fin
        dt = min(CFL*dx/np.max(np.abs(v)+np.sqrt(gamma*p/rho)), t_fin-t)
        U = lax(U, dt, dx); t += dt
    # Solución analítica: onda que viaja a cs0 en +x durante t_fin
    k = 2*np.pi/(xf-x0)
    rho_exact = rho0 + A*rho0*np.cos(k*(x-x0) - cs0*k*t_fin + 5*np.pi/8)
    # Error L2 (ec. 22): raíz cuadrática media de la diferencia puntual
    L2 = np.sqrt(np.mean((U[0]-rho_exact)**2))
    return dx, L2

N_vals = [100, 200, 400, 800]
dx_vals, L2_vals = [], []
for N in N_vals:
    dx, L2 = simular_N(N)
    dx_vals.append(dx); L2_vals.append(L2)
    print(f"N={N:4d}  dx={dx:.5f}  L2={L2:.4e}")

# Ajuste lineal en log-log: pendiente = orden de convergencia
# Si L2 proporcional a dx^p, entonces log(L2) = p*log(dx) + cte
orden, b = np.polyfit(np.log10(dx_vals), np.log10(L2_vals), 1)
print(f"\nOrden de convergencia empírico = {orden:.3f}")

dx_arr = np.array(dx_vals)
fig, ax = plt.subplots(figsize=(7, 5))
ax.loglog(dx_vals, L2_vals, 'o-', color='navy', lw=2, ms=8, label=f"L2 numérico (orden={orden:.2f})")
# Líneas de referencia para comparar visualmente con orden 1 y 2
ax.loglog(dx_arr, 10**b * dx_arr**1,   ':', color='gray',   lw=1.5, label="Orden 1 (ref.)")
ax.loglog(dx_arr, 10**b * dx_arr**2,   '--', color='orange', lw=1.5, label="Orden 2 (ref.)")
for N, dxi, L2i in zip(N_vals, dx_vals, L2_vals):
    ax.annotate(f"N={N}", (dxi, L2i), textcoords="offset points", xytext=(5,5), fontsize=9)
ax.set_xlabel(r"$\Delta x$"); ax.set_ylabel(r"$L_2$")
ax.set_title("Ejercicio 5: Convergencia de Lax-Friedrichs")
ax.legend(); ax.grid(True, alpha=0.3, which='both')
plt.tight_layout()
plt.show()