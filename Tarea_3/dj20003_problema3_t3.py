"""
Práctica Numérica 3 - Ejercicio 3: Propagación en Sentido Opuesto
Invertir el signo de v' selecciona la invariante de Riemann en -x.
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

def simular(signo_v, t_end=1.5):
    """Simula la onda con la dirección controlada por signo_v.
    signo_v = +1 → onda viaja en +x (invariante R+: δv = +cs δρ/ρ₀)
    signo_v = -1 → onda viaja en -x (invariante R-: δv = -cs δρ/ρ₀)
    El signo de la perturbación de velocidad es lo que determina la dirección."""
    w = np.cos(2*np.pi*(x-x0)/(xf-x0) + 5*np.pi/8)
    U = prim2cons(rho0+A*rho0*w, p0+A*gamma*p0*w, v0+signo_v*A*cs0*w)
    t_sal = [0.0, 0.5, 1.0, 1.5]
    snaps = {}
    t, idx = 0.0, 0
    if abs(t-t_sal[0]) < 1e-12:
        snaps[t_sal[0]] = cons2prim(U); idx += 1
    while t < t_end and idx < len(t_sal):
        dt = min(dt_cfl(U), t_sal[idx]-t)
        U = lax(U, dt); t += dt
        if abs(t-t_sal[idx]) < 1e-10:
            snaps[t_sal[idx]] = cons2prim(U); idx += 1
    return snaps

def analitica(t, signo_v):
    """Solución analítica lineal: la fase se mueve en la dirección de signo_v."""
    k = 2*np.pi/(xf-x0)
    w = np.cos(k*(x-x0) - signo_v*cs0*k*t + 5*np.pi/8)
    return rho0+A*rho0*w

res_pos = simular(+1)   # onda en +x
res_neg = simular(-1)   # onda en -x

# Verificar la dirección midiendo el desplazamiento del pico entre t=0 y t=1
for signo, res, lbl in [(+1,res_pos,'+x'),(-1,res_neg,'-x')]:
    dx_pico = x[np.argmax(res[1.0][0]-rho0)] - x[np.argmax(res[0.0][0]-rho0)]
    print(f"Onda {lbl}: desplazamiento={dx_pico:.4f}  esperado={signo*cs0:.4f}")

t_sal = [0.0, 0.5, 1.0, 1.5]
fig, axes = plt.subplots(2, 2, figsize=(11, 7))
fig.suptitle("Ejercicio 3: Ondas en sentido opuesto — densidad relativa")

for idx_t, tv in enumerate(t_sal):
    ax = axes[idx_t//2][idx_t%2]
    ax.set_title(f"t = {tv:.1f}")
    # Azul: onda en +x; Rojo: onda en -x; sólido: numérico; punteado: analítico
    ax.plot(x, (res_pos[tv][0]-rho0)/rho0, 'b-',  lw=2.0, label="Num. +x")
    ax.plot(x, (analitica(tv,+1)-rho0)/rho0,'b--', lw=1.0, alpha=0.6, label="Anal. +x")
    ax.plot(x, (res_neg[tv][0]-rho0)/rho0, 'r-',  lw=2.0, label="Num. -x")
    ax.plot(x, (analitica(tv,-1)-rho0)/rho0,'r--', lw=1.0, alpha=0.6, label="Anal. -x")
    ax.set_xlabel("x"); ax.set_ylabel(r"$(ρ-ρ_0)/ρ_0$")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\nFísica: onda +x → δv = +cs δρ/ρ₀  (invariante R+)")
print("        onda -x → δv = -cs δρ/ρ₀  (invariante R-)")