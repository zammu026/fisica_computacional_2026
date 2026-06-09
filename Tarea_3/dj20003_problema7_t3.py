"""
Práctica Numérica 3 - Ejercicio 7: Régimen No Lineal
A = 1e-3, 1e-2, 1e-1  — empinamiento, armónicos y formación de choques.
"""

import numpy as np
import matplotlib.pyplot as plt

gamma = 5/3
rho0, p0, v0 = 1.0, 1.0, 0.0
cs0 = np.sqrt(gamma*p0/rho0)

N = 800   # mayor resolución para capturar correctamente los choques
x0, xf = 0.0, 1.0
dx = (xf-x0)/N
x  = x0 + (np.arange(N)+0.5)*dx   # centros de celda
CFL = 0.45
k = 2*np.pi/(xf-x0)   # número de onda fundamental del dominio

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
    """Un paso de Lax-Friedrichs con fronteras periódicas."""
    F = flujos(U)
    return 0.5*(np.roll(U,-1,1)+np.roll(U,1,1)) - dt/(2*dx)*(np.roll(F,-1,1)-np.roll(F,1,1))

def simular(A):
    """Simula la evolución para una amplitud A dada.
    El tiempo de choque teórico tc = 1/(A*k*cs0*(gamma+1)/2) marca cuándo
    la onda no lineal forma una discontinuidad (choque).
    Se toman snapshots en fracciones de tc para observar el empinamiento."""
    tc = 1.0 / (A * k * cs0 * (gamma+1)/2)   # tiempo de formación del choque
    t_sal = sorted(set([0.0, 0.25*tc, 0.5*tc, 0.75*tc, tc, min(1.1*tc, 3.0)]))
    w = np.cos(2*np.pi*(x-x0)/(xf-x0) + 5*np.pi/8)
    U = prim2cons(rho0+A*rho0*w, p0+A*gamma*p0*w, v0+A*cs0*w)
    snaps = {}
    t, idx = 0.0, 0
    if abs(t-t_sal[0]) < 1e-14:
        snaps[t_sal[0]] = U[0].copy(); idx += 1
    t_max = t_sal[-1]   # simular hasta el último tiempo de salida
    while idx < len(t_sal):
        rho_c, p_c, v_c = cons2prim(U)
        dt = min(CFL*dx/np.max(np.abs(v_c)+np.sqrt(gamma*p_c/rho_c)), t_sal[idx]-t)
        U = lax(U, dt); t += dt
        if abs(t-t_sal[idx]) < 1e-10:
            snaps[t_sal[idx]] = U[0].copy(); idx += 1
    return snaps, t_sal, tc

amplitudes = [1e-3, 1e-2, 1e-1]
colores = ['navy','royalblue','seagreen','darkorange','crimson','purple']

for A in amplitudes:
    snaps, t_sal, tc = simular(A)
    print(f"A={A:.0e}  t_choque≈{tc:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Ejercicio 7: A = {A:.0e}  (t_choque ≈ {tc:.3f})")

    for ci, tv in enumerate(t_sal):
        lbl = f"t={tv:.3f}"
        # Panel izquierdo: perfil de densidad — se aprecia el empinamiento hacia tc
        axes[0].plot(x, (snaps[tv]-rho0)/rho0, lw=1.8, color=colores[ci], label=lbl)
        # Panel derecho: espectro de potencia de Fourier de la perturbación
        # Si hay no linealidad, aparecen armónicos (picos en frecuencias múltiplos de k)
        fc = np.abs(np.fft.rfft(snaps[tv]-rho0))**2 / N**2
        fr = np.fft.rfftfreq(N, d=dx)
        axes[1].semilogy(fr[:N//4], fc[:N//4]+1e-30, lw=1.5, color=colores[ci], label=lbl)

    axes[0].set_xlabel("x"); axes[0].set_ylabel(r"$(ρ-ρ_0)/ρ_0$")
    axes[0].set_title("Perfil de densidad"); axes[0].legend(fontsize=7); axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel("Frecuencia"); axes[1].set_ylabel("Potencia")
    axes[1].set_title("Espectro (armónicos)"); axes[1].legend(fontsize=7); axes[1].grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.show()

# Gráfica final: comparación del perfil en t≈tc para los tres valores de A
# Muestra cómo la no linealidad es más pronunciada cuanto mayor es A
fig_comp, ax_comp = plt.subplots(figsize=(9, 4))
ax_comp.set_title("Ejercicio 7: Empinamiento comparativo en t ≈ t_choque")
for ci, A in enumerate(amplitudes):
    snaps, t_sal, tc = simular(A)
    tv = min(t_sal, key=lambda tv: abs(tv-tc))   # tiempo más cercano a tc
    ax_comp.plot(x, (snaps[tv]-rho0)/rho0, lw=2, color=colores[ci],
                 label=f"A={A:.0e} (t={tv:.3f})")
ax_comp.set_xlabel("x"); ax_comp.set_ylabel(r"$(ρ-ρ_0)/ρ_0$")
ax_comp.legend(); ax_comp.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()