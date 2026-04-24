r"""
=============================================================
Práctica Numérica 2 — Física Computacional
=============================================================

Potencial bidimensional:
    V(x,y) = s * x^2 * y^2 * e^{-(x^2+y^2)}
donde s = +1 (repulsivo) o s = -1 (atractivo).

  (a) Heatmap y superficie 3D del potencial
  (b,c) Ecuaciones de movimiento y verificación de máximos
  (d) Integrador RK4
  (e,f) Condiciones iniciales y parámetros
  (g) Trayectorias en el plano xy
  (h) Espacio de fases [x,vx] y [y,vy]
  (i) Ángulo de dispersión theta(b)
  (j) Discusión de discontinuidades en d_theta/db → sigma(theta)
  (k) Comparación a energías E < Vmax y E > Vmax
  (l) Retardo temporal T(b) + zoom en regiones oscilatorias
=============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ─────────────────────────────────────────────
#  PARÁMETROS
# ─────────────────────────────────────────────
M    = 0.5      # masa de la partícula
VX0  = 0.5      # velocidad inicial \dot{x}(0) = v_{x0}
VY0  = 0.0      # velocidad inicial \dot{y}(0) = 0
DT   = 0.01     # paso de tiempo \Delta t para RK4
TMAX = 200.0    # tiempo máximo de integración
TOL  = 1e-6     # tolerancia PE/KE \leq 10^{-6}

# V_{max} = e^{-2}: escala de energía del problema  (inciso k)
VMAX = np.exp(-2)       # ≈ 0.1353

SIGNOS = {"+1 (repulsivo)": +1,
          "-1 (atractivo)": -1}

# Valores de b: -1 \leq b \leq 1, \Delta b = 0.05
B_VALS = np.arange(-1.0, 1.05, 0.05)


# ─────────────────────────────────────────────
#  FUERZA Y POTENCIAL
# ─────────────────────────────────────────────

def V(x, y, s):
    r"""
    (a)  V(x,y) = s \cdot x^2 y^2 \, e^{-(x^2+y^2)}
    """
    return s * x**2 * y**2 * np.exp(-(x**2 + y**2))


def gauss(x, y):
    r"""Factor gaussiano e^{-(x^2+y^2)}, compartido entre Fx y Fy."""
    return np.exp(-(x**2 + y**2))


def Fx(x, y, s):
    r"""
    (b)  F_x = -\partial_x V = -s \cdot 2xy^2(1-x^2)\,e^{-(x^2+y^2)}
    """
    return -s * 2.0 * x * y**2 * (1.0 - x**2) * gauss(x, y)


def Fy(x, y, s):
    r"""
    (b)  F_y = -\partial_y V = -s \cdot 2x^2y(1-y^2)\,e^{-(x^2+y^2)}
    """
    return -s * 2.0 * x**2 * y * (1.0 - y**2) * gauss(x, y)


# ─────────────────────────────────────────────
#  RK4
# ─────────────────────────────────────────────

def derivadas(q, s):
    r"""
    (d)  dq/dt = [v_x, v_y, F_x/m, F_y/m]
    con q = [x, y, v_x, v_y].
    """
    x, y, vx, vy = q
    return np.array([vx, vy,
                     Fx(x, y, s) / M,
                     Fy(x, y, s) / M])


def rk4_step(q, dt, s):
    r"""
    Un paso RK4:
      k_i calculados con el campo f = derivadas
      q_{n+1} = q_n + (dt/6)(k_1 + 2k_2 + 2k_3 + k_4)
    Error global O(dt^4).
    """
    k1 = derivadas(q,             s)
    k2 = derivadas(q + dt/2 * k1, s)
    k3 = derivadas(q + dt/2 * k2, s)
    k4 = derivadas(q + dt   * k3, s)
    return q + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


# ─────────────────────────────────────────────
#  CONDICIÓN INICIAL EN x
# ─────────────────────────────────────────────

def x_inicial(b, s, vx0=VX0):
    r"""
    (e)  Busca x_0 < 0 tal que |V(x_0,b)| / KE \leq 10^{-6}.
    KE = (1/2) m v_{x0}^2.
    Aleja de a 0.5 hasta cumplir la condición.
    """
    KE = 0.5 * M * vx0**2
    x0 = -2.0
    while abs(V(x0, b, s)) / KE > TOL:
        x0 -= 0.5
    return x0


# ─────────────────────────────────────────────
#  SIMULACIÓN DE UNA TRAYECTORIA
# ─────────────────────────────────────────────

def simular(b, s, vx0=VX0, dt=DT, tmax=TMAX):
    r"""
    Integra la trayectoria para parámetro de impacto b y signo s.

    Devuelve:
      xs, ys   : posiciones
      vxs, vys : velocidades
      theta    : \theta = \mathrm{atan2}(v_y, v_x) al salir [grados]
      t_sal    : tiempo de salida

    Detección de salida:
      "entró"  → PE/KE > TOL
      "salió"  → PE/KE \leq TOL  AND  x > 0
    """
    x0 = x_inicial(b, s, vx0)
    q  = np.array([x0, b, vx0, VY0])

    xs,  ys  = [q[0]], [q[1]]
    vxs, vys = [q[2]], [q[3]]

    dentro = False
    t      = 0.0
    t_sal  = tmax

    while t < tmax:
        q = rk4_step(q, dt, s)
        x, y, vx, vy = q

        xs.append(x);   ys.append(y)
        vxs.append(vx); vys.append(vy)
        t += dt

        KE    = 0.5 * M * (vx**2 + vy**2)
        PE    = abs(V(x, y, s))
        ratio = PE / KE if KE > 0 else 0.0

        if ratio > TOL:
            dentro = True

        if dentro and ratio <= TOL and x > 0:
            theta = np.degrees(np.arctan2(vy, vx))
            t_sal = t
            return xs, ys, vxs, vys, theta, t_sal

    theta = np.degrees(np.arctan2(vys[-1], vxs[-1]))
    return xs, ys, vxs, vys, theta, t_sal


# ─────────────────────────────────────────────
#  CÁLCULO PARA AMBOS SIGNOS
# ─────────────────────────────────────────────

def calcular_todo(b_vals=B_VALS, vx0=VX0):
    r"""
    Corre la simulación para todos los b y ambos signos.
    Devuelve resultados[etiqueta] con trayectorias, ángulos y tiempos.
    """
    resultados = {}
    for etiq, s in SIGNOS.items():
        print(f"\n  Simulando potencial {etiq}  ({len(b_vals)} valores de b)...")
        trajs, angulos, t_sals = [], [], []
        for b in b_vals:
            datos = simular(b, s, vx0=vx0)
            trajs.append(datos)
            angulos.append(datos[4])
            t_sals.append(datos[5])
            print(f"    b={b:+.3f}  theta={datos[4]:+7.2f} deg  t_sal={datos[5]:.1f}")
        resultados[etiq] = {'trayectorias': trajs,
                            'angulos'     : angulos,
                            'tiempos_sal' : t_sals,
                            's'           : s,
                            'b_vals'      : b_vals}
    return resultados


# ─────────────────────────────────────────────
#  (c) VERIFICACIÓN ANALÍTICA DE LOS MÁXIMOS
# ─────────────────────────────────────────────

def verificar_maximos():
    r"""
    (c)  En los puntos x=\pm1, y=\pm1 se cumple:
           F_x = F_y = 0   (son extremos del potencial)
           V(\pm1,\pm1) = \pm e^{-2}  \approx \pm 0.1353

    Esta es la escala de energía V_{max} del problema.
    """
    print("\n" + "="*58)
    print("(c) Verificacion: F=0 en los maximos (x,y)=(+/-1,+/-1)")
    print("="*58)
    puntos = [(1,1),(1,-1),(-1,1),(-1,-1)]
    for s_val, etiq in [(+1,"repulsivo"),(-1,"atractivo")]:
        print(f"\n  Potencial {etiq} (s={s_val:+d}):")
        print(f"  {'Punto':^12}  {'V':^10}  {'V_teo':^10}  {'Fx':^10}  {'Fy':^10}")
        for (px, py) in puntos:
            v  = V(px, py, s_val)
            fx = Fx(px, py, s_val)
            fy = Fy(px, py, s_val)
            vt = s_val * np.exp(-2)
            print(f"  ({px:+d},{py:+d}){' ':4}  {v:+.6f}  {vt:+.6f}"
                  f"  {fx:+.2e}  {fy:+.2e}")


# ─────────────────────────────────────────────
#  GRAFICAS
# ─────────────────────────────────────────────

CMAP_TRAY = plt.cm.plasma

def malla_potencial(s, N=200):
    xy = np.linspace(-3.5, 3.5, N)
    X, Y = np.meshgrid(xy, xy)
    return X, Y, V(X, Y, s)


# ── (a) Potencial ────────────────────────────

def graficar_potencial():
    r"""
    (a)  Heatmap y superficie 3D de V(x,y) para ambos signos.
    Maximos en (\pm1,\pm1) marcados con cruces negras.
    """
    fig = plt.figure(figsize=(16, 7))
    xy  = np.linspace(-3, 3, 250)
    X, Y = np.meshgrid(xy, xy)
    maximos = [(1,1),(1,-1),(-1,1),(-1,-1)]

    for col, (etiq, s) in enumerate(SIGNOS.items(), start=1):
        Z = V(X, Y, s)
        ax = fig.add_subplot(2, 2, col)
        cf = ax.contourf(X, Y, Z, levels=60, cmap='RdBu_r')
        fig.colorbar(cf, ax=ax, label=r'$V(x,y)$', shrink=0.85)
        for px, py in maximos:
            ax.plot(px, py, 'k+', ms=8, mew=2)
        ax.set_title(f'Heatmap — {etiq}')
        ax.set_xlabel('$x$'); ax.set_ylabel('$y$')
        ax.set_aspect('equal')

        xy2 = np.linspace(-3, 3, 70)
        X2, Y2 = np.meshgrid(xy2, xy2)
        ax3 = fig.add_subplot(2, 2, col+2, projection='3d')
        ax3.plot_surface(X2, Y2, V(X2, Y2, s), cmap='RdBu_r',
                         alpha=0.85, linewidth=0)
        ax3.set_title(f'Superficie 3D — {etiq}')
        ax3.set_xlabel('$x$'); ax3.set_ylabel('$y$'); ax3.set_zlabel('$V$')

    fig.suptitle(r'$V(x,y)=s\,x^2y^2\,e^{-(x^2+y^2)}$', fontsize=13)
    plt.tight_layout()
    plt.savefig('potencial.png', dpi=150)
    plt.show()
    print("Guardado: potencial.png")


# ── (g) Trayectorias xy ──────────────────────

def graficar_trayectorias(resultados):
    r"""
    (g)  Trayectorias [x(t), y(t)] para -1 \leq b \leq 1.
    Color codifica b; contorno gris = potencial de fondo.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for ax, (etiq, res) in zip(axes, resultados.items()):
        colores = CMAP_TRAY(np.linspace(0, 1, len(res['b_vals'])))
        X, Y, Z = malla_potencial(res['s'])
        ax.contourf(X, Y, Z, levels=25, cmap='Greys', alpha=0.25)
        ax.contour( X, Y, Z, levels=10, colors='gray', linewidths=0.4, alpha=0.5)
        for i, (xs, ys, *_) in enumerate(res['trayectorias']):
            ax.plot(xs, ys, color=colores[i], lw=0.7, alpha=0.85)
        for px in [-1,1]:
            for py in [-1,1]:
                ax.plot(px, py, 'k+', ms=9, mew=1.8)
        ax.set_xlim(-5, 5); ax.set_ylim(-3.5, 3.5)
        ax.set_xlabel('$x$'); ax.set_ylabel('$y$')
        ax.set_title(f'Trayectorias — {etiq}')
        ax.set_aspect('equal')

    sm = plt.cm.ScalarMappable(cmap=CMAP_TRAY,
                               norm=plt.Normalize(B_VALS[0], B_VALS[-1]))
    fig.colorbar(sm, ax=axes, label='$b$', shrink=0.6)
    fig.suptitle(r'Trayectorias $[x(t),\,y(t)]$', fontsize=13)
    plt.tight_layout()
    plt.savefig('trayectorias.png', dpi=150)
    plt.show()
    print("Guardado: trayectorias.png")


# ── (h) Espacio de fases ─────────────────────

def graficar_espacio_de_fases(resultados):
    r"""
    (h)  Espacio de fases [x(t), \dot{x}(t)] y [y(t), \dot{y}(t)].

    Diferencia clave con estados ligados:
      - Ligado   → curvas cerradas (la particula orbita indefinidamente)
      - Dispersion → curvas abiertas (entra con v_{x0}, sale con v distinta)
    """
    b_sel   = [-0.8, -0.4, 0.0, 0.4, 0.8]
    col_sel = ['tab:blue','tab:green','tab:red','tab:orange','tab:purple']
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    for fila, (etiq, res) in enumerate(resultados.items()):
        ax_x = axes[fila, 0]
        ax_y = axes[fila, 1]
        for b, color in zip(b_sel, col_sel):
            idx = np.argmin(np.abs(res['b_vals'] - b))
            xs, ys, vxs, vys, *_ = res['trayectorias'][idx]
            lbl = f'$b={b}$'
            ax_x.plot(xs,  vxs, color=color, lw=0.8, label=lbl)
            ax_y.plot(ys,  vys, color=color, lw=0.8, label=lbl)
        ax_x.set_xlabel('$x$');  ax_x.set_ylabel(r'$\dot{x}$')
        ax_y.set_xlabel('$y$');  ax_y.set_ylabel(r'$\dot{y}$')
        ax_x.set_title(rf'$(x,\dot{{x}})$ — {etiq}')
        ax_y.set_title(rf'$(y,\dot{{y}})$ — {etiq}')
        ax_x.legend(fontsize=8); ax_y.legend(fontsize=8)
        ax_x.grid(alpha=0.3);    ax_y.grid(alpha=0.3)

    fig.suptitle('Espacio de fases', fontsize=13)
    plt.tight_layout()
    plt.savefig('espacio_de_fases.png', dpi=150)
    plt.show()
    print("Guardado: espacio_de_fases.png")


# ── (i) + (j)  Ángulo de dispersión theta(b) ─

def graficar_angulo_dispersion(resultados):
    r"""
    (i)   \theta(b) = \mathrm{atan2}(v_y, v_x)\big|_{\rm salida}  [grados]

    (j)   Discontinuidades en d\theta/db:
          Cuando la trayectoria pasa muy cerca de un maximo del potencial,
          pequeñas variaciones en b producen grandes cambios en theta.
          Esto genera picos ("rainbow scattering") en la seccion eficaz:

              \sigma(\theta) = \frac{b}{\sin\theta}
                               \left|\frac{db}{d\theta}\right|

          Los valores de b donde |d\theta/db| -> infty se llaman
          "angulos arcoiris" y producen divergencias en sigma.
          En estas regiones las trayectorias son irregulares porque la
          particula pasa varias veces cerca de los maximos antes de salir.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    for ax, (etiq, res) in zip(axes, resultados.items()):
        b_vals  = res['b_vals']
        angulos = res['angulos']
        ax.plot(b_vals, angulos, 'o-', ms=3.5, lw=1.2, color='steelblue')
        ax.axhline(0, color='k', lw=0.6, ls='--')

        # Marcar visualmente las discontinuidades (saltos > 20°)
        for i in range(1, len(angulos)):
            if abs(angulos[i] - angulos[i-1]) > 20:
                ax.axvline(b_vals[i], color='red', lw=0.8,
                           ls=':', alpha=0.7, label='discontinuidad' if i==1 else '')

        ax.set_xlabel('$b$')
        ax.set_ylabel(r'$\theta$ [grados]')
        ax.set_title(rf'$\theta(b)$ — {etiq}')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle(
        r'(i,j)  $\theta(b)=\mathrm{atan2}(v_y,v_x)$'
        ' — líneas rojas: posibles discontinuidades en $d\theta/db$',
        fontsize=11)
    plt.tight_layout()
    plt.savefig('angulo_dispersion.png', dpi=150)
    plt.show()
    print("Guardado: angulo_dispersion.png")


# ── (k)  Comparación E < Vmax vs E > Vmax ────

def graficar_comparacion_energias():
    r"""
    (k)  V_{max} = e^{-2} \approx 0.1353  fija la escala de energía.

    La energía cinética inicial es KE = (1/2) m v_{x0}^2.
    Variamos v_{x0} para obtener:
      E < V_{max}  →  v_{x0} pequeña  (particula puede quedar atrapada)
      E > V_{max}  →  v_{x0} grande   (particula siempre atraviesa)

    KE = V_{max}  →  v_{x0}^* = sqrt(2 V_{max} / m) = sqrt(2 e^{-2} / m)
    """
    # v_{x0}^* tal que KE = Vmax
    vx_critico = np.sqrt(2 * VMAX / M)
    print(f"\n  V_max = e^(-2) = {VMAX:.6f}")
    print(f"  v_x0 critico (KE = Vmax): {vx_critico:.4f}")

    # Tres energías representativas: baja, crítica, alta
    casos = {
        f"E < Vmax  (vx0={vx_critico*0.5:.3f})": vx_critico * 0.5,
        f"E ≈ Vmax  (vx0={vx_critico:.3f})":     vx_critico,
        f"E > Vmax  (vx0={vx_critico*2.0:.3f})": vx_critico * 2.0,
    }

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharey='row')
    col_b = plt.cm.viridis(np.linspace(0, 1, len(B_VALS)))

    for col, (label_caso, vx0_caso) in enumerate(casos.items()):
        for fila, (etiq, s) in enumerate(SIGNOS.items()):
            ax = axes[fila, col]
            X, Y, Z = malla_potencial(s)
            ax.contourf(X, Y, Z, levels=20, cmap='Greys', alpha=0.2)

            for i, b in enumerate(B_VALS):
                xs, ys, *_ = simular(b, s, vx0=vx0_caso)
                ax.plot(xs, ys, color=col_b[i], lw=0.6, alpha=0.8)

            for px in [-1,1]:
                for py in [-1,1]:
                    ax.plot(px, py, 'k+', ms=7, mew=1.5)

            ax.set_xlim(-5, 5); ax.set_ylim(-3.5, 3.5)
            ax.set_xlabel('$x$'); ax.set_ylabel('$y$')
            ax.set_title(f'{etiq}\n{label_caso}', fontsize=8)
            ax.set_aspect('equal')

    fig.suptitle(r'(k)  Trayectorias para $E < V_{\max}$, '
                 r'$E \approx V_{\max}$, $E > V_{\max}$', fontsize=12)
    plt.tight_layout()
    plt.savefig('comparacion_energias.png', dpi=150)
    plt.show()
    print("Guardado: comparacion_energias.png")


# ── (l)  Retardo temporal T(b) + zoom fino ───

def graficar_retardo_temporal(resultados):
    r"""
    (l)  T(b) = t_{\rm sal}(b) - t_{\rm libre}

    donde t_{\rm libre} = |2x_0| / v_{x0} es el tiempo de cruce libre.

    Escala semilogarítmica para detectar regiones donde T es muy grande
    (la partícula queda quasi-ligada dando vueltas antes de escapar).
    Esas regiones corresponden exactamente a las discontinuidades en theta(b).

    Zoom fino: cuando se identifica un pico en T(b) sobre [b_L, b_R],
    se repite la simulación con b -> b/10 (paso 10x más fino) para
    resolver la estructura interna de la región caótica.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for col, (etiq, res) in enumerate(resultados.items()):
        s      = res['s']
        b_vals = res['b_vals']
        x0_ref = x_inicial(0.0, s)
        t_lib  = abs(2.0 * x0_ref) / VX0
        retardos = np.abs(np.array(res['tiempos_sal']) - t_lib) + 1e-10

        # --- Gráfica global semilog ---
        ax_g = axes[0, col]
        ax_g.semilogy(b_vals, retardos, 'o-', ms=3.5, lw=1.2,
                      color='darkorange')
        ax_g.set_xlabel('$b$')
        ax_g.set_ylabel('$|T(b)|$ [log]')
        ax_g.set_title(f'(l) Retardo temporal — {etiq}')
        ax_g.grid(True, which='both', alpha=0.3)

        # Detectar el b con mayor retardo (candidato a zoom)
        idx_pico = np.argmax(retardos)
        b_pico   = b_vals[idx_pico]
        ax_g.axvline(b_pico, color='red', ls='--', lw=1,
                     label=f'pico b={b_pico:.2f}')
        ax_g.legend(fontsize=8)

        # --- Zoom: región b_pico ± 0.1, paso \Delta b / 10 ---
        b_zoom = np.arange(b_pico - 0.1, b_pico + 0.1, 0.005)
        ret_zoom = []
        print(f"\n  Zoom fino cerca de b={b_pico:.3f}  ({etiq})...")
        for b in b_zoom:
            _, _, _, _, _, t_s = simular(b, s)
            ret_zoom.append(abs(t_s - t_lib) + 1e-10)
            print(f"    b={b:+.4f}  T={ret_zoom[-1]:.3f}")

        ax_z = axes[1, col]
        ax_z.semilogy(b_zoom, ret_zoom, 's-', ms=3, lw=1,
                      color='steelblue')
        ax_z.set_xlabel('$b$ (zoom)')
        ax_z.set_ylabel('$|T(b)|$ [log]')
        ax_z.set_title(f'Zoom — {etiq}  (\\Delta b / 10)')
        ax_z.grid(True, which='both', alpha=0.3)

    fig.suptitle('(l)  Retardo temporal $T(b)$ y zoom en región oscilatoria',
                 fontsize=12)
    plt.tight_layout()
    plt.savefig('retardo_temporal.png', dpi=150)
    plt.show()
    print("Guardado: retardo_temporal.png")


# ─────────────────────────────────────────────
#  PROGRAMA PRINCIPAL
# ─────────────────────────────────────────────

if __name__ == "__main__":

    print("="*58)
    print("  Practica Numerica 2 — Dispersion por potencial 2D")
    print(f"  V_max = e^(-2) = {VMAX:.6f}  (escala de energia)")
    print("="*58)

    # (c) Verificar que F=0 en (±1,±1) y V = ±e^{-2}
    verificar_maximos()

    # (a) Heatmap + superficie 3D para ambos signos
    print("\nGraficando potencial...")
    graficar_potencial()

    # Simular todos los b para ambos signos (vx0 nominal)
    print("\nCorriendo simulaciones principales...")
    resultados = calcular_todo()

    # (g) Trayectorias en el plano xy
    print("\nGraficando trayectorias...")
    graficar_trayectorias(resultados)

    # (h) Espacio de fases
    print("\nGraficando espacio de fases...")
    graficar_espacio_de_fases(resultados)

    # (i) + (j) Angulo de dispersion y discontinuidades
    print("\nGraficando angulo de dispersion (i,j)...")
    graficar_angulo_dispersion(resultados)

    # (k) Comparacion E < Vmax vs E > Vmax
    print("\nGraficando comparacion de energias (k)...")
    graficar_comparacion_energias()

    # (l) Retardo temporal + zoom fino
    print("\nGraficando retardo temporal + zoom (l)...")
    graficar_retardo_temporal(resultados)

    print("\n" + "="*58)
    print("  Archivos generados:")
    for nombre in ["potencial.png", "trayectorias.png",
                   "espacio_de_fases.png", "angulo_dispersion.png",
                   "comparacion_energias.png", "retardo_temporal.png"]:
        print(f"    {nombre}")
    print("="*58)