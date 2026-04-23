r"""
=============================================================
Práctica Numérica 2
=============================================================

Potencial bidimensional:
    V(x,y) = s * x^2 * y^2 * e^{-(x^2+y^2)}
donde s = +1 (repulsivo) o s = -1 (atractivo).
r"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#  PARAMETROS 
 
M    = 0.5      # masa de la particula
VX0  = 0.5      # velocidad inicial: \dot{x}(0) = v_{x0}
VY0  = 0.0      # velocidad inicial: \dot{y}(0) = 0
DT   = 0.01     # paso de tiempo \Delta t para RK4
TMAX = 200.0    # tiempo máximo de integración
TOL  = 1e-6     # tolerancia: PE/KE \leq 10^{-6}

SIGNOS = {"+1 (repulsivo)": +1,
          "-1 (atractivo)": -1}

# Valores de b a simular: -1 \leq b \leq 1, \Delta b = 0.05
B_VALS = np.arange(-1.0, 1.05, 0.05)


#  Potencial y fuerzas

def V(x, y, s):
    r"""
    (a)  V(x,y) = s \cdot x^2 y^2 \, e^{-(x^2+y^2)}
    El factor exponencial confinante hace que V \to 0 para r \to \infty.
    r"""
    return s * x**2 * y**2 * np.exp(-(x**2 + y**2))


def gauss(x, y):
    r"""Factor gaussiano e^{-(x^2+y^2)}, reutilizado en Fx y Fy.r"""
    return np.exp(-(x**2 + y**2))


def Fx(x, y, s):
    r"""
    (b)  F_x = -\partial V/\partial x
           = -s \cdot 2xy^2(1-x^2)\,e^{-(x^2+y^2)}

    Se deriva con la regla del producto sobre x^2 e^{-x^2}:
      \partial_x [x^2 e^{-x^2}] = 2x(1-x^2)e^{-x^2}
    r"""
    return -s * 2.0 * x * y**2 * (1.0 - x**2) * gauss(x, y)


def Fy(x, y, s):
    r"""
    (b)  F_y = -\partial V/\partial y
           = -s \cdot 2x^2 y(1-y^2)\,e^{-(x^2+y^2)}
    r"""
    return -s * 2.0 * x**2 * y * (1.0 - y**2) * gauss(x, y)


# ─────────────────────────────────────────────
#  INTEGRADOR: RK4
# ─────────────────────────────────────────────

def derivadas(q, s):
    r"""
    (d)  Vector de derivadas del estado q = [x, y, v_x, v_y]:

        dq/dt = [ v_x,
                  v_y,
                  F_x(x,y)/m,
                  F_y(x,y)/m ]

    Las ecuaciones de movimiento vienen de la 2a ley de Newton:
      m\ddot{x} = F_x,   m\ddot{y} = F_y
    r"""
    x, y, vx, vy = q
    return np.array([vx,
                     vy,
                     Fx(x, y, s) / M,
                     Fy(x, y, s) / M])


def rk4_step(q, dt, s):
    r"""
    Un paso de Runge-Kutta de 4o orden:

      k_1 = f(q)
      k_2 = f(q + (dt/2) k_1)
      k_3 = f(q + (dt/2) k_2)
      k_4 = f(q + dt k_3)
      q_{n+1} = q_n + (dt/6)(k_1 + 2k_2 + 2k_3 + k_4)

    Error local O(dt^5), error global O(dt^4).
    r"""
    k1 = derivadas(q,              s)
    k2 = derivadas(q + dt/2 * k1, s)
    k3 = derivadas(q + dt/2 * k2, s)
    k4 = derivadas(q + dt   * k3, s)
    return q + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


# ─────────────────────────────────────────────
#  CONDICIÓN INICIAL EN x
# ─────────────────────────────────────────────

def x_inicial(b, s):
    r"""
    (e)  Encuentra x_0 < 0 tal que PE/KE \leq 10^{-6}.

    KE = (1/2) m v_{x0}^2  (constante al inicio, vy0=0)
    PE = |V(x_0, b, s)|

    Empezamos en x_0 = -2 y alejamos de a 0.5 hasta cumplir la condicion.
    r"""
    KE = 0.5 * M * VX0**2      # energia cinetica inicial (vy0=0)
    x0 = -2.0
    while abs(V(x0, b, s)) / KE > TOL:
        x0 -= 0.5
    return x0


# ─────────────────────────────────────────────
#  SIMULACIÓN DE UNA TRAYECTORIA
# ─────────────────────────────────────────────

def simular(b, s, dt=DT, tmax=TMAX):
    r"""
    Integra las ecuaciones de movimiento para parametro de impacto b
    y signo s, devolviendo:

      xs, ys   : trayectoria en el plano xy
      vxs, vys : velocidades (espacio de fases)
      theta    : angulo de dispersion \theta = \mathrm{atan2}(v_y, v_x) [grados]
      t_sal    : tiempo de salida (para el retardo temporal T(b))

    Logica de deteccion de salida:
      - "entro"  cuando PE/KE > TOL   (particula en la region de interaccion)
      - "salio"  cuando PE/KE <= TOL  AND  x > 0  (ya cruzo al otro lado)
    r"""
    x0 = x_inicial(b, s)
    q  = np.array([x0, b, VX0, VY0])   # estado inicial q = [x, y, vx, vy]

    xs,  ys  = [q[0]], [q[1]]
    vxs, vys = [q[2]], [q[3]]

    dentro = False   # flag: ya entro a la region de interaccion?
    t      = 0.0
    t_sal  = tmax    # tiempo de salida por defecto

    while t < tmax:
        q = rk4_step(q, dt, s)
        x, y, vx, vy = q

        xs.append(x);  ys.append(y)
        vxs.append(vx); vys.append(vy)
        t += dt

        KE    = 0.5 * M * (vx**2 + vy**2)
        PE    = abs(V(x, y, s))
        ratio = PE / KE if KE > 0 else 0.0

        if ratio > TOL:
            dentro = True           # entro a la region

        if dentro and ratio <= TOL and x > 0:
            # (i)  \theta = \mathrm{atan2}(v_y, v_x)
            theta = np.degrees(np.arctan2(vy, vx))
            t_sal = t
            return xs, ys, vxs, vys, theta, t_sal

    # Si nunca salio (particula atrapada), usamos la velocidad final
    theta = np.degrees(np.arctan2(vys[-1], vxs[-1]))
    return xs, ys, vxs, vys, theta, t_sal


# ─────────────────────────────────────────────
#  CÁLCULO MASIVO PARA AMBOS SIGNOS
# ─────────────────────────────────────────────

def calcular_todo():
    r"""
    Corre la simulacion para todos los b y ambos signos.
    Devuelve un dict con los resultados:
      resultados[etiqueta] = {
          'trayectorias': [(xs,ys,vxs,vys,theta,t_sal), ...],
          'angulos'     : [theta(b), ...],
          'tiempos_sal' : [t_sal(b), ...],
          's'           : signo numerico
      }
    r"""
    resultados = {}
    for etiq, s in SIGNOS.items():
        print(f"\n  Simulando potencial {etiq}  ({len(B_VALS)} valores de b)...")
        trajs, angulos, t_sals = [], [], []
        for b in B_VALS:
            datos = simular(b, s)
            trajs.append(datos)
            angulos.append(datos[4])
            t_sals.append(datos[5])
            print(f"    b={b:+.2f}  theta={datos[4]:+7.2f} deg  t_sal={datos[5]:.1f}")
        resultados[etiq] = {'trayectorias': trajs,
                            'angulos'     : angulos,
                            'tiempos_sal' : t_sals,
                            's'           : s}
    return resultados


# ─────────────────────────────────────────────
#  VERIFICACIÓN ANALÍTICA DE LOS MÁXIMOS  (inciso c)
# ─────────────────────────────────────────────

def verificar_maximos():
    r"""
    (c)  Los maximos de V se encuentran en x=\pm1, y=\pm1
         donde la fuerza se anula: F_x = F_y = 0.

         V(\pm1,\pm1) = \pm e^{-2}  \approx \pm 0.1353

    Verificamos numericamente para ambos signos.
    r"""
    print("\n" + "="*58)
    print("(c) Verificacion: F=0 en los maximos (x,y) = (+/-1, +/-1)")
    print("="*58)
    puntos = [(1,1),(1,-1),(-1,1),(-1,-1)]
    for s_val, etiq in [(+1,"repulsivo"),(-1,"atractivo")]:
        print(f"\n  Potencial {etiq} (s={s_val:+d}):")
        print(f"  {'Punto':^12}  {'V':^10}  {'V_teo':^10}  {'Fx':^10}  {'Fy':^10}")
        for (px, py) in puntos:
            v   = V(px, py, s_val)
            fx  = Fx(px, py, s_val)
            fy  = Fy(px, py, s_val)
            vt  = s_val * np.exp(-2)
            print(f"  ({px:+d},{py:+d}){' ':4}  {v:+.6f}  {vt:+.6f}  {fx:+.2e}  {fy:+.2e}")


# ─────────────────────────────────────────────
#  GRAFICACIONES
# ─────────────────────────────────────────────

CMAP_TRAY = plt.cm.plasma     # colormap compartido para las trayectorias


def malla_potencial(s, N=200):
    r"""Malla (X,Y,Z) de N puntos para los contornos de fondo.r"""
    xy = np.linspace(-3.5, 3.5, N)
    X, Y = np.meshgrid(xy, xy)
    return X, Y, V(X, Y, s)


# ── (a) Potencial: heatmap + superficie 3D ───────────────────

def graficar_potencial():
    r"""
    (a)  V(x,y) = s \cdot x^2 y^2 e^{-(x^2+y^2)}

    Maximos en (\pm1,\pm1) marcados con circulos negros.
    Para s>0: maximos positivos (colinas repulsivas).
    Para s<0: maximos negativos (pozos atractivos).
    r"""
    fig = plt.figure(figsize=(16, 7))
    xy  = np.linspace(-3, 3, 250)
    X, Y = np.meshgrid(xy, xy)
    maximos = [(1,1),(1,-1),(-1,1),(-1,-1)]

    for col, (etiq, s) in enumerate(SIGNOS.items(), start=1):
        Z = V(X, Y, s)

        # --- heatmap ---
        ax = fig.add_subplot(2, 2, col)
        cf = ax.contourf(X, Y, Z, levels=60, cmap='RdBu_r')
        fig.colorbar(cf, ax=ax, label=r'$V(x,y)$', shrink=0.85)
        for px, py in maximos:
            ax.plot(px, py, 'ko', ms=5, label=r'$(\pm1,\pm1)$')
        ax.set_title(f'Heatmap — {etiq}')
        ax.set_xlabel('$x$');  ax.set_ylabel('$y$')
        ax.set_aspect('equal')

        # --- superficie 3D (malla mas gruesa para rendimiento) ---
        xy2 = np.linspace(-3, 3, 70)
        X2, Y2 = np.meshgrid(xy2, xy2)
        Z2 = V(X2, Y2, s)
        ax3 = fig.add_subplot(2, 2, col+2, projection='3d')
        ax3.plot_surface(X2, Y2, Z2, cmap='RdBu_r', alpha=0.85, linewidth=0)
        ax3.set_title(f'Superficie 3D — {etiq}')
        ax3.set_xlabel('$x$'); ax3.set_ylabel('$y$'); ax3.set_zlabel('$V$')

    fig.suptitle(r'$V(x,y)=s\,x^2y^2\,e^{-(x^2+y^2)}$', fontsize=13)
    plt.tight_layout()
    plt.savefig('potencial.png', dpi=150)
    plt.show()
    print("Guardado: potencial.png")


# ── (g) Trayectorias en el plano xy ──────────────────────────

def graficar_trayectorias(resultados):
    r"""
    (g)  Trayectorias [x(t), y(t)] para -1 \leq b \leq 1.
    El color codifica el valor de b (plasma colormap).
    El contorno del potencial se muestra de fondo en gris.
    r"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    colores   = CMAP_TRAY(np.linspace(0, 1, len(B_VALS)))

    for ax, (etiq, res) in zip(axes, resultados.items()):
        X, Y, Z = malla_potencial(res['s'])
        ax.contourf(X, Y, Z, levels=25, cmap='Greys', alpha=0.25)
        ax.contour( X, Y, Z, levels=10, colors='gray', linewidths=0.4, alpha=0.5)

        for i, (xs, ys, *_) in enumerate(res['trayectorias']):
            ax.plot(xs, ys, color=colores[i], lw=0.7, alpha=0.85)

        # marcar los cuatro maximos del potencial
        for px in [-1, 1]:
            for py in [-1, 1]:
                ax.plot(px, py, 'k+', ms=9, mew=1.8)

        ax.set_xlim(-5, 5);  ax.set_ylim(-3.5, 3.5)
        ax.set_xlabel('$x$'); ax.set_ylabel('$y$')
        ax.set_title(f'Trayectorias — {etiq}')
        ax.set_aspect('equal')

    # barra de color comun que indica el valor de b
    sm = plt.cm.ScalarMappable(cmap=CMAP_TRAY,
                               norm=plt.Normalize(B_VALS[0], B_VALS[-1]))
    fig.colorbar(sm, ax=axes, label='Parametro de impacto $b$', shrink=0.6)
    fig.suptitle(r'Trayectorias $[x(t),\,y(t)]$', fontsize=13)
    plt.tight_layout()
    plt.savefig('trayectorias.png', dpi=150)
    plt.show()
    print("Guardado: trayectorias.png")


# ── (h) Espacio de fases ─────────────────────────────────────

def graficar_espacio_de_fases(resultados):
    r"""
    (h)  Espacio de fases [x(t), \dot{x}(t)] y [y(t), \dot{y}(t)].

    Para estados ligados las trayectorias son curvas cerradas.
    Para estados de dispersion son curvas abiertas: la particula
    entra y sale con velocidades distintas (no vuelve al origen).
    r"""
    b_sel   = [-0.8, -0.4, 0.0, 0.4, 0.8]
    col_sel = ['tab:blue','tab:green','tab:red','tab:orange','tab:purple']

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    for fila, (etiq, res) in enumerate(resultados.items()):
        ax_x = axes[fila, 0]
        ax_y = axes[fila, 1]

        for b, color in zip(b_sel, col_sel):
            # Buscamos el indice del b mas cercano en B_VALS
            idx = np.argmin(np.abs(B_VALS - b))
            xs, ys, vxs, vys, *_ = res['trayectorias'][idx]

            lbl = f'$b={b}$'
            ax_x.plot(xs,  vxs, color=color, lw=0.8, label=lbl)
            ax_y.plot(ys,  vys, color=color, lw=0.8, label=lbl)

        ax_x.set_xlabel('$x$');   ax_x.set_ylabel(r'$\dot{x}$')
        ax_y.set_xlabel('$y$');   ax_y.set_ylabel(r'$\dot{y}$')
        ax_x.set_title(rf'Fase $(x,\dot{{x}})$ — {etiq}')
        ax_y.set_title(rf'Fase $(y,\dot{{y}})$ — {etiq}')
        ax_x.legend(fontsize=8);  ax_y.legend(fontsize=8)
        ax_x.grid(alpha=0.3);     ax_y.grid(alpha=0.3)

    fig.suptitle('Espacio de fases', fontsize=13)
    plt.tight_layout()
    plt.savefig('espacio_de_fases.png', dpi=150)
    plt.show()
    print("Guardado: espacio_de_fases.png")


# ── (i) Ángulo de dispersión θ(b) ────────────────────────────

def graficar_angulo_dispersion(resultados):
    r"""
    (i)  \theta(b) = \mathrm{atan2}(v_y,\, v_x)\big|_{\text{salida}}  [grados]

    Las discontinuidades en d\theta/db indican regiones de variacion
    rapida de la seccion eficaz diferencial \sigma(\theta).
    r"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    for ax, (etiq, res) in zip(axes, resultados.items()):
        ax.plot(B_VALS, res['angulos'], 'o-', ms=3.5, lw=1.2, color='steelblue')
        ax.axhline(0, color='k', lw=0.6, ls='--')
        ax.set_xlabel('Parametro de impacto $b$')
        ax.set_ylabel(r'$\theta$ [grados]')
        ax.set_title(rf'$\theta(b)$ — {etiq}')
        ax.grid(alpha=0.3)

    fig.suptitle(r'Angulo de dispersion $\theta = \mathrm{atan2}(v_y,v_x)$',
                 fontsize=13)
    plt.tight_layout()
    plt.savefig('angulo_dispersion.png', dpi=150)
    plt.show()
    print("Guardado: angulo_dispersion.png")


# ── (l) Retardo temporal T(b) ────────────────────────────────

def graficar_retardo_temporal(resultados):
    r"""
    (l)  T(b) = t_{sal}(b) - t_{libre}

    Donde t_{libre} = |2x_0| / v_{x0} es el tiempo que tardaria
    una particula libre en cruzar la misma region.

    Escala semilogaritmica para detectar regiones con T muy grande
    (comportamiento caotico / quasi-ligado).
    r"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    for ax, (etiq, res) in zip(axes, resultados.items()):
        # Tiempo de referencia: particula libre con b=0
        x0_ref  = x_inicial(0.0, res['s'])
        t_libre = abs(2.0 * x0_ref) / VX0

        retardos = np.abs(np.array(res['tiempos_sal']) - t_libre) + 1e-10

        ax.semilogy(B_VALS, retardos, 'o-', ms=3.5, lw=1.2, color='darkorange')
        ax.set_xlabel('Parametro de impacto $b$')
        ax.set_ylabel('$|T(b)|$ [escala log]')
        ax.set_title(f'Retardo temporal — {etiq}')
        ax.grid(True, which='both', alpha=0.3)

    fig.suptitle('Retardo temporal $T(b)$', fontsize=13)
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
    print("  V(x,y) = s * x^2 * y^2 * exp(-(x^2+y^2))")
    print("  s in {+1 (repulsivo), -1 (atractivo)}")
    print("="*58)

    # (c) Verificacion analitica: F=0 en (±1,±1),  V_max = ±e^{-2}
    verificar_maximos()

    # (a) Grafica del potencial (ambos signos en una sola figura)
    print("\nGraficando potencial...")
    graficar_potencial()

    # Calcula TODAS las trayectorias de una vez (ambos signos, todos los b)
    print("\nCorriendo simulaciones para ambos potenciales...")
    resultados = calcular_todo()

    # (g) Trayectorias en el plano xy
    print("\nGraficando trayectorias...")
    graficar_trayectorias(resultados)

    # (h) Espacio de fases [x, vx] y [y, vy]
    print("\nGraficando espacio de fases...")
    graficar_espacio_de_fases(resultados)

    # (i) Angulo de dispersion theta(b)
    print("\nGraficando angulo de dispersion...")
    graficar_angulo_dispersion(resultados)

    # (l) Retardo temporal T(b) en escala semilogaritmica
    print("\nGraficando retardo temporal...")
    graficar_retardo_temporal(resultados)

    print("\n" + "="*58)
    print("  Listo! Archivos PNG generados:")
    for f in ["potencial.png", "trayectorias.png",
              "espacio_de_fases.png", "angulo_dispersion.png",
              "retardo_temporal.png"]:
        print(f"    {f}")
    print("="*58)