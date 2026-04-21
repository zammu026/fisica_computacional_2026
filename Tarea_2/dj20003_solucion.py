"""
=============================================================
Práctica Numérica 2 - Física Computacional
Universidad de El Salvador
Dispersión de partícula por un potencial bidimensional
=============================================================

Potencial:  V(x, y) = ± x² y² e^(-(x²+y²))

Los signos:
  + → potencial repulsivo
  - → potencial atractivo
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # necesario para gráfica 3D

# ==============================================================
# PARÁMETROS GLOBALES
# ==============================================================
MASA = 0.5        # masa de la partícula
SIGNO = +1        # +1 = repulsivo,  -1 = atractivo
VX0  = 0.5        # velocidad inicial en x  (la partícula viene en x)
VY0  = 0.0        # velocidad inicial en y  (ninguna al inicio)
DT   = 0.01       # paso de tiempo para RK4
TMAX = 200.0      # tiempo máximo de simulación
TOLERANCIA = 1e-6 # relación PE/KE para decidir si está fuera de la región


# ==============================================================
# (a) POTENCIAL  V(x, y)
# ==============================================================
def potencial(x, y, signo=SIGNO):
    """
    Calcula el potencial en el punto (x, y).
    V(x,y) = signo * x² * y² * exp(-(x²+y²))
    """
    return signo * x**2 * y**2 * np.exp(-(x**2 + y**2))


# ==============================================================
# (b) FUERZAS  (negativo del gradiente del potencial)
#
#  Fx = -∂V/∂x = ∓ 2xy²(1 - x²) e^(-(x²+y²))
#  Fy = -∂V/∂y = ∓ 2x²y(1 - y²) e^(-(x²+y²))
#
# (el signo "∓" significa que si signo=+1 en V, la fuerza lleva -,
#  y si signo=-1 en V, la fuerza lleva +)
# ==============================================================
def fuerza_x(x, y, signo=SIGNO):
    """Componente x de la fuerza: Fx = -∂V/∂x"""
    return -signo * 2 * x * y**2 * (1 - x**2) * np.exp(-(x**2 + y**2))

def fuerza_y(x, y, signo=SIGNO):
    """Componente y de la fuerza: Fy = -∂V/∂y"""
    return -signo * 2 * x**2 * y * (1 - y**2) * np.exp(-(x**2 + y**2))


# ==============================================================
# (d) MÉTODO RUNGE-KUTTA DE 4° ORDEN (RK4)
#
# El estado del sistema es el vector:
#   estado = [x, y, vx, vy]
#
# Las ecuaciones de movimiento son:
#   dx/dt  = vx
#   dy/dt  = vy
#   dvx/dt = Fx / m
#   dvy/dt = Fy / m
# ==============================================================
def derivadas(estado, signo=SIGNO):
    """
    Calcula las derivadas del estado en un instante.
    estado = [x, y, vx, vy]
    Devuelve [dx/dt, dy/dt, dvx/dt, dvy/dt]
    """
    x, y, vx, vy = estado

    dx  = vx
    dy  = vy
    dvx = fuerza_x(x, y, signo) / MASA
    dvy = fuerza_y(x, y, signo) / MASA

    return np.array([dx, dy, dvx, dvy])


def paso_rk4(estado, dt, signo=SIGNO):
    """
    Avanza un paso dt usando RK4.
    Devuelve el nuevo estado.
    """
    k1 = derivadas(estado,          signo)
    k2 = derivadas(estado + dt/2*k1, signo)
    k3 = derivadas(estado + dt/2*k2, signo)
    k4 = derivadas(estado + dt*k3,  signo)

    nuevo_estado = estado + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
    return nuevo_estado


# ==============================================================
# FUNCIÓN AUXILIAR: energías cinética y potencial
# ==============================================================
def energia_cinetica(vx, vy):
    return 0.5 * MASA * (vx**2 + vy**2)

def energia_potencial(x, y, signo=SIGNO):
    return potencial(x, y, signo)


# ==============================================================
# (e) POSICIÓN INICIAL EN X
#
# La partícula empieza lejos, donde PE/KE ≤ 1e-6.
# Buscamos el x0 (negativo, porque la partícula viene de la izquierda)
# tal que esa condición se cumpla.
# ==============================================================
def calcular_x0(b, vx0=VX0, signo=SIGNO):
    """
    Encuentra la posición inicial x0 (lejana, negativa) tal que
    |PE/KE| <= 1e-6 para el parámetro de impacto b.
    """
    KE = energia_cinetica(vx0, VY0)
    x0 = -1.0
    while True:
        PE = abs(energia_potencial(x0, b, signo))
        if PE / KE <= TOLERANCIA:
            break
        x0 -= 0.5   # alejar más
    return x0


# ==============================================================
# SIMULACIÓN COMPLETA DE UNA TRAYECTORIA
# ==============================================================
def simular_trayectoria(b, vx0=VX0, signo=SIGNO, dt=DT, tmax=TMAX):
    """
    Simula la trayectoria de una partícula con parámetro de impacto b.

    Parámetros:
      b    : parámetro de impacto (posición inicial en y)
      vx0  : velocidad inicial en x
      signo: +1 repulsivo, -1 atractivo
      dt   : paso de tiempo
      tmax : tiempo máximo

    Devuelve:
      xs, ys, vxs, vys : listas con la trayectoria
      angulo           : ángulo de dispersión final (en grados)
      tiempo_total     : tiempo que tardó en salir
    """
    x0 = calcular_x0(b, vx0, signo)
    estado = np.array([x0, b, vx0, VY0])

    xs, ys, vxs, vys = [x0], [b], [vx0], [VY0]

    t = 0.0
    tiempo_salida = tmax   # por defecto
    angulo = None

    # Variable para saber si ya entró a la región de interacción
    entro = False

    while t < tmax:
        estado = paso_rk4(estado, dt, signo)
        x, y, vx, vy = estado

        xs.append(x)
        ys.append(y)
        vxs.append(vx)
        vys.append(vy)

        t += dt

        # Detectar si entró a la región (PE/KE > 1e-6)
        KE = energia_cinetica(vx, vy)
        PE = abs(energia_potencial(x, y, signo))
        if KE > 0 and PE / KE > TOLERANCIA:
            entro = True

        # Detectar si ya salió de la región (después de haber entrado)
        if entro and KE > 0 and PE / KE <= TOLERANCIA and x > 0:
            # La partícula ya salió por la derecha
            # (i) calcular el ángulo de dispersión
            # atan2(vy, vx) da el ángulo respecto al eje +x
            angulo = np.degrees(np.arctan2(vy, vx))
            tiempo_salida = t
            break

    # Si nunca salió, calcular con el último estado
    if angulo is None:
        vx, vy = vxs[-1], vys[-1]
        angulo = np.degrees(np.arctan2(vy, vx))
        tiempo_salida = tmax

    return xs, ys, vxs, vys, angulo, tiempo_salida


# ==============================================================
# GRAFICACIONES
# ==============================================================

def graficar_potencial():
    """(a) Heatmap y superficie del potencial."""
    x = np.linspace(-3, 3, 300)
    y = np.linspace(-3, 3, 300)
    X, Y = np.meshgrid(x, y)
    V = potencial(X, Y, signo=SIGNO)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Heatmap ---
    ax = axes[0]
    cm = ax.contourf(X, Y, V, levels=50, cmap='RdBu_r')
    fig.colorbar(cm, ax=ax, label='V(x,y)')
    ax.set_title(f'Heatmap del potencial (signo={SIGNO:+d})')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    # marcar los máximos teóricos en (±1, ±1)
    for px in [-1, 1]:
        for py in [-1, 1]:
            ax.plot(px, py, 'ko', markersize=6)
    ax.set_aspect('equal')

    # --- Superficie 3D ---
    ax3 = fig.add_subplot(122, projection='3d')
    # Usamos una malla más gruesa para que sea más rápida
    x2 = np.linspace(-3, 3, 80)
    y2 = np.linspace(-3, 3, 80)
    X2, Y2 = np.meshgrid(x2, y2)
    V2 = potencial(X2, Y2, signo=SIGNO)
    ax3.plot_surface(X2, Y2, V2, cmap='RdBu_r', alpha=0.8)
    ax3.set_title(f'Superficie 3D (signo={SIGNO:+d})')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_zlabel('V(x,y)')

    plt.tight_layout()
    plt.savefig('potencial.png', dpi=150)
    plt.show()
    print("Guardado: potencial.png")


def graficar_trayectorias():
    """(g) Trayectorias en el plano xy para varios parámetros de impacto."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Dibujar contornos del potencial de fondo
    x = np.linspace(-4, 4, 200)
    y = np.linspace(-4, 4, 200)
    X, Y = np.meshgrid(x, y)
    V = potencial(X, Y, SIGNO)
    ax.contourf(X, Y, V, levels=30, cmap='Greys', alpha=0.3)

    # Simular para varios valores de b
    b_valores = np.arange(-1.0, 1.05, 0.1)
    colores = plt.cm.plasma(np.linspace(0, 1, len(b_valores)))

    for i, b in enumerate(b_valores):
        xs, ys, _, _, angulo, _ = simular_trayectoria(b, signo=SIGNO)
        ax.plot(xs, ys, color=colores[i], lw=0.8, alpha=0.9,
                label=f'b={b:.1f}')

    ax.set_xlim(-5, 5)
    ax.set_ylim(-4, 4)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    signo_str = "repulsivo (+)" if SIGNO == 1 else "atractivo (-)"
    ax.set_title(f'Trayectorias — potencial {signo_str}')
    ax.legend(fontsize=7, ncol=3, loc='upper right')
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('trayectorias.png', dpi=150)
    plt.show()
    print("Guardado: trayectorias.png")


def graficar_espacio_de_fases():
    """(h) Espacio de fases: [x, vx] y [y, vy] para algunos b."""
    b_seleccionados = [-0.8, -0.4, 0.0, 0.4, 0.8]
    colores = ['blue', 'green', 'red', 'orange', 'purple']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for b, color in zip(b_seleccionados, colores):
        xs, ys, vxs, vys, _, _ = simular_trayectoria(b, signo=SIGNO)
        ax1.plot(xs, vxs, color=color, lw=0.8, label=f'b={b}')
        ax2.plot(ys, vys, color=color, lw=0.8, label=f'b={b}')

    ax1.set_xlabel('x')
    ax1.set_ylabel('vx')
    ax1.set_title('Espacio de fases: x vs vx')
    ax1.legend(fontsize=8)

    ax2.set_xlabel('y')
    ax2.set_ylabel('vy')
    ax2.set_title('Espacio de fases: y vs vy')
    ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig('espacio_de_fases.png', dpi=150)
    plt.show()
    print("Guardado: espacio_de_fases.png")


def graficar_angulo_dispersion():
    """(i) Ángulo de dispersión θ en función del parámetro de impacto b."""
    b_valores = np.arange(-1.0, 1.05, 0.05)
    angulos   = []

    print("Calculando ángulos de dispersión...")
    for b in b_valores:
        _, _, _, _, ang, _ = simular_trayectoria(b, signo=SIGNO)
        angulos.append(ang)
        print(f"  b = {b:+.2f}   θ = {ang:.2f}°")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(b_valores, angulos, 'o-', color='steelblue', markersize=4)
    ax.set_xlabel('Parámetro de impacto b')
    ax.set_ylabel('Ángulo de dispersión θ (grados)')
    signo_str = "repulsivo (+)" if SIGNO == 1 else "atractivo (-)"
    ax.set_title(f'Ángulo de dispersión — potencial {signo_str}')
    ax.axhline(0, color='k', lw=0.5, ls='--')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('angulo_dispersion.png', dpi=150)
    plt.show()
    print("Guardado: angulo_dispersion.png")


def graficar_retardo_temporal():
    """
    (l) Retardo temporal T(b).
    T(b) = tiempo que tarda la partícula en atravesar la región de interacción.
    Se grafica en escala semilogarítmica.
    """
    b_valores = np.arange(-1.0, 1.05, 0.05)
    tiempos   = []

    # Tiempo de referencia: partícula libre (sin potencial)
    # La partícula libre tarda aprox  |2*x0| / vx0  en cruzar
    x0_ref = calcular_x0(0.0, signo=SIGNO)
    t_libre = abs(2 * x0_ref) / VX0

    print("Calculando retardos temporales...")
    for b in b_valores:
        _, _, _, _, _, t_total = simular_trayectoria(b, signo=SIGNO)
        retardo = t_total - t_libre
        tiempos.append(abs(retardo) + 1e-10)   # evitar log(0)
        print(f"  b = {b:+.2f}   T = {retardo:.3f}")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogy(b_valores, tiempos, 'o-', color='darkorange', markersize=4)
    ax.set_xlabel('Parámetro de impacto b')
    ax.set_ylabel('|Retardo temporal| T(b)  [escala log]')
    signo_str = "repulsivo (+)" if SIGNO == 1 else "atractivo (-)"
    ax.set_title(f'Retardo temporal — potencial {signo_str}')
    ax.grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    plt.savefig('retardo_temporal.png', dpi=150)
    plt.show()
    print("Guardado: retardo_temporal.png")


# ==============================================================
# (c) VERIFICACIÓN ANALÍTICA: los máximos están en (±1, ±1)
# ==============================================================
def verificar_maximos():
    """
    Muestra que la fuerza se anula en (±1, ±1) y que
    V_max = ±e^(-2).
    """
    print("=" * 50)
    print("(c) Verificación de los máximos del potencial")
    print("=" * 50)
    puntos = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    for (x, y) in puntos:
        V  = potencial(x, y, SIGNO)
        Fx = fuerza_x(x, y, SIGNO)
        Fy = fuerza_y(x, y, SIGNO)
        Vmax_teorico = SIGNO * np.exp(-2)
        print(f"  Punto ({x:+d}, {y:+d}):  "
              f"V = {V:.6f}  (teórico = {Vmax_teorico:.6f}),  "
              f"Fx = {Fx:.2e},  Fy = {Fy:.2e}")
    print()


# ==============================================================
# PROGRAMA PRINCIPAL
# ==============================================================
if __name__ == "__main__":

    print("=" * 60)
    print("  Práctica Numérica 2 — Dispersión por potencial 2D")
    print(f"  Signo del potencial: {SIGNO:+d}  "
          f"({'repulsivo' if SIGNO==1 else 'atractivo'})")
    print("=" * 60)
    print()

    # (c) Verificar máximos analíticamente
    verificar_maximos()

    # (a) Graficar el potencial
    print("Graficando potencial...")
    graficar_potencial()

    # (g) Trayectorias en el plano xy
    print("Graficando trayectorias...")
    graficar_trayectorias()

    # (h) Espacio de fases
    print("Graficando espacio de fases...")
    graficar_espacio_de_fases()

    # (i) Ángulo de dispersión
    print("Graficando ángulo de dispersión...")
    graficar_angulo_dispersion()

    # (l) Retardo temporal
    print("Graficando retardo temporal...")
    graficar_retardo_temporal()

    print()
    print("¡Listo! Revisa los archivos PNG generados.")
    print()
    print("Para cambiar entre potencial repulsivo y atractivo,")
    print("  edita la variable SIGNO al inicio del script:")
    print("  SIGNO = +1   →  repulsivo")
    print("  SIGNO = -1   →  atractivo")