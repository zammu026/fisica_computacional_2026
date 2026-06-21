"""
§8.4 Growth Models — Landau & Páez
Computational Problems for Physics (CRC Press, 2018)
Física Computacional, UES — Ciclo I 2026

Implementa:
  8.4.1  Protein Folding como Self-Avoiding Walk (SAW)
  8.4.2  Fundamentos IFS: escalado, traslación, rotación
  8.4.3  Barnsley's Fern  (ec. 8.29)
  8.4.4  Self-Affine Tree (ec. 8.30)
"""

import numpy as np
import matplotlib.pyplot as plt
import random

np.random.seed(42)
random.seed(42)


# ══════════════════════════════════════════════════════════════
# 8.4.1  Protein Folding — Self-Avoiding Walk (SAW)
# ══════════════════════════════════════════════════════════════

def protein_fold(length=30, max_tries=3000):
    """
    Simula el plegado de proteínas como SAW en red 2D.

    Modelo HP (Hydrophobic-Polar):
      - Monomeros H (hidrofóbico) con prob 0.7
      - Monomeros P (polar)       con prob 0.3
      - Energía E = -epsilon * f,  f = pares H-H no consecutivos

    Parámetros
    ----------
    length    : int  Longitud de la cadena.
    max_tries : int  Número de conformaciones a explorar.

    Retorna
    -------
    best  : tuple (chain, types)  Conformación de mínima energía.
    best_e: int                   Energía mínima encontrada.
    """
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    best = None
    best_e = 1e9

    for _ in range(max_tries):
        chain = [(0, 0)]
        visited = {(0, 0)}
        types = ['H' if random.random() < 0.7 else 'P']
        stuck = False

        for _ in range(length - 1):
            x, y = chain[-1]
            # Solo sitios no visitados (aspecto "autoevitante")
            opts = [(x + dx, y + dy) for dx, dy in directions
                    if (x + dx, y + dy) not in visited]
            if not opts:
                stuck = True
                break
            nx, ny = random.choice(opts)
            chain.append((nx, ny))
            visited.add((nx, ny))
            types.append('H' if random.random() < 0.7 else 'P')

        if stuck:
            continue

        # Calcular energía: -1 por cada par H-H no enlazado consecutivo
        energy = 0
        for i, (x, y) in enumerate(chain):
            if types[i] == 'H':
                for dx, dy in directions:
                    nb = (x + dx, y + dy)
                    if nb in visited:
                        j = chain.index(nb)
                        if types[j] == 'H' and abs(i - j) > 1:
                            energy -= 1
        energy //= 2  # cada par se cuenta dos veces

        if energy < best_e:
            best_e = energy
            best = (chain[:], types[:])

    return best, best_e


# ══════════════════════════════════════════════════════════════
# 8.4.3  Barnsley's Fern  (ecuación 8.29)
# ══════════════════════════════════════════════════════════════

def barnsley_fern(n=100000):
    """
    Genera el helecho de Barnsley con 4 transformaciones afines.

    Probabilidades: 2% / 15% / 13% / 70%
    (tallo / fronda pequeña / fronda media / hoja principal)

    Referencia: Barnsley & Hurd (1992); Landau & Páez ec. (8.29)
    """
    x, y = 0.0, 0.0
    xs, ys = [], []

    for _ in range(n):
        r = random.random()
        if r < 0.02:                            # tallo — 2%
            x, y = 0.5, 0.27 * y
        elif r < 0.17:                          # fronda pequeña — 15%
            x, y = (-0.139 * x + 0.263 * y + 0.57,
                     0.246 * x + 0.224 * y - 0.036)
        elif r < 0.30:                          # fronda media — 13%
            x, y = (0.17 * x - 0.215 * y + 0.408,
                     0.222 * x + 0.176 * y + 0.0893)
        else:                                   # hoja principal — 70%
            x, y = (0.781 * x + 0.034 * y + 0.1075,
                    -0.032 * x + 0.739 * y + 0.27)
        xs.append(x)
        ys.append(y)

    return xs, ys


# ══════════════════════════════════════════════════════════════
# 8.4.4  Self-Affine Tree  (ecuación 8.30)
# ══════════════════════════════════════════════════════════════

def self_affine_tree(n=120000):
    """
    Genera un árbol fractal con 6 transformaciones auto-afines.

    Probabilidades: 10% / 10% / 20% / 20% / 20% / 20%

    Referencia: Landau & Páez ec. (8.30)
    """
    x, y = 0.0, 0.0
    xs, ys = [], []

    transforms = [
        (0.10, lambda x, y: (0.05 * x,  0.60 * y)),
        (0.10, lambda x, y: (0.05 * x, -0.50 * y + 1.0)),
        (0.20, lambda x, y: (0.46 * x - 0.15 * y,  0.39 * x + 0.38 * y + 0.6)),
        (0.20, lambda x, y: (0.47 * x - 0.15 * y,  0.17 * x + 0.42 * y + 1.1)),
        (0.20, lambda x, y: (0.43 * x + 0.28 * y, -0.25 * x + 0.45 * y + 1.0)),
        (0.20, lambda x, y: (0.42 * x + 0.26 * y, -0.35 * x + 0.31 * y + 0.7)),
    ]
    cumprob = np.cumsum([t[0] for t in transforms])

    for _ in range(n):
        r = random.random()
        for i, cp in enumerate(cumprob):
            if r < cp:
                x, y = transforms[i][1](x, y)
                break
        xs.append(x)
        ys.append(y)

    return xs, ys


# ══════════════════════════════════════════════════════════════
# Ejecución y visualización
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── 8.4.1  Protein Folding ────────────────────────────────
    print("8.4.1  Simulando plegado de proteína (SAW)...")
    result, energy = protein_fold(length=30, max_tries=3000)
    chain, types = result
    print(f"       Energía mínima encontrada: E = {energy}")

    fig, ax = plt.subplots(figsize=(5, 5), facecolor='#1a1a1a')
    ax.set_facecolor('#1a1a1a')
    xs_p = [p[0] for p in chain]
    ys_p = [p[1] for p in chain]
    ax.plot(xs_p, ys_p, '-', color='gray', lw=1.5, zorder=1)
    for (x, y), t in zip(chain, types):
        color = '#ff4444' if t == 'H' else '#ffffff'
        ax.scatter(x, y, s=120, color=color, zorder=2, edgecolors='none')
    ax.scatter([], [], s=60, color='#ff4444', label='H (hidrofóbico)')
    ax.scatter([], [], s=60, color='#ffffff', label='P (polar)')
    ax.legend(facecolor='#2a2a2a', labelcolor='white', fontsize=9)
    ax.set_title(f'Protein Folding SAW  (L=30, E={energy})',
                 color='white', fontsize=11)
    ax.axis('equal')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig('protein_fold.png', dpi=150, bbox_inches='tight',
                facecolor='#1a1a1a')
    plt.show()

    # ── 8.4.3  Barnsley's Fern ────────────────────────────────
    print("8.4.3  Generando helecho de Barnsley...")
    xs_f, ys_f = barnsley_fern(100000)

    fig, ax = plt.subplots(figsize=(4, 6), facecolor='#0a1a0a')
    ax.set_facecolor('#0a1a0a')
    ax.scatter(xs_f, ys_f, s=0.05, c='#55cc55', alpha=0.6, linewidths=0)
    ax.axis('off')
    ax.set_title("Barnsley's Fern\n(100 000 iteraciones)",
                 color='white', fontsize=11)
    plt.tight_layout()
    plt.savefig('barnsley_fern.png', dpi=150, bbox_inches='tight',
                facecolor='#0a1a0a')
    plt.show()

    # ── 8.4.4  Self-Affine Tree ───────────────────────────────
    print("8.4.4  Generando árbol auto-afín...")
    xs_t, ys_t = self_affine_tree(120000)

    fig, ax = plt.subplots(figsize=(4, 6), facecolor='#0a0f1a')
    ax.set_facecolor('#0a0f1a')
    ax.scatter(xs_t, ys_t, s=0.05, c='#88aaff', alpha=0.6, linewidths=0)
    ax.axis('off')
    ax.set_title('Self-Affine Tree\n(120 000 iteraciones)',
                 color='white', fontsize=11)
    plt.tight_layout()
    plt.savefig('self_affine_tree.png', dpi=150, bbox_inches='tight',
                facecolor='#0a0f1a')
    plt.show()

    print("Listo. Imágenes guardadas: protein_fold.png, barnsley_fern.png, self_affine_tree.png")