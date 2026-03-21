from wien_func import f, h, c, kB
from bisection import binary_search

# ── b) Resolver 5e^{-x} + x - 5 = 0 ────────────────────────────
x_sol = binary_search(f, a=1, b=10, eps=1e-6)
print(f"Solución x            = {x_sol:.6f}")
print(f"Comprobación f(x)     = {f(x_sol):.2e}")

# ── Constante de Wien: b = hc / (kB * x) ───────────────────────
b_wien = h * c / (kB * x_sol)
print(f"\nConstante de Wien b   = {b_wien:.4e} m·K")
print(f"  (valor aceptado)      = 2.8978e-03 m·K")

# ── c) Temperatura superficial del Sol (λ_max = 502 nm) ─────────
lam = 502e-9   # metros
T_sun = b_wien / lam
print(f"\nTemperatura del Sol    = {T_sun:.0f} K")
print(f"  (valor aceptado)      ≈ 5778 K")
