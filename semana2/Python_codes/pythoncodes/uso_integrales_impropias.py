#MODULO PARA DEFINIR LAS FUNCIONES
import numpy as np
import integrales_impropias as ii

#───────────────────────────────────────────
print("EJEMPLO 1: ∫₀^∞  e^(-x) dx")
print("Resultado exacto = 1")
#La que use  para lo de la integral de plank
resultado, error = ii.integral_0_inf(
    f = lambda x: x**3/(np.exp(x)-1)
)
print(f"Coincide con el exacto: {abs(resultado - 1) < 1e-8}")
print()


# ─────────────────────────────────────────────
print("EJEMPLO 2: ∫₁^∞  1/x² dx")
print("Resultado exacto = 1")

resultado, error = ii.integral_a_inf(
    f = lambda x: 1 / x**2,
    a = 1
)
print(f"Coincide con el exacto: {abs(resultado - 1) < 1e-8}")
print()

# ─────────────────────────────────────────────
print("EJEMPLO 3: ∫₋∞^0  e^x dx")
print("Resultado exacto = 1")

resultado, error = ii.integral_menos_inf_0(
    f = lambda x: np.exp(x)
)
print(f"Coincide con el exacto: {abs(resultado - 1) < 1e-8}")
print()


# ─────────────────────────────────────────────
print("EJEMPLO 4: ∫₋∞^+∞  1/(1+x²) dx")
print(f"Resultado exacto = π ≈ {np.pi:.10f}")

resultado, error = ii.integral_menos_inf_mas_inf(
    f = lambda x: 1 / (1 + x**2)
)
print(f"Coincide con el exacto: {abs(resultado - np.pi) < 1e-8}")
print()


# ─────────────────────────────────────────────
print("EJEMPLO 5: ∫₋∞^+∞  e^(-x²) dx  (Integral de Gauss)")
print(f"Resultado exacto = √π ≈ {np.sqrt(np.pi):.10f}")

resultado, error = ii.integral_menos_inf_mas_inf(
    f = lambda x: np.exp(-x**2)
)
print(f"Coincide con el exacto: {abs(resultado - np.sqrt(np.pi)) < 1e-8}")
print()


# ─────────────────────────────────────────────
print("EJEMPLO 6: ∫₋∞^(-1)  e^x dx")
print(f"Resultado exacto = e^(-1) ≈ {np.exp(-1):.10f}")

resultado, error = ii.integral_menos_inf_b(
    f = lambda x: np.exp(x),
    b = -1
)
print(f"Coincide con el exacto: {abs(resultado - np.exp(-1)) < 1e-8}")
print()


# USANDO LA FUNCIÓN INTELIGENTE 'integrar'  (detecta automáticamente el tipo de intervalo)
print("═" * 50)
print("USANDO LA FUNCIÓN INTELIGENTE: ii.integrar(f, a, b)")
print("═" * 50)
print()

casos = [
    {
        "descripcion" : "∫₀^∞  x·e^(-x²) dx",
        "f"           : lambda x: x * np.exp(-x**2),
        "a"           : 0,
        "b"           : np.inf,
        "exacto"      : 0.5
    },
    {
        "descripcion" : "∫₋∞^∞  e^(-x²/2) dx",
        "f"           : lambda x: np.exp(-x**2 / 2),
        "a"           : -np.inf,
        "b"           : np.inf,
        "exacto"      : np.sqrt(2 * np.pi)
    },
    {
        "descripcion" : "∫₂^∞  1/x³ dx",
        "f"           : lambda x: 1 / x**3,
        "a"           : 2,
        "b"           : np.inf,
        "exacto"      : 1/8
    },
]

for caso in casos:
    print(f"  >> {caso['descripcion']}")
    print(f"     Exacto = {caso['exacto']:.10f}")

    res, err = ii.integrar(
        f             = caso["f"],
        a             = caso["a"],
        b             = caso["b"],
        mostrar_pasos = True       # pon False para ver solo el resultado
    )
    print(f"Correcto: {abs(res - caso['exacto']) < 1e-6}")
    print()


print("Integral de Plank")
print("═" * 50)
print()

print("-"*50)
def mi_funcion(x):
	return x**3/(np.exp(x)-1)

res, err = ii.integrar(mi_funcion, a=0, b=np.inf)
print(f"Correcto: {abs(res - 0.5) < 1e-6}")
print()

