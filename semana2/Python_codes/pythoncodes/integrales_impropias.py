import numpy as np
from scipy import integrate

# FUNCIONES DE CAMBIO DE VARIABLE
def _cambio_0_inf(f):
    """
    Intervalo [0, +inf) → [0, 1)
    Cambio: x = z/(1-z),  dx = dz/(1-z)^2
    """
    def integrando_transformado(z):
        # Evitamos z muy cercano a 1 para no dividir entre cero
        if np.isscalar(z):
            if z >= 1.0:
                return 0.0
            x = z / (1 - z)
            jacobiano = 1.0 / (1 - z)**2
            return f(x) * jacobiano
        else:
            z = np.asarray(z, dtype=float)
            mascara = z < 1.0
            resultado = np.zeros_like(z)
            z_valido = z[mascara]
            x = z_valido / (1 - z_valido)
            jacobiano = 1.0 / (1 - z_valido)**2
            resultado[mascara] = f(x) * jacobiano
            return resultado

    return integrando_transformado


def _cambio_menos_inf_0(f):
    """
    Intervalo (-inf, 0] → (0, 1]
    Cambio: x = -z/(1-z),  dx = dz/(1-z)^2
    """
    def integrando_transformado(z):
        if np.isscalar(z):
            if z >= 1.0:
                return 0.0
            x = -z / (1 - z)
            jacobiano = 1.0 / (1 - z)**2
            return f(x) * jacobiano
        else:
            z = np.asarray(z, dtype=float)
            mascara = z < 1.0
            resultado = np.zeros_like(z)
            z_valido = z[mascara]
            x = -z_valido / (1 - z_valido)
            jacobiano = 1.0 / (1 - z_valido)**2
            resultado[mascara] = f(x) * jacobiano
            return resultado

    return integrando_transformado


def _cambio_menos_inf_mas_inf(f):
    """
    Intervalo (-inf, +inf) → (-1, 1)
    Cambio: x = z/(1-z^2),  dx = (1+z^2)/(1-z^2)^2 dz
    """
    def integrando_transformado(z):
        if np.isscalar(z):
            if abs(z) >= 1.0:
                return 0.0
            x = z / (1 - z**2)
            jacobiano = (1 + z**2) / (1 - z**2)**2
            return f(x) * jacobiano
        else:
            z = np.asarray(z, dtype=float)
            mascara = np.abs(z) < 1.0
            resultado = np.zeros_like(z)
            z_valido = z[mascara]
            x = z_valido / (1 - z_valido**2)
            jacobiano = (1 + z_valido**2) / (1 - z_valido**2)**2
            resultado[mascara] = f(x) * jacobiano
            return resultado

    return integrando_transformado


def _cambio_a_inf(f, a):
    """
    Intervalo [a, +inf) → [0, 1)
    Cambio: x = a + z/(1-z),  dx = dz/(1-z)^2
    """
    def integrando_transformado(z):
        if np.isscalar(z):
            if z >= 1.0:
                return 0.0
            x = a + z / (1 - z)
            jacobiano = 1.0 / (1 - z)**2
            return f(x) * jacobiano
        else:
            z = np.asarray(z, dtype=float)
            mascara = z < 1.0
            resultado = np.zeros_like(z)
            z_valido = z[mascara]
            x = a + z_valido / (1 - z_valido)
            jacobiano = 1.0 / (1 - z_valido)**2
            resultado[mascara] = f(x) * jacobiano
            return resultado

    return integrando_transformado


def _cambio_menos_inf_b(f, b):
    """
    Intervalo (-inf, b] → (0, 1]
    Cambio: x = b - (1-z)/z,  dx = dz/z^2
    """
    def integrando_transformado(z):
        if np.isscalar(z):
            if z <= 0.0:
                return 0.0
            x = b - (1 - z) / z
            jacobiano = 1.0 / z**2
            return f(x) * jacobiano
        else:
            z = np.asarray(z, dtype=float)
            mascara = z > 0.0
            resultado = np.zeros_like(z)
            z_valido = z[mascara]
            x = b - (1 - z_valido) / z_valido
            jacobiano = 1.0 / z_valido**2
            resultado[mascara] = f(x) * jacobiano
            return resultado

    return integrando_transformado


# FUNCIONES PRINCIPALES (las que usa el usuario)

def integral_0_inf(f, mostrar_pasos=True):
    """
    Calcula:  ∫ f(x) dx  desde 0 hasta +infinito

    Parámetros:
        f             : función Python  f(x)
        mostrar_pasos : si True, imprime el procedimiento

    Retorna:
        resultado (float), error_estimado (float)

    Ejemplo:
        integral_0_inf(lambda x: np.exp(-x))
    """
    if mostrar_pasos:
        print("=" * 50)
        print("  INTEGRAL: ∫₀^∞ f(x) dx")
        print("  Cambio de variable: x = z/(1-z)")
        print("  Nuevo intervalo: [0, 1)")
        print("  Jacobiano: dx = dz/(1-z)²")
        print("=" * 50)

    g = _cambio_0_inf(f)
    resultado, error = integrate.quad(g, 0, 1, limit=200)

    if mostrar_pasos:
        print(f"  Resultado  = {resultado:.10f}")
        print(f"  Error est. = {error:.2e}")
        print()

    return resultado, error


def integral_menos_inf_0(f, mostrar_pasos=True):
    """
    Calcula:  ∫ f(x) dx  desde -infinito hasta 0

    Parámetros:
        f             : función Python  f(x)
        mostrar_pasos : si True, imprime el procedimiento

    Retorna:
        resultado (float), error_estimado (float)
    """
    if mostrar_pasos:
        print("=" * 50)
        print("  INTEGRAL: ∫₋∞^0 f(x) dx")
        print("  Cambio de variable: x = -z/(1-z)")
        print("  Nuevo intervalo: (0, 1]")
        print("  Jacobiano: dx = dz/(1-z)²")
        print("=" * 50)

    g = _cambio_menos_inf_0(f)
    resultado, error = integrate.quad(g, 0, 1, limit=200)

    if mostrar_pasos:
        print(f"  Resultado  = {resultado:.10f}")
        print(f"  Error est. = {error:.2e}")
        print()

    return resultado, error


def integral_menos_inf_mas_inf(f, mostrar_pasos=True):
    """
    Calcula:  ∫ f(x) dx  desde -infinito hasta +infinito

    Parámetros:
        f             : función Python  f(x)
        mostrar_pasos : si True, imprime el procedimiento

    Retorna:
        resultado (float), error_estimado (float)
    """
    if mostrar_pasos:
        print("=" * 50)
        print("  INTEGRAL: ∫₋∞^+∞ f(x) dx")
        print("  Cambio de variable: x = z/(1-z²)")
        print("  Nuevo intervalo: (-1, 1)")
        print("  Jacobiano: dx = (1+z²)/(1-z²)² dz")
        print("=" * 50)

    g = _cambio_menos_inf_mas_inf(f)
    resultado, error = integrate.quad(g, -1, 1, limit=200)

    if mostrar_pasos:
        print(f"  Resultado  = {resultado:.10f}")
        print(f"  Error est. = {error:.2e}")
        print()

    return resultado, error


def integral_a_inf(f, a, mostrar_pasos=True):
    """
    Calcula:  ∫ f(x) dx  desde a hasta +infinito

    Parámetros:
        f             : función Python  f(x)
        a             : límite inferior (número)
        mostrar_pasos : si True, imprime el procedimiento

    Retorna:
        resultado (float), error_estimado (float)

    Ejemplo:
        integral_a_inf(lambda x: 1/x**2, a=1)
    """
    if mostrar_pasos:
        print("=" * 50)
        print(f"  INTEGRAL: ∫_{a}^∞ f(x) dx")
        print(f"  Cambio de variable: x = {a} + z/(1-z)")
        print("  Nuevo intervalo: [0, 1)")
        print("  Jacobiano: dx = dz/(1-z)²")
        print("=" * 50)

    g = _cambio_a_inf(f, a)
    resultado, error = integrate.quad(g, 0, 1, limit=200)

    if mostrar_pasos:
        print(f"  Resultado  = {resultado:.10f}")
        print(f"  Error est. = {error:.2e}")
        print()

    return resultado, error


def integral_menos_inf_b(f, b, mostrar_pasos=True):
    """
    Calcula:  ∫ f(x) dx  desde -infinito hasta b

    Parámetros:
        f             : función Python  f(x)
        b             : límite superior (número)
        mostrar_pasos : si True, imprime el procedimiento

    Retorna:
        resultado (float), error_estimado (float)

    Ejemplo:
        integral_menos_inf_b(lambda x: np.exp(x), b=0)
    """
    if mostrar_pasos:
        print("=" * 50)
        print(f"  INTEGRAL: ∫₋∞^{b} f(x) dx")
        print(f"  Cambio de variable: x = {b} - (1-z)/z")
        print("  Nuevo intervalo: (0, 1]")
        print("  Jacobiano: dx = dz/z²")
        print("=" * 50)

    g = _cambio_menos_inf_b(f, b)
    resultado, error = integrate.quad(g, 0, 1, limit=200)

    if mostrar_pasos:
        print(f"  Resultado  = {resultado:.10f}")
        print(f"  Error est. = {error:.2e}")
        print()

    return resultado, error


# FUNCIÓN INTELIGENTE (detecta el intervalo)

def integrar(f, a, b, mostrar_pasos=True):
    """
    Función principal que detecta automáticamente el tipo
    de intervalo y aplica el cambio de variable correcto.

    Parámetros:
        f             : función Python  f(x)
        a             : límite inferior (número o -np.inf)
        b             : límite superior (número o np.inf)
        mostrar_pasos : si True, imprime el procedimiento

    Retorna:
        resultado (float), error_estimado (float)

    Ejemplos:
        integrar(lambda x: np.exp(-x),    a=0,        b=np.inf)
        integrar(lambda x: np.exp(x),     a=-np.inf,  b=0)
        integrar(lambda x: 1/(1+x**2),    a=-np.inf,  b=np.inf)
        integrar(lambda x: 1/x**2,        a=1,        b=np.inf)
        integrar(lambda x: np.exp(x),     a=-np.inf,  b=-1)
    """
    inf = np.inf

    # Caso 1: [0, +inf)
    if a == 0 and b == inf:
        return integral_0_inf(f, mostrar_pasos)

    # Caso 2: (-inf, 0]
    elif a == -inf and b == 0:
        return integral_menos_inf_0(f, mostrar_pasos)

    # Caso 3: (-inf, +inf)
    elif a == -inf and b == inf:
        return integral_menos_inf_mas_inf(f, mostrar_pasos)

    # Caso 4: [a, +inf)  con a finito
    elif a != -inf and b == inf:
        return integral_a_inf(f, a, mostrar_pasos)

    # Caso 5: (-inf, b]  con b finito
    elif a == -inf and b != inf:
        return integral_menos_inf_b(f, b, mostrar_pasos)

    else:
        raise ValueError(
            f"Intervalo [{a}, {b}] no es impropio o no está soportado.\n"
            "Usa scipy.integrate.quad para integrales ordinarias."
        )
