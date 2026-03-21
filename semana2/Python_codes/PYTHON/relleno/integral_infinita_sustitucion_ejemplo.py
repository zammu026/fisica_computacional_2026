import numpy as np
from scipy.integrate import quad

def integral_infinita_sustitucion(f):
    """
    Calcula la integral de 0 a infinito de f(x) dx 
    aplicando el cambio de variable x = t / (1 - t).
    """
    # Definimos el nuevo integrando: f(x(t)) * dx/dt
    integrando_t = lambda t: f(t / (1 - t)) * (1 / (1 - t)**2)
    
    # Los límites ahora son de 0 a 1
    # Usamos eps (un valor muy pequeño) para evitar división por cero en 1
    resultado, error = quad(integrando_t, 0, 1)
    return resultado

# --- Ejemplo de uso -  Definimos una función general, por ejemplo: f(x) = 1 / (1 + x)^2
f_ejemplo = lambda x: 1 / (1 + x)**2

resultado = integral_infinita_sustitucion(f_ejemplo)
print(f"Resultado de la integral: {resultado}")
