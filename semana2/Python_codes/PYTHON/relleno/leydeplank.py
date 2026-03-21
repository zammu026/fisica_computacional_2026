import numpy as np

def ley_planck(longitud_onda, temperatura):
    # Constantes físicas
    h = 6.626e-34  # Constante de Planck (J*s)
    c = 3.0e8      # Velocidad de la luz (m/s)
    k = 1.38e-23   # Constante de Boltzmann (J/K)
    
    # Fórmula: B(λ, T) = (2hc²) / (λ⁵ * (exp(hc / (λkT)) - 1))
    numerador = 2 * h * c**2
    exponente = (h * c) / (longitud_onda * k * temperatura)
    radiancia = numerador / (longitud_onda**5 * (np.exp(exponente) - 1))
    
    return radiancia

# Ejemplo: Radiancia para 500nm (luz visible) a 5800K (Temp. del Sol)
longitud = 500e-9 # metros
temp = 5800       # Kelvin
print(f"Radiancia: {ley_planck(longitud, temp):.2e} W/(sr m^3)")
