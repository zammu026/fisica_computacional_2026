import math
from scipy.constants import h, c, k, Wien
from newtonrapson import NewtonR

# Parámetros solicitados
x0 = 5            # Valor inicial cercano a la raíz no trivial (~4.96)
dx_num = 3.e-4       # Diferencial para la derivada numérica
eps = 1e-6           # Precisión de 10^-6
Nmax = 100

def f(x):
    # Ecuación trascendente de Wien: 5*exp(-x) + x - 5 = 0
    return 5 * math.exp(-x) + x - 5

x_sol = NewtonR(x0, dx_num, eps, Nmax)

# 2. Calcular la constante de Wien teórica (b = hc / xk)
# Comparamos con la constante b directa de scipy (Wien)
b_calculada = (h * c) / (x_sol * k)
print(f"Constante de Wien calculada: {b_calculada:.8e} m·K")
print(f"Constante de Wien (SciPy):    {Wien:.8e} m·K")

# 3. Temperatura del Sol (usando lambda_max aprox de 502 nm)
lambda_sol = 502e-9 
T_sol = b_calculada / lambda_sol

print(f"\n--- Resultado Final ---")
print(f"Temperatura estimada del Sol: {T_sol:.2f} K")
