from sympy import symbols, Sum, oo
n = symbols('n')
# Ejemplo: Suma de 1/n desde 1 hasta k
k = symbols('k')
termino_general = 1/n**2
suma_general = Sum(termino_general, (n, 1, k))
print(suma_general.subs(k, 10).doit().evalf()) 