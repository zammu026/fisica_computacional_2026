# DERIVADAS POR DEFINICIÓN 
def central_deriv(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)
def forward_deriv(f, x, h):
    return (f(x + h) - f(x)) / h
def backward_deriv(f, x, h):
    return (f(x) - f(x - h)) / h    
def second_deriv(f, x, h):
    return (f(x + h) - 2 * f(x) + f(x - h)) / (h ** 2)
print("Derivada central:", central_deriv(lambda x: x**2, 1, 0.01))
print("Derivada hacia adelante:", forward_deriv(lambda x: x**2, 1, 0.01))
print("Derivada hacia atrás:", backward_deriv(lambda x: x**2, 1, 0.01))
print("Segunda derivada:", second_deriv(lambda x: x**2, 1, 0.01))   

'''
 def f(x):
    return x**2  ------> f = lambda x: x**2

# ... (las definiciones de las funciones derivadas) ...

print("Derivada central:", central_deriv(f, 1, 0.01))
'''