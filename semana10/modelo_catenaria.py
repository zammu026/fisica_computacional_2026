# modelo catenaria
modelo = 'exponencial' # catenaria
if modelo == 'exponencial':
    rho = rho0 * np.exp(alpha * x)
    T = T0 * np.exp(alpha * x)

elif modelo == 'catenaria':
    g = 9.8
    rho = rho0 * np.ones_likes(x)
    T = T0 * np.cosh(rho0 * g * x/T0)

    