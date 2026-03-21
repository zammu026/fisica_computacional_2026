import numpy as np

def trapecio (f,a,b,n):
	x = np.linspace(a,b,n+1)
	h =(b-a)/n
	y = f(x)

	I = h*(0.5*y[0]+np.sum(y[1:n])+0.5*y[n])
	return I


