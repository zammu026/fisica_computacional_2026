import numpy as np
def simpson(f,a,b,n):
	if n%2==1:
		n += 1
	x = np.linspace(a,b,n+1)
	h = (b-a)/n
	y = f(x)
	I = h/3*(y[0]+y[n]+4*np.sum(y[1:n:2])+2*np.sum(y[2:n-1:2]))
	return I
