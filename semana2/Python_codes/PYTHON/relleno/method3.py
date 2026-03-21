import numpy as np
def montecarlo(f,a,b,N):
	x=np.random.uniform(a,b,N)
	I=(b-a)*np.mean(f(x))
	return I
