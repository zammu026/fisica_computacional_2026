import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

N = 300
dx = 0.4
dt = 0.1
eps = 0.2
mu = 0.1

x = np.arange(N) * dx

# condiciones iniciales (pulso del soliton)
#u = 0.5 * (1 - np.tanh(x/5 - 5)) # tipo soliton (default)
#u = np.exp(-(x-30)**2/50) # gaussiano
#u = np.exp(-(x-20)**2/40) + 0.5*np.exp(-(x-60)**2/40) # dos solitones
#u = np.zeros_like(x) # rectangular
#u[60:100] = 1
u = 0.2*np.random.randn(N)

u_old = u.copy()
u_new = np.zeros_like(u)

fig, ax = plt.subplots()
line, = ax.plot(x, u)
#ax.set_ylim(-0.2, 10.2)

def update(frame):
    global u, u_old, u_new

    for i in range(2, N-2):
        nonlinear = (u[i+1] + u[i] + u[i-1]) * (u[i] - u[i-1])
        dispersion = (u[i+2] + 2*u[i-1] - 2*u[i+1] - u[i-2])

        u_new[i] = (u_old[i] - eps*dt/(3*dx) * nonlinear - mu*dt/(dx**3)*dispersion)

    u_old = u.copy()
    u = u_new.copy()

    line.set_ydata(u)
    return line,
ani = FuncAnimation(fig, update, frames = 200)
plt.show()