import numpy as np
import matplotlib.pyplot as plt

Nx = 70
Ny = 24

L_beam = 8
H_beam = 4

v0 = 1.0
R = 0.1

omega = 0.3

max_iter = 5000
tol = 1e-3

u = np.zeros((Nx, Ny))
w = np.zeros((Nx, Ny))

# posicion viga
x0 = 20
x1 = x0 + L_beam

y0 = (Ny // 2) - (H_beam // 2)
y1 = y0 + H_beam

def boundary_conditions(): # funciones de rutina
    # entradas 
    for j in range(Ny):
        u[0, j] = v0 * j
        
    #salida
    u[-1, :] = u[-2, :]
    w[-1, :] = w[-2, :]

    # superficie superior
    u[:, -1] = v0 * (Ny - 1)

    # inferior
    u[:, 0] = 0

    #viga
    u[x0:x1, y0:y1] = u[x0, y0]

def relax():
    max_residual = 0
    for i in range(1, Nx-1):
        for j in range(1, Ny-1):

            # saltar interior de la viga
            if x0 <= i < x1 and y0 <= j < y1:
                continue 

# funcion u(vx, vy) y w(vx, vy)

            u_new = 0.25 * (
                u[i+1, j] + u[i-1, j] +
                u[i, j+1] + u[i, j-1] +
                w[i, j]
            )        
            ru = u_new - u[i, j]
            u[i, j] += omega * ru

            a1 = w[i+1, j] + w[i-1, j]
            a2 = w[i, j+1] + w[i, j-1]

            a3 = (R/4.0) * ((u[i, j+1] - u[i, j-1]) *
                    (w[i+1, j] - w[i-1, j]) - (u[i+1, j] - u[i-1, j]) *
                    (w[i, j+1] - w[i, j-1]))
            w_new = 0.25 * (a1 + a2 + a3)
            rw = w_new - w[i, j]
            w[i, j] += omega * rw

            max_residual = max (
                max_residual, 
                abs(ru),
                abs(rw)
            )
            return max_residual

for it in range(max_iter):
    boundary_conditions()
    residual = relax()

    if it % 100 == 0:
        print(f"Iteracion {it}")
        print(
            "u upstream = ",
            u[x0 - 5, Ny // 2]
        )
        print(
            "u arriba = ",
            u[x0 + L_beam // 2, y1 + 2]
        )
        print(
            "u downstream = ",
            u[x1 + 5, Ny // 2]
        )
        print("Residual = ", residual)
        print("----------------------------------------")

    if residual < tol:
        print("Convergencia alcanzada")
        break