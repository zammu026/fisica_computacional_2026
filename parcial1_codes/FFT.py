import numpy as np
max = 2100
points = 1026
N = 16
isign = -1 # =1 to TF, 1 inverse TF

data = np.zeros(max)
dtr = np.zeros((points, 2))

def FFT(N, isign):
    n = 2*N
    for i in range(0, N):
        j = 2*i+1

        data[j] = dtr[i, 0]
        data[j+1] = dtr[i, 1]

        print('data.', data[j], data[j+1])
    j = 1
    for i in range(1, n+2, 2):
        if (i - j) < 0:
            tempr = data[j]
            tempi = data[j+1]

            data[j] = data[i]
            data[j+1] = data[i+1] #verificar esta parte

            data[i] = tempr
            data[i+1] = tempi
        m = n // 2
        while (m >= 2):
            if (j - m) <= 0:
                break

            j = j - m
            m = m // 2
        j = j + m

    print('\n Bit-Reversed Input Data ')

    for i in range(1, n+1, 2):
        print('%2d data[%2d] = %9.5f' % (i//2, i, data[i]))
    nmax = 2
    while (nmax < n):
        istep = 2*nmax
        theta = 6.2831853/(isign*nmax)
        sinth = np.sin(theta/2.)
        wstpr = -2.0*sinth**2
        wstpi = np.sin(theta)
        wr = 1.0
        wi = 0.0

#####################
        for m in range(1, nmax+1, 2):
            for i in range(m, n+1, istep):
                j = i+nmax
                tempr = wr*data[j] - wi*data[j+1]
                tempi = wr*data[j+1] + wi*data[j]
                data[j] = data[i] - tempr
                data[j+1] = data[i+1] - tempi
                data[i] = data[i] + tempr
                data[i+1] = data[i+1] + tempi  #####

            tempr = wr
            wr = wr*wstpr - wi*wstpi + wr
            wi = wi*wstpr + tempr*wstpi + wi
        nmax = istep
    for i in range(0, N):
        j = 2*i + 1
        dtr[i, 0] = data[j]
        dtr[i, 1] = data[j+1]

print('\n Input')
print('i Re part Im part')
for i in range(0, N):
    dtr[i, 0] = 1.0*i
    dtr[i, 1] = 1.0*i
    print('%2d %9.5f %9.5f' %(i, dtr[i, 0], dtr[i, 1]))
FFT(N, isign)
print('\n Fourier Transform \n i Re Im ')
for i in range(0, N):
    print('%2d %9.5f %9.5f' %(i, dtr[i, 0], dtr[i, 1]))