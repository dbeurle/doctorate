
import numpy as np

Voigt2d = [[0, 0], [1, 1], [0, 1]]
Voigt3d = [[0, 0], [1, 1], [2, 2], [1, 2], [0, 2], [0, 1]]


def Identity2_2():
    return np.eye(2)

def Identity2_3():
    return np.eye(3)

def Identity4_3():
    I4_2 = np.zeros((3, 3, 3, 3))
    for i in [0, 1, 2]:
        for j in [0, 1, 2]:
            for k in [0, 1, 2]:
                for l in [0, 1, 2]:
                    δik = float(i == k)
                    δjl = float(j == l)
                    δil = float(i == l)
                    δjk = float(j == k)

                    I4_2[i,j,k,l] = 0.5 * (δik * δjl + δil * δjk)
    return I4_2

def Identity4_2():
    I4_2 = np.zeros((2, 2, 2, 2))
    for i in [0, 1]:
        for j in [0, 1]:
            for k in [0, 1]:
                for l in [0, 1]:
                    δik = float(i == k)
                    δjl = float(j == l)
                    δil = float(i == l)
                    δjk = float(j == k)

                    I4_2[i,j,k,l] = 0.5 * (δik * δjl + δil * δjk)
    return I4_2

def DeviatoricIdentity4_2():
    I4_2 = np.zeros((2, 2, 2, 2))
    for i in range(0, 2):
        for j in range(0, 2):
            for k in range(0, 2):
                for l in range(0, 2):
                    δij = float(i == j)
                    δik = float(i == k)
                    δil = float(i == l)
                    δjk = float(j == k)
                    δkl = float(k == l)
                    δjl = float(j == l)

                    I4_2[i,j,k,l] = 0.5*( δik*δjl + δik*δjk ) - 1.0/3.0*δij*δkl
    return I4_2

def DeviatoricIdentity4_3():
    I4_3 = np.zeros((3, 3, 3, 3))
    for i in range(0, 3):
        for j in range(0, 3):
            for k in range(0, 3):
                for l in range(0, 3):
                    δij = float(i == j)
                    δik = float(i == k)
                    δil = float(i == l)
                    δjk = float(j == k)
                    δkl = float(k == l)
                    δjl = float(j == l)

                    I4_3[i,j,k,l] = 0.5*( δik*δjl + δik*δjk ) - 1.0/3.0*δij*δkl
    return I4_3

def DeviatoricIdentity4_2V():
    return Voigt4_2(DeviatoricIdentity4_2())

def Voigt4_2(T):
    Tv = np.zeros((3, 3))
    for i in range(0, 3):
        a, b = Voigt2d[i]
        for j in range(0, 3):
            c, d = Voigt2d[j]
            Tv[i, j] = T[a, b, c, d]
    return Tv

def Voigt4_3(T):

    Tv = np.zeros((6, 6))

    for i in range(0, 6):
        a, b = Voigt3d[i]
        for j in range(0, 6):
            c, d = Voigt3d[j]
            Tv[i, j] = T[a, b, c, d]
    return Tv
