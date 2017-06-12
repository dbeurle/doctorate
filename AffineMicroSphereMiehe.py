
# Python script implementing the affine micro-sphere model presented
# in Miehe et al. 2004

import numpy as np

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Change the font size of the legend
mpl.rcParams['legend.fontsize'] = 18
mpl.rcParams['axes.labelsize'] = 18
mpl.rcParams['xtick.labelsize'] = 18
mpl.rcParams['ytick.labelsize'] = 18

def UnitSphereQuadrature():
    '''Returns the coordinates and weightings for the 21 point unit sphere quadrature scheme'''

    # Weightings for unit sphere
    w1, w2, w3 = 0.0265214244093, 0.0199301476312, 0.0250712367487
    w = [w1, w1, w1, w2, w2, w2, w2, w2, w2, w3, w3, w3, w3, w3, w3, w3, w3, w3, w3, w3, w3]

    # Directional cosines from unit sphere integration
    dc0, dc1, dc2 = 0.707106781187, 0.387907304067, 0.836095596749

    clist = [[1.0, 0.0, 0.0],   [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],  [dc0, dc0, 0.0],
             [dc0, -dc0, 0.0],  [dc0, 0.0, dc0], [dc0, 0.0, -dc0], [0.0, dc0, dc0],
             [0.0, dc0, -dc0],  [dc1, dc1, dc2], [dc1, dc1, -dc2], [dc1, -dc1, dc2],
             [dc1, -dc1, -dc2], [dc1, dc2, dc1], [dc1, dc2, -dc1], [dc1, -dc2, dc1],
             [dc1, -dc2, -dc1], [dc2, dc1, dc1], [dc2, dc1, -dc1], [dc2, -dc1, dc1],
             [dc2, -dc1, -dc1]]
    return clist, w

def UnimodularDecomposition(F):
    return np.linalg.det(F)**(-1.0/3.0) * F

def OuterProduct4(t):
    return np.array([[t[0]**4          , t[0]**2*t[1]**2  , t[0]**2*t[2]**2  , t[0]**2*t[1]*t[2], t[0]**3*t[2]     , t[0]**3*t[1]],
                     [t[0]**2*t[1]**2  , t[1]**4          , t[1]**2*t[2]**2  , t[1]**3*t[2]     , t[0]*t[1]**2*t[2], t[0]*t[1]**3],
                     [t[0]**2*t[2]**2  , t[1]**2*t[2]**2  , t[2]**4          , t[1]*t[2]**3     , t[0]*t[2]**3     , t[0]*t[1]*t[2]**2],
                     [t[0]**2*t[1]*t[2], t[1]**3*t[2]     , t[1]*t[2]**3     , t[1]**2*t[2]**2  , t[0]*t[1]*t[2]**2, t[0]*t[1]**2*t[2]],
                     [t[0]**3*t[2]     , t[0]*t[1]**2*t[2], t[0]*t[2]**3     , t[0]*t[1]*t[2]**2, t[0]**2*t[2]**2  , t[0]**2*t[1]*t[2]],
                     [t[0]**3*t[1]     , t[0]*t[1]**3     , t[0]*t[1]*t[2]**2, t[0]*t[1]**2*t[2], t[0]**2*t[1]*t[2], t[0]**2*t[1]**2]])

clist, w = UnitSphereQuadrature()

μ = 0.25*10**6  # Material parameter (Pa)
N = 64          # Number of segments

# Deformation gradient
F = np.eye(3)

uniaxialPlot = {}

for N in [16, 25, 36, 64, 100]:

    stress11_normalized = []
    stress22_normalized = []
    stress33_normalized = []

    # The range of values to plot the normalized stress
    λ_ua_range = np.arange(1.0, np.sqrt(N)*0.98, 0.01)

    # Uniaxial tension
    for λ_ua in λ_ua_range:

        # Deformation gradient for uniaxial test
        F[0, 0] = λ_ua
        F[1, 1] = 1.0 / np.sqrt(λ_ua)
        F[2, 2] = 1.0 / np.sqrt(λ_ua)

        τ = np.zeros((3, 3))  # Kirchhoff stress
        C = np.zeros((6, 6))  # Constitutive model

        # Obtain the unimodular part of the deformation gradient
        F_unimodular = UnimodularDecomposition(F)

        # Stretch directions and perform the integration
        for (λr, λs, λt), wl in zip(clist, w):

            ri = np.array((λr, λs, λt))

            # Deformed tangents
            t = np.matmul(F_unimodular, ri)

            # Affine microstretches
            λi = np.linalg.norm(t)

            # Macro-stresses (Kirchhoff)
            τ += μ * (3*N - λi**2) / (N - λi**2) * np.outer(t, t) * wl

            # Compute the macro-moduli
            C += μ * ( (λi**4 + 3*N**2) / (N - λi**2)**2 * λi**(-2) - (3*N-λi**2) / (N-λi**2) * λi**(-2) ) * OuterProduct4(t) * wl

            # Deviatoric projection

        # Accumulate and account for symmetry of quadrature
        stress11_normalized.append(τ[0, 0] * 2 / μ)
        stress22_normalized.append(τ[1, 1] * 2 / μ)
        stress33_normalized.append(τ[2, 2] * 2 / μ)

    uniaxialPlot[N] = stress11_normalized

    plt.plot(λ_ua_range, stress33_normalized, linewidth=2, label='N = '+ str(N))
    plt.grid(True)

plt.legend(loc=2)
plt.xticks(range(0, 11))
plt.xlim([0, 10])
plt.ylim([0, 350])
plt.xlabel('Uniaxial stretch')
plt.ylabel('Normalized stress (sigma11 / mu)')
plt.show()
