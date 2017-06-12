
# Implementation of the Non-Affine microsphere model by Miehe 2004

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

import json
from pprint import pprint

import vtk
from vtk import *

# My crappy libraries
from Q1Interpolations import LagrangeH8
from libtensor import *

np.set_printoptions(linewidth=160)

nodal_dofs = 3

def VolumetricStrainDerivative1(J, K):
    return K / 2.0 * (J - 1.0 / J)


def VolumetricStrainDerivative2(J, K):
    return K / 2.0 * ( 1.0 + 1.0 / (J**2) )


def IdentityMatrix():
    return np.array([[1.0, 0, 0, 0, 0, 0],
                     [0, 1.0, 0, 0, 0, 0],
                     [0, 0, 1.0, 0, 0, 0],
                     [0, 0, 0, 0.5, 0, 0],
                     [0, 0, 0, 0, 0.5, 0],
                     [0, 0, 0, 0, 0, 0.5]])


def ComputeDeviatoricTangent(C, τ):
    return np.array([[-2*(C[0,1] - 2*(τ[0, 0] + τ[1, 1])/3.0)/9.0 - 2*(C[0,2] - 2*(τ[0, 0] + τ[2, 2])/3.0)/9.0 - 2*(C[1,0] - 2*(τ[0, 0] + τ[1, 1])/3.0)/9.0 + (C[1,2] - 2*(τ[1, 1] + τ[2, 2])/3.0)/9.0 - 2*(C[2,0] - 2*(τ[0, 0] + τ[2, 2])/3.0)/9.0 + (C[2,1] - 2*(τ[1, 1] + τ[2, 2])/3.0)/9.0 + 4*(C[0,0] + 2*(τ[0, 0] + τ[1, 1] + τ[2, 2])/3.0 - 4*τ[0, 0]/3.0)/9.0 + (C[1,1] + 2*(τ[0, 0] + τ[1, 1] + τ[2, 2])/3.0 - 4*τ[1, 1]/3.0)/9.0 + (C[2,2] + 2*(τ[0, 0] + τ[1, 1] + τ[2, 2])/3.0 - 4*τ[2, 2]/3.0)/9.0,
                       4*(C[0,1] - 2*(τ[0, 0] + τ[1, 1])/3.0)/9.0 - 2*(C[0,2] - 2*(τ[0, 0] + τ[2, 2])/3.0)/9.0 + (C[1,0] - 2*(τ[0, 0] + τ[1, 1])/3.0)/9.0 + (C[1,2] - 2*(τ[1, 1] + τ[2, 2])/3.0)/9.0 + (C[2,0] - 2*(τ[0, 0] + τ[2, 2])/3.0)/9.0 - 2*(C[2,1] - 2*(τ[1, 1] + τ[2, 2])/3.0)/9.0 - 2*(C[0,0] + 2*(τ[0, 0] + τ[1, 1] + τ[2, 2])/3.0 - 4*τ[0, 0]/3.0)/9.0 - 2*(C[1,1] + 2*(τ[0, 0] + τ[1, 1] + τ[2, 2])/3.0 - 4*τ[1, 1]/3.0)/9.0 + (C[2,2] + 2*(τ[0, 0] + τ[1, 1] + τ[2, 2])/3.0 - 4*τ[2, 2]/3.0)/9.0,
                      -2*(C[0,1] - 2*(τ[0, 0] + τ[1, 1])/3.0)/9.0 + 4*(C[0,2] - 2*(τ[0, 0] + τ[2, 2])/3.0)/9.0 + (C[1,0] - 2*(τ[0, 0] + τ[1, 1])/3.0)/9.0 - 2*(C[1,2] - 2*(τ[1, 1] + τ[2, 2])/3.0)/9.0 + (C[2,0] - 2*(τ[0, 0] + τ[2, 2])/3.0)/9.0 + (C[2,1] - 2*(τ[1, 1] + τ[2, 2])/3.0)/9.0 - 2*(C[0,0] + 2*(τ[0, 0] + τ[1, 1] + τ[2, 2])/3.0 - 4*τ[0, 0]/3.0)/9.0 + (C[1,1] + 2*(τ[0, 0] + τ[1, 1] + τ[2, 2])/3.0 - 4*τ[1, 1]/3.0)/9.0 - 2*(C[2,2] + 2*(τ[0, 0] + τ[1, 1] + τ[2, 2])/3.0 - 4*τ[2, 2]/3.0)/9.0,
                      (C[0,3] - 2*τ[1, 2]/3.0)/3.0 + (C[0,3] - 2*τ[2, 1]/3.0)/3.0 - (C[1,3] - 2*τ[1, 2]/3.0)/6.0 - (C[1,3] - 2*τ[2, 1]/3.0)/6.0 - (C[2,3] - 2*τ[1, 2]/3.0)/6.0 - (C[2,3] - 2*τ[2, 1]/3.0)/6.0,
                      (C[0,4] - 2*τ[0, 2]/3.0)/3.0 + (C[0,4] - 2*τ[2, 0]/3.0)/3.0 - (C[1,4] - 2*τ[0, 2]/3.0)/6.0 - (C[1,4] - 2*τ[2, 0]/3.0)/6.0 - (C[2,4] - 2*τ[0, 2]/3.0)/6.0 - (C[2,4] - 2*τ[2, 0]/3.0)/6.0,
                      (C[0,5] - 2*τ[0, 1]/3.0)/3.0 + (C[0,5] - 2*τ[1, 0]/3.0)/3.0 - (C[1,5] - 2*τ[0, 1]/3.0)/6.0 - (C[1,5] - 2*τ[1, 0]/3.0)/6.0 - (C[2,5] - 2*τ[0, 1]/3.0)/6.0 - (C[2,5] - 2*τ[1, 0]/3.0)/6.0],
                     [(C[0,1] - 2*(τ[0, 0] + τ[1, 1])/3.0)/9.0 + (C[0,2] - 2*(τ[0, 0] + τ[2, 2])/3.0)/9.0 + 4*(C[1,0] - 2*(τ[0, 0] + τ[1, 1])/3.0)/9.0 - 2*(C[1,2] - 2*(τ[1, 1] + τ[2, 2])/3.0)/9.0 - 2*(C[2,0] - 2*(τ[0, 0] + τ[2, 2])/3.0)/9.0 + (C[2,1] - 2*(τ[1, 1] + τ[2, 2])/3.0)/9.0 - 2*(C[0,0] + 2*(τ[0, 0] + τ[1, 1] + τ[2, 2])/3.0 - 4*τ[0, 0]/3.0)/9.0 - 2*(C[1,1] + 2*(τ[0, 0] + τ[1, 1] + τ[2, 2])/3.0 - 4*τ[1, 1]/3.0)/9.0 + (C[2,2] + 2*(τ[0, 0] + τ[1, 1] + τ[2, 2])/3.0 - 4*τ[2, 2]/3.0)/9.0,
                      -2*(C[0,1] - 2*(τ[0, 0] + τ[1, 1])/3.0)/9.0 + (C[0,2] - 2*(τ[0, 0] + τ[2, 2])/3.0)/9.0 - 2*(C[1,0] - 2*(τ[0, 0] + τ[1, 1])/3.0)/9.0 - 2*(C[1,2] - 2*(τ[1, 1] + τ[2, 2])/3.0)/9.0 + (C[2,0] - 2*(τ[0, 0] + τ[2, 2])/3.0)/9.0 - 2*(C[2,1] - 2*(τ[1, 1] + τ[2, 2])/3.0)/9.0 + (C[0,0] + 2*(τ[0, 0] + τ[1, 1] + τ[2, 2])/3.0 - 4*τ[0, 0]/3.0)/9.0 + 4*(C[1,1] + 2*(τ[0, 0] + τ[1, 1] + τ[2, 2])/3.0 - 4*τ[1, 1]/3.0)/9.0 + (C[2,2] + 2*(τ[0, 0] + τ[1, 1] + τ[2, 2])/3.0 - 4*τ[2, 2]/3.0)/9.0,
                      (C[0,1] - 2*(τ[0, 0] + τ[1, 1])/3.0)/9.0 - 2*(C[0,2] - 2*(τ[0, 0] + τ[2, 2])/3.0)/9.0 - 2*(C[1,0] - 2*(τ[0, 0] + τ[1, 1])/3.0)/9.0 + 4*(C[1,2] - 2*(τ[1, 1] + τ[2, 2])/3.0)/9.0 + (C[2,0] - 2*(τ[0, 0] + τ[2, 2])/3.0)/9.0 + (C[2,1] - 2*(τ[1, 1] + τ[2, 2])/3.0)/9.0 + (C[0,0] + 2*(τ[0, 0] + τ[1, 1] + τ[2, 2])/3.0 - 4*τ[0, 0]/3.0)/9.0 - 2*(C[1,1] + 2*(τ[0, 0] + τ[1, 1] + τ[2, 2])/3.0 - 4*τ[1, 1]/3.0)/9.0 - 2*(C[2,2] + 2*(τ[0, 0] + τ[1, 1] + τ[2, 2])/3.0 - 4*τ[2, 2]/3.0)/9.0,
                      -(C[0,3] - 2*τ[1, 2]/3.0)/6.0 - (C[0,3] - 2*τ[2, 1]/3.0)/6.0 + (C[1,3] - 2*τ[1, 2]/3.0)/3.0 + (C[1,3] - 2*τ[2, 1]/3.0)/3.0 - (C[2,3] - 2*τ[1, 2]/3.0)/6.0 - (C[2,3] - 2*τ[2, 1]/3.0)/6.0,
                      -(C[0,4] - 2*τ[0, 2]/3.0)/6.0 - (C[0,4] - 2*τ[2, 0]/3.0)/6.0 + (C[1,4] - 2*τ[0, 2]/3.0)/3.0 + (C[1,4] - 2*τ[2, 0]/3.0)/3.0 - (C[2,4] - 2*τ[0, 2]/3.0)/6.0 - (C[2,4] - 2*τ[2, 0]/3.0)/6.0,
                      -(C[0,5] - 2*τ[0, 1]/3.0)/6.0 - (C[0,5] - 2*τ[1, 0]/3.0)/6.0 + (C[1,5] - 2*τ[0, 1]/3.0)/3.0 + (C[1,5] - 2*τ[1, 0]/3.0)/3.0 - (C[2,5] - 2*τ[0, 1]/3.0)/6.0 - (C[2,5] - 2*τ[1, 0]/3.0)/6.0],
                     [(C[0,1] - 2*(τ[0, 0]+ τ[1, 1])/3.0)/9.0 + (C[0,2] - 2*(τ[0, 0] + τ[2, 2])/3.0)/9.0 - 2*(C[1,0] - 2*(τ[0, 0] + τ[1, 1])/3.0)/9.0 + (C[1,2] - 2*(τ[1, 1] + τ[2, 2])/3.0)/9.0 + 4*(C[2,0] - 2*(τ[0, 0] + τ[2, 2])/3.0)/9.0 - 2*(C[2,1] - 2*(τ[1, 1] + τ[2, 2])/3.0)/9.0 - 2*(C[0,0] + 2*(τ[0, 0] + τ[1, 1] + τ[2, 2])/3.0 - 4*τ[0, 0]/3.0)/9.0 + (C[1,1] + 2*(τ[0, 0] + τ[1, 1] + τ[2, 2])/3.0 - 4*τ[1, 1]/3.0)/9.0 - 2*(C[2,2] + 2*(τ[0, 0] + τ[1, 1] + τ[2, 2])/3.0 - 4*τ[2, 2]/3.0)/9.0,
                      -2*(C[0,1] - 2*(τ[0, 0] + τ[1, 1])/3.0)/9.0 + (C[0,2] - 2*(τ[0, 0] + τ[2, 2])/3.0)/9.0 + (C[1,0] - 2*(τ[0, 0] + τ[1, 1])/3.0)/9.0 + (C[1,2] - 2*(τ[1, 1] + τ[2, 2])/3.0)/9.0 - 2*(C[2,0] - 2*(τ[0, 0] + τ[2, 2])/3.0)/9.0 + 4*(C[2,1] - 2*(τ[1, 1] + τ[2, 2])/3.0)/9.0 + (C[0,0] + 2*(τ[0, 0] + τ[1, 1] + τ[2, 2])/3.0 - 4*τ[0, 0]/3.0)/9.0 - 2*(C[1,1] + 2*(τ[0, 0] + τ[1, 1] + τ[2, 2])/3.0 - 4*τ[1, 1]/3.0)/9.0 - 2*(C[2,2] + 2*(τ[0, 0] + τ[1, 1] + τ[2, 2])/3.0 - 4*τ[2, 2]/3.0)/9.0,
                      (C[0,1] - 2*(τ[0, 0] + τ[1, 1])/3.0)/9.0 - 2*(C[0,2] - 2*(τ[0, 0] + τ[2, 2])/3.0)/9.0 + (C[1,0] - 2*(τ[0, 0] + τ[1, 1])/3.0)/9.0 - 2*(C[1,2] - 2*(τ[1, 1] + τ[2, 2])/3.0)/9.0 - 2*(C[2,0] - 2*(τ[0, 0] + τ[2, 2])/3.0)/9.0 - 2*(C[2,1] - 2*(τ[1, 1] + τ[2, 2])/3.0)/9.0 + (C[0,0] + 2*(τ[0, 0] + τ[1, 1] + τ[2, 2])/3.0 - 4*τ[0, 0]/3.0)/9.0 + (C[1,1] + 2*(τ[0, 0] + τ[1, 1] + τ[2, 2])/3.0 - 4*τ[1, 1]/3.0)/9.0 + 4*(C[2,2] + 2*(τ[0, 0] + τ[1, 1] + τ[2, 2])/3.0 - 4*τ[2, 2]/3.0)/9.0,
                      -(C[0,3] - 2*τ[1, 2]/3.0)/6.0 - (C[0,3] - 2*τ[2, 1]/3.0)/6.0 - (C[1,3] - 2*τ[1, 2]/3.0)/6.0 - (C[1,3] - 2*τ[2, 1]/3.0)/6.0 + (C[2,3] - 2*τ[1, 2]/3.0)/3.0 + (C[2,3] - 2*τ[2, 1]/3.0)/3.0, -(C[0,4] - 2*τ[0, 2]/3.0)/6.0 - (C[0,4] - 2*τ[2, 0]/3.0)/6.0 - (C[1,4] - 2*τ[0, 2]/3.0)/6.0 - (C[1,4] - 2*τ[2, 0]/3.0)/6.0 + (C[2,4] - 2*τ[0, 2]/3.0)/3.0 + (C[2,4] - 2*τ[2, 0]/3.0)/3.0,
                      -(C[0,5] - 2*τ[0, 1]/3.0)/6.0 - (C[0,5] - 2*τ[1, 0]/3.0)/6.0 - (C[1,5] - 2*τ[0, 1]/3.0)/6.0 - (C[1,5] - 2*τ[1, 0]/3.0)/6.0 + (C[2,5] - 2*τ[0, 1]/3.0)/3.0 + (C[2,5] - 2*τ[1, 0]/3.0)/3.0],
                     [(C[3,0] - 2*τ[1, 2]/3.0)/3.0 + (C[3,0] - 2*τ[2, 1]/3.0)/3.0 - (C[3,1] - 2*τ[1, 2]/3.0)/6.0 - (C[3,1] - 2*τ[2, 1]/3.0)/6.0 - (C[3,2] - 2*τ[1, 2]/3.0)/6.0 - (C[3,2] - 2*τ[2, 1]/3.0)/6.0,
                      -(C[3,0] - 2*τ[1, 2]/3.0)/6.0 - (C[3,0] - 2*τ[2, 1]/3.0)/6.0 + (C[3,1] - 2*τ[1, 2]/3.0)/3.0 + (C[3,1] - 2*τ[2, 1]/3.0)/3.0 - (C[3,2] - 2*τ[1, 2]/3.0)/6.0 - (C[3,2] - 2*τ[2, 1]/3.0)/6.0,
                      -(C[3,0] - 2*τ[1, 2]/3.0)/6.0 - (C[3,0] - 2*τ[2, 1]/3.0)/6.0 - (C[3,1] - 2*τ[1, 2]/3.0)/6.0 - (C[3,1] - 2*τ[2, 1]/3.0)/6.0 + (C[3,2] - 2*τ[1, 2]/3.0)/3.0 + (C[3,2] - 2*τ[2, 1]/3.0)/3.0,
                      C[3,3] + (τ[0, 0] + τ[1, 1] + τ[2, 2])/3.0,
                      C[3,4],
                      C[3,5]],
                      [(C[4,0] - 2*τ[0, 2]/3.0)/3.0 + (C[4,0] - 2*τ[2, 0]/3.0)/3.0 - (C[4,1] - 2*τ[0, 2]/3.0)/6.0 - (C[4,1] - 2*τ[2, 0]/3.0)/6.0 - (C[4,2] - 2*τ[0, 2]/3.0)/6.0 - (C[4,2] - 2*τ[2, 0]/3.0)/6.0,
                      -(C[4,0] - 2*τ[0, 2]/3.0)/6.0 - (C[4,0] - 2*τ[2, 0]/3.0)/6.0 + (C[4,1] - 2*τ[0, 2]/3.0)/3.0 + (C[4,1] - 2*τ[2, 0]/3.0)/3.0 - (C[4,2] - 2*τ[0, 2]/3.0)/6.0 - (C[4,2] - 2*τ[2, 0]/3.0)/6.0,
                      -(C[4,0] - 2*τ[0, 2]/3.0)/6.0 - (C[4,0] - 2*τ[2, 0]/3.0)/6.0 - (C[4,1] - 2*τ[0, 2]/3.0)/6.0 - (C[4,1] - 2*τ[2, 0]/3.0)/6.0 + (C[4,2] - 2*τ[0, 2]/3.0)/3.0 + (C[4,2] - 2*τ[2, 0]/3.0)/3.0,
                      C[4,3],
                      C[4,4] + (τ[0, 0] + τ[1, 1] + τ[2, 2])/3.0,
                      C[4,5]],
                     [(C[5,0] - 2*τ[0, 1]/3.0)/3.0 + (C[5,0] - 2*τ[1, 0]/3.0)/3.0 - (C[5,1] - 2*τ[0, 1]/3.0)/6.0 - (C[5,1] - 2*τ[1, 0]/3.0)/6.0 - (C[5,2] - 2*τ[0, 1]/3.0)/6.0 - (C[5,2] - 2*τ[1, 0]/3.0)/6.0,
                     -(C[5,0] - 2*τ[0, 1]/3.0)/6.0 - (C[5,0] - 2*τ[1, 0]/3.0)/6.0 + (C[5,1] - 2*τ[0, 1]/3.0)/3.0 + (C[5,1] - 2*τ[1, 0]/3.0)/3.0 - (C[5,2] - 2*τ[0, 1]/3.0)/6.0 - (C[5,2] - 2*τ[1, 0]/3.0)/6.0,
                     -(C[5,0] - 2*τ[0, 1]/3.0)/6.0 - (C[5,0] - 2*τ[1, 0]/3.0)/6.0 - (C[5,1] - 2*τ[0,1]/3.0)/6.0 - (C[5,1] - 2*τ[1, 0]/3.0)/6.0 + (C[5,2] - 2*τ[0, 1]/3.0)/3.0 + (C[5,2] -2*τ[1,0]/3.0)/3.0,
                      C[5,3],
                      C[5,4],
                      C[5,5] + (τ[0, 0] + τ[1, 1] + τ[2, 2])/3.0]])


def OuterProduct2(h):

      return np.array([[    h[0,0]**2, h[0,0]*h[1,1], h[0,0]*h[2,2], h[0,0]*h[1,2], h[0,0]*h[0,2], h[0,0]*h[0,1]],
                       [h[0,0]*h[1,1],     h[1,1]**2, h[1,1]*h[2,2], h[1,1]*h[1,2], h[0,2]*h[1,1], h[0,1]*h[1,1]],
                       [h[0,0]*h[2,2], h[1,1]*h[2,2],     h[2,2]**2, h[1,2]*h[2,2], h[0,2]*h[2,2], h[0,1]*h[2,2]],
                       [h[0,0]*h[1,2], h[1,1]*h[1,2], h[1,2]*h[2,2],     h[1,2]**2, h[0,2]*h[1,2], h[0,1]*h[1,2]],
                       [h[0,0]*h[0,2], h[0,2]*h[1,1], h[0,2]*h[2,2], h[0,2]*h[1,2],     h[0,2]**2, h[0,1]*h[0,2]],
                       [h[0,0]*h[0,1], h[0,1]*h[1,1], h[0,1]*h[2,2], h[0,1]*h[1,2], h[0,1]*h[0,2],    h[0,1]**2]])


def OuterProduct4(t):
    """ Returns a fourth order tensor for the application of the outer product to a vector three times """
    return np.array([[t[0]**4          , t[0]**2*t[1]**2  , t[0]**2*t[2]**2  , t[0]**2*t[1]*t[2], t[0]**3*t[2]     , t[0]**3*t[1]],
                     [t[0]**2*t[1]**2  , t[1]**4          , t[1]**2*t[2]**2  , t[1]**3*t[2]     , t[0]*t[1]**2*t[2], t[0]*t[1]**3],
                     [t[0]**2*t[2]**2  , t[1]**2*t[2]**2  , t[2]**4          , t[1]*t[2]**3     , t[0]*t[2]**3     , t[0]*t[1]*t[2]**2],
                     [t[0]**2*t[1]*t[2], t[1]**3*t[2]     , t[1]*t[2]**3     , t[1]**2*t[2]**2  , t[0]*t[1]*t[2]**2, t[0]*t[1]**2*t[2]],
                     [t[0]**3*t[2]     , t[0]*t[1]**2*t[2], t[0]*t[2]**3     , t[0]*t[1]*t[2]**2, t[0]**2*t[2]**2  , t[0]**2*t[1]*t[2]],
                     [t[0]**3*t[1]     , t[0]*t[1]**3     , t[0]*t[1]*t[2]**2, t[0]*t[1]**2*t[2], t[0]**2*t[1]*t[2], t[0]**2*t[1]**2]])


def ArrowOperatorVoigt(M, N):
    return np.array([[M[0, 0]*N[0, 0], M[0, 1]*N[0, 1], M[0, 2]*N[0, 2], (M[0, 1]*N[0, 2] + M[0, 2]*N[0, 1])/2.0, (M[0, 0]*N[0, 2] + M[0, 2]*N[0, 0])/2.0, (M[0, 0]*N[0, 1] + M[0, 1]*N[0, 0])/2.0],
                     [M[1, 0]*N[1, 0], M[1, 1]*N[1, 1], M[1, 2]*N[1, 2], (M[1, 1]*N[1, 2] + M[1, 2]*N[1, 1])/2.0, (M[1, 0]*N[1, 2] + M[1, 2]*N[1, 0])/2.0, (M[1, 0]*N[1, 1] + M[1, 1]*N[1, 0])/2.0],
                     [M[2, 0]*N[2, 0], M[2, 1]*N[2, 1], M[2, 2]*N[2, 2], (M[2, 1]*N[2, 2] + M[2, 2]*N[2, 1])/2.0, (M[2, 0]*N[2, 2] + M[2, 2]*N[2, 0])/2.0, (M[2, 0]*N[2, 1] + M[2, 1]*N[2, 0])/2.0],
                     [M[1, 0]*N[2, 0], M[1, 1]*N[2, 1], M[1, 2]*N[2, 2], (M[1, 1]*N[2, 2] + M[1, 2]*N[2, 1])/2.0, (M[1, 0]*N[2, 2] + M[1, 2]*N[2, 0])/2.0, (M[1, 0]*N[2, 1] + M[1, 1]*N[2, 0])/2.0],
                     [M[0, 0]*N[2, 0], M[0, 1]*N[2, 1], M[0, 2]*N[2, 2], (M[0, 1]*N[2, 2] + M[0, 2]*N[2, 1])/2.0, (M[0, 0]*N[2, 2] + M[0, 2]*N[2, 0])/2.0, (M[0, 0]*N[2, 1] + M[0, 1]*N[2, 0])/2.0],
                     [M[0, 0]*N[1, 0], M[0, 1]*N[1, 1], M[0, 2]*N[1, 2], (M[0, 1]*N[1, 2] + M[0, 2]*N[1, 1])/2.0, (M[0, 0]*N[1, 2] + M[0, 2]*N[1, 0])/2.0, (M[0, 0]*N[1, 1] + M[0, 1]*N[1, 0])/2.0]])


def SymmetricGradient(L, nodes_per_element):

    # Create the symmetric gradient operator
    B = np.zeros((6, nodal_dofs * nodes_per_element))
    for a in range(0, nodes_per_element):
        b = a * nodal_dofs

        B[0, b + 0] = L[0, a]
        B[1, b + 1] = L[1, a]
        B[2, b + 2] = L[2, a]
        B[3, b + 2] = L[1, a]
        B[3, b + 1] = L[2, a]
        B[4, b + 0] = L[2, a]
        B[4, b + 2] = L[0, a]
        B[5, b + 0] = L[1, a]
        B[5, b + 1] = L[0, a]
    return B


def DofAllocator(connectivity, nodal_dofs):
    # Create dofmap
    dofmap = []
    for lnodes in connectivity:
        dofmap.append([])
        for lnode in lnodes:
            for nodalDof in range(0, nodal_dofs):
                dofmap[-1].append(lnode * nodal_dofs + nodalDof)

    return dofmap


def UnitSphereQuadrature():
    '''Returns the coordinates and weightings for the 21 point unit sphere quadrature scheme'''

    # Weightings for unit sphere
    w1, w2, w3 = 0.0265214244093*2.0, 0.0199301476312*2.0, 0.0250712367487*2.0
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


def AssembleStiffnessMatrix(elements):

    Kt = lil_matrix((dofs, dofs))

    # Assemble the stiffness matrix
    for e in range(0, elements):

        x_local = np.array([x[i] for i in nodalConnectivity[e]]).T

        lnodes = len(nodalConnectivity[e])

        k_mat = np.zeros((lnodes * nodal_dofs, lnodes * nodal_dofs))
        H = np.zeros((lnodes, lnodes))

        # Build the element stiffness matrices
        for l, (wl, rheal) in enumerate(zip(H8Q1.weights(), H8Q1.femValues())):

            rhea = np.array(rheal)

            # Current configuration mapping
            x_ξ = np.matmul(x_local, rhea)

            # Derivatives of the shape function in global coordinates
            B = SymmetricGradient(np.matmul(rhea, np.linalg.inv(x_ξ)).T, lnodes)
            j = np.linalg.det(x_ξ)

            D = volCvec[e * H8Q1.points() + l] + devCvec[e * H8Q1.points() + l]

            k_mat += np.matmul(B.T, np.matmul(D, B)) * j * wl

        for l, (wl, rheal) in enumerate(zip(H8Q1.weights(), H8Q1.femValues())):

            rhea = np.array(rheal)

            # Current configuration mapping
            x_ξ = np.matmul(x_local, rhea)

            # Derivatives of the shape function in global coordinates
            B = np.matmul(rhea, np.linalg.inv(x_ξ)).T
            j = np.linalg.det(x_ξ)

            σ = σvec[e * H8Q1.points() + l]

            H += np.matmul( B.T, np.matmul(σ, B) ) * j * wl

        # Create the geometric part of the tangent stiffness matrix
        k_geo = np.zeros((lnodes*nodal_dofs, lnodes*nodal_dofs))
        for i in range(0, lnodes):
            for j in range(0, lnodes):
                for k in range(0, nodal_dofs):
                    k_geo[i * nodal_dofs + k, j * nodal_dofs + k] = H[i, j]

        # Assemble into the global matrix
        for a in range(0, 8 * nodal_dofs):
            p = dofmap[e][a]
            for b in range(0, 8 * nodal_dofs):
                q = dofmap[e][b]
                Kt[p, q] += k_mat[a, b] + k_geo[a, b]

    return Kt


def InternalForce():

    fint = np.zeros(dofs)

    for e in range(0, elements):

        x_local = np.array([x[i] for i in nodalConnectivity[e]]).T

        lnodes = len(nodalConnectivity[e])

        fe_int = np.zeros((lnodes, nodal_dofs))

        # Build the element internal force matrix
        for l, (wl, rheal) in enumerate(zip(H8Q1.weights(), H8Q1.femValues())):

            rhea = np.array(rheal)

            # Current configuration mapping
            x_ξ = np.matmul(x_local, rhea)

            # Derivatives of the shape function in global coordinates
            Bt = np.matmul(rhea, np.linalg.inv(x_ξ))

            j = np.linalg.det(x_ξ)

            σ = σvec[e * H8Q1.points() + l]

            fe_int += np.matmul(Bt, σ) * j * wl

        fe_int = np.reshape(fe_int, lnodes*nodal_dofs)

        fint[dofmap[e]] += fe_int

    return fint


def UpdateDeformationMeasures():

    for e in range(0, elements):

        # Gather the nodal coordinates and displacements
        x_local = np.array([x[i] for i in nodalConnectivity[e]]).T
        X_local = np.array([X[i] for i in nodalConnectivity[e]]).T

        # Update measure at each quadrature point
        for l, rheal in enumerate(H8Q1.femValues()):

            rhea = np.array(rheal)

            # Current configuration mapping
            F_ξ = np.matmul(x_local, rhea)
            F0_ξ = np.matmul(X_local, rhea)

            F = np.matmul(F_ξ, np.linalg.inv(F0_ξ))

            Jvec[e * H8Q1.points() + l] = np.linalg.det(F)
            Fvec[e * H8Q1.points() + l] = np.copy(F)

    return None


def Deviatoric(σ):
    """return the deviatoric part of the tensor σ"""
    return σ - 1.0 / 3.0 * np.trace(σ) * np.eye(2)


def MatrixToVoigt(σ):
    return np.array([σ[0, 0], σ[1, 1], σ[2, 2], σ[1, 2], σ[0, 2], σ[0, 1]]).T


def VoigtToMatrix(σ):
    return np.array([[σ[0], σ[5], σ[4]],
                     [σ[5], σ[1], σ[3]],
                     [σ[4], σ[3], σ[2]]])


def VonMisesStress(σ):
    """ Compute the von Mises equivalent stress """
    return np.sqrt(3.0 / 2.0 * np.tensordot(Deviatoric(σ), Deviatoric(σ)))


def UpdateNonAffineModel():

    I = IdentityMatrix()

    p = 1.472  # 3D locking characteristics
    N = 22.01  # Chain locking response
    μ = 0.292  # Ground state stiffness
    q = 0.1086 # Shape of constraint stress
    U = 0.744  # Additional constraint stiffness
    bulk_modulus = 10.0   # Bulk modulus

    for i, (F, J) in enumerate(zip(Fvec, Jvec)):

        λ = 0.0

        h = np.zeros((3, 3))
        k = np.zeros((3, 3))

        G = np.zeros((6, 6))
        H = np.zeros((6, 6))
        K = np.zeros((6, 6))

        # Obtain the unimodular part of the deformation gradient
        Fdev = UnimodularDecomposition(F)

        # Perform the homogenization routine
        for (λr, λs, λt), wl in zip(clist, w):

            ri = np.array((λr, λs, λt))

            # Deformed tangents
            t = np.matmul(Fdev, ri)

            # Affine microstretches
            λbar = np.linalg.norm(t)

            # Non-affine microstretches, derivatives and friends
            λ += λbar**p * wl

            h += λbar**(p - 2.0) * np.outer(t, t) * wl
            H += λbar**(p - 4.0) * OuterProduct4(t) * wl

            # Non-affine tube contribution

            # Deformed normals
            n = np.matmul(np.linalg.inv(Fdev).T, ri)

            # Affine area-stretches
            v = np.linalg.norm(n)

            k += q * v**(q - 2.0) * np.outer(n, n) * wl

            K += v**(q - 4.0) * OuterProduct4(n) * wl

            Gdash = ArrowOperatorVoigt(np.eye(3), np.outer(n, n)) \
                  + ArrowOperatorVoigt(np.outer(n, n), np.eye(3))

            G += v**(q - 2.0) * 0.5 * (Gdash.T + Gdash) * wl

        K *= q * (q - 2.0)
        G *= 2.0 * q

        # Macro-stresses and macro-moduli
        τc = -μ * N * U * k
        Cc =  μ * N * U * (K + G)

        λ = λ**(1.0 / p)
        H *= (p - 2.0)

        # Micro-stresses and micro-moduli
        τf_micro = μ * λ * (3.0 * N - λ**2) / (N - λ**2)
        cf_micro = μ * (λ**4 + 3*N**2) / ( (N - λ**2)**2 )

        # Macro-stresses and macro-moduli
        τf = τf_micro * λ**(1.0 - p) * h

        Cf = ( cf_micro * λ**(2.0 - 2.0*p) - (p - 1.0) * τf_micro * λ**(1.0 - 2.0*p) ) \
               * OuterProduct2(h) + τf_micro * λ**(1.0 - p) * H

        # Superimposed stress response
        τdev = τf + τc
        Cdev = Cf + Cc

        if (J < 0.5 or J > 1.5):
            print('det(F) is out of range:', J)
            exit()

        pressure = J * VolumetricStrainDerivative1(J, bulk_modulus)
        kappa = J**2 * VolumetricStrainDerivative2(J, bulk_modulus)

        # Perform the deviatoric projection using the tensor P
        τ = pressure * np.eye(3) + np.array([[2*τdev[0, 0]/3.0.0 - τdev[1, 1]/3.0.0 - τdev[2, 2]/3.0.0, τdev[0, 1]/2.0 + τdev[1, 0]/2.0, τdev[0, 2]/2.0 + τdev[2, 0]/2.0],
                                             [τdev[0, 1]/2.0 + τdev[1, 0]/2.0, -τdev[0, 0]/3.0.0 + 2*τdev[1, 1]/3.0.0 - τdev[2, 2]/3.0.0, τdev[1, 2]/2.0 + τdev[2, 1]/2.0],
                                             [τdev[0, 2]/2.0 + τdev[2, 0]/2.0, τdev[1, 2]/2.0 + τdev[2, 1]/2.0, -τdev[0, 0]/3.0.0 - τdev[1, 1]/3.0.0 + 2*τdev[2, 2]/3.0.0]])

        devCvec[i] = ComputeDeviatoricTangent(Cdev, τdev)
        volCvec[i] = (kappa + pressure) * OuterProduct2(np.eye(3)) - 2.0 * pressure * I

        σvec[i] = τ / J

    return None


def UpdateInternalVariables():

    UpdateDeformationMeasures()

    UpdateNonAffineModel()


def ApplyDirichletConditions(Kt, r):

    for boundary in boundaryConditions:

        (val, ndof, lnodes) = boundaryConditions[boundary]

        boundaryDofs = [i * nodal_dofs + ndof for i in lnodes]

        Kt[:, boundaryDofs] = Kt[boundaryDofs, :] = 0.0
        r[boundaryDofs] = 0.0

        for i in boundaryDofs:
            Kt[i, i] = 1.0

    return Kt.tocsr(), r


with open('Geometry/cube.mesh') as data_file:
    data = json.load(data_file)

# Allocate fully integrated quadrilateral element
H8Q1 = LagrangeH8(isReducedQuadrature=False)
H8Q1r = LagrangeH8(isReducedQuadrature=True)

clist, w = UnitSphereQuadrature()

# Define the mesh parameters
X, nodalConnectivity = data["Nodes"][0]["Coordinates"], data["Elements"][4]["NodalConnectivity"]

# Take a copy of the coordinates
X = np.array(X, dtype=np.float64)
x = np.copy(X)

# Create the boundary conditions nodal lists
boundaryConditions = {}
boundaryConditions['XSym'] = 0.0, 0, list(set([i for j in data["Elements"][0]['NodalConnectivity'] for i in j]))
boundaryConditions['YSym'] = 0.0, 1, list(set([i for j in data["Elements"][1]['NodalConnectivity'] for i in j]))
boundaryConditions['ZSym'] = 0.0, 2, list(set([i for j in data["Elements"][3]['NodalConnectivity'] for i in j]))
boundaryConditions['UniaxialLoad'] = 6.0, 2, list(set([i for j in data["Elements"][2]['NodalConnectivity'] for i in j]))

dofs = len(X) * nodal_dofs
elements = len(nodalConnectivity)

# Create the lookup table for the global system
dofmap = DofAllocator(nodalConnectivity, nodal_dofs)

numInternalVariables = elements * H8Q1.points()

# Accumulated scalars
Jvec = [0.0 for i in range(0, numInternalVariables)]

# Stress and strain tensors
σvec = [np.zeros((3, 3)) for i in range(0, numInternalVariables)]
Fvec = [np.eye(3) for i in range(0, numInternalVariables)]

devCvec = [np.zeros((6, 6)) for i in range(0, numInternalVariables)]
volCvec = [np.zeros((6, 6)) for i in range(0, numInternalVariables)]

ΔL = 0.1

for loadStep in np.arange(0.0, 1.0 + ΔL, ΔL):

    print('Equilibrium iterations for load step: ', loadStep)

    # Do load step update (apply boundary conditions)
    for boundary in boundaryConditions:

        (val, ndof, lnodes) = boundaryConditions[boundary]

        for lnode in lnodes:
            x[lnode, ndof] = X[lnode, ndof] + loadStep * val

    u = x - X

    UpdateInternalVariables()

    iterations, maxIterations = 0, 15
    while iterations < maxIterations:

        Kt = AssembleStiffnessMatrix(elements)

        fint = InternalForce()

        r = -fint

        # Apply Dirichlet type conditions that results in ensuring
        # nodes on Dirichlet boundaries results in Δu = 0.0 and are enforced
        # through the internal variable update (I hope)
        Kt, r = ApplyDirichletConditions(Kt, r)

        δu = spsolve(Kt, r)

        δu = np.reshape(δu, (int(dofs / nodal_dofs), nodal_dofs))

        # Update the nodal coordinates
        x += δu

        u = x - X

        UpdateInternalVariables()

        print('Nonlinear iteration', iterations,
              '\n\tDisp norm     :', np.linalg.norm(δu),
              '\n\tResidual norm :', np.linalg.norm(r))

        # Check convergence criteria
        if (np.linalg.norm(δu) < 1.0e-10):
            print('-----------------------------------------------------------')
            print('Displacement norm converged!')
            print('Maximum Jacobian determinant', max(Jvec))
            print('-----------------------------------------------------------')
            break

        iterations = iterations + 1

    if iterations >= maxIterations:
        print('\nNewton-Raphson iterations exceeded...exiting')
        exit()

    # Write out the file to vtk
    ugrid = vtk.vtkUnstructuredGrid()
    ugrid.Allocate()

    Points = vtk.vtkPoints()
    for (x1, x2, x3) in X:
        Points.InsertNextPoint(x1, x2, x3)
    ugrid.SetPoints(Points)

    # Insert the local nodal connectivity
    Hexahedron = vtk.vtkHexahedron()
    for localConnectivity in nodalConnectivity:
        for l, c in enumerate(localConnectivity):
            Hexahedron.GetPointIds().SetId(l, c)
        ugrid.InsertNextCell(Hexahedron.GetCellType(), Hexahedron.GetPointIds())

    Displacement = vtk.vtkFloatArray()
    Displacement.SetName("Displacement")
    Displacement.SetNumberOfComponents(3)
    for (u1, u2, u3) in u:
        Displacement.InsertNextTuple3(u1, u2, u3)

    ugrid.GetPointData().AddArray(Displacement)

    print("Writing out results to file")

    filename = "Cube_" + str(int(loadStep / ΔL)) + ".vtu"
    print(filename)

    writer = vtk.vtkXMLUnstructuredGridWriter();
    writer.SetFileName(filename);
    writer.SetInputData(ugrid)
    writer.SetDataModeToAscii()
    writer.Write()

    del writer
    del ugrid

print("Completed!")
