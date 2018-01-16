
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# My crappy libraries
from LagrangeQ4 import LagrangeQ4
from StructuredMesher import *

np.set_printoptions(linewidth=160)

# Implementation of the selective integration approach for
# almost incompressible material and comparison with fully integrated
# quadrilateral elements in the incompressible limit

def SymmetricGradient(L, numberOfNodesPerElement):
    # Create the symmetric gradient operator
    B = np.zeros((3, 2 * numberOfNodesPerElement))
    for a in range(0, numberOfNodesPerElement):
        b = a * 2
        B[0, b] = L[0, a]
        B[1, b+1] = L[1, a]
        B[2, b] = L[1, a]
        B[2, b+1] = L[0, a]
    return B

def DofAllocator(connectivity, ndofs):

    # Create dofmap
    dofmap = []
    for lnodes in connectivity:
        dofmap.append([])
        for lnode in lnodes:
            for ndof in range(0, ndofs):
                dofmap[-1].append(lnode*ndofs + ndof)

    return dofmap

def DofFilter(connectivity, ndofs, filterDof):
    return [dof[filterDof-1::ndofs] for dof in DofAllocator(connectivity, ndofs)]


P = 10000.0 # Pressure for shear
L = 16.0 # Length of the beam
c = 2.0  # Boundary condition constant

I = 2.0 * c**3 / 3.0

ν = 0.499999999
E = 200.0E6

λ = ν * E / ((1.0 + ν) * (1.0 - 2.0*ν))
μ = E / (2.0 * (1.0 + ν))

print("Lamé parameters are: ", λ, ',', μ)

# Lambda part of the constitutive model
Dbarbar = λ * np.array([ [1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0] ])

# Mu part of the constitutive model
Dbar = μ * np.array([ [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0] ])

# Allocate our elements
Q4 = LagrangeQ4(False)
Q4r = LagrangeQ4(False)

# Define the mesh parameters
meshGen = MeshGenerator2d(64, 32, 16.0, 2.0)

nodalCoordinates, nodalConnectivity = meshGen.generate()

# Create the boundary conditions nodal lists
boundaryConditions = {}
boundaryConditions['Dirichlet_u-x'] = list(set([i for j in DofFilter(meshGen.bottomBoundary(), 2, 1) for i in j]))
boundaryConditions['Dirichlet_TopLeft_u-x'] = meshGen.topLeftCorner() * 2
boundaryConditions['Dirichlet_bottom_left_u-xy'] = [0, 1]

boundaryConditions['Traction_leftH1'] = meshGen.leftBoundary()
boundaryConditions['Traction_leftH2'] = meshGen.leftBoundary()
boundaryConditions['Traction_rightH2'] = meshGen.rightBoundary()

dofs = len(nodalCoordinates) * 2
elements = len(nodalConnectivity)

# Create the lookup table for the global system
dofmap = DofAllocator(nodalConnectivity, 2)

# System matrix and right hand side
K = np.zeros((dofs, dofs))
f = np.zeros(dofs)

# Assemble the stiffness matrix
for e in range(0, elements):

    x = np.array([nodalCoordinates[i] for i in nodalConnectivity[e]]).T

    lnodes = len(nodalConnectivity[e])

    kebarbar = np.zeros((lnodes*2, lnodes*2))
    kebar = np.zeros((lnodes*2, lnodes*2))

    # Build the deviatoric part
    for wl, rheal in zip(Q4.weights(), Q4.femValues()):

        rhea = np.array(rheal)

        # Current configuration mapping
        x_ξ = np.matmul(x, rhea)

        # Derivatives of the shape function in global coordinates
        B = SymmetricGradient(np.matmul(rhea, np.linalg.inv(x_ξ)).T, lnodes)
        j = np.linalg.det(x_ξ)

        kebar += np.matmul(B.T, np.matmul(Dbar, B)) * j * wl

    # Build the volumetric part
    for wl, rheal in zip(Q4r.weights(), Q4r.femValues()):

        rhea = np.array(rheal)

        # Current configuration mapping
        x_ξ = np.matmul(x, rhea)

        # Derivatives of the shape function in global coordinates
        B = SymmetricGradient(np.matmul(rhea, np.linalg.inv(x_ξ)).T, lnodes)
        j = np.linalg.det(x_ξ)

        kebarbar += np.matmul(B.T, np.matmul(Dbarbar, B)) * j * wl

    # Assemble into the global matrix
    for a in range(0, 8):
        p = dofmap[e][a]
        for b in range(0, 8):
            q = dofmap[e][b]
            K[p, q] += kebar[a, b] + kebarbar[a, b]

#------------------------------------------------------------------------------#
#                            Boundary contributions                            #
#------------------------------------------------------------------------------#

# Right boundary shear loading
#  _______________________________
# |                              |\
# |                              |\   P/(2*I)*(c^2-x_2^2)
# |______________________________|\/
for lnodes in boundaryConditions['Traction_rightH2']:

    x = np.array([nodalCoordinates[lnode] for lnode in lnodes])

    Δh = np.sqrt( (x[0, 0] - x[1, 0])**2 + (x[0, 1] - x[1, 1])**2 )
    # Contribution to shear component
    h_1 = P / (2.0 * I) * np.array([ [c**2 - x[0, 1]**2],
                                     [c**2 - x[1, 1]**2] ])
    # Approximate the quadratic function linearly
    f_e = Δh / 6.0 * np.matmul(np.array([ [2.0, 1.0],
                                          [1.0, 2.0] ]), h_1)
    # Add these into the system
    for lnode, f_a in zip(lnodes, f_e):
        f[lnode*2+1] += f_a

# Left boundary shear loading
#                         ______________________________
#  PLx_2 / I            /|                              |
#  -P/(2*I)*(c^2-x_2^2) /|                              |
#                       /|______________________________|
for h1lnodes, h2lnodes in zip(boundaryConditions['Traction_leftH1'],
                              boundaryConditions['Traction_leftH2']):

    x = np.array([nodalCoordinates[lnode] for lnode in h1lnodes])

    Δh = np.sqrt( (x[0, 0] - x[1, 0])**2 + (x[0, 1] - x[1, 1])**2 )

    # Contribution to shear component
    h_1 =  P * L / I * np.array([ [x[0, 1]],
                                  [x[1, 1]] ])
    h_2 = -P / (2.0 * I) * np.array([ [c**2 - x[0, 1]**2],
                                      [c**2 - x[1, 1]**2] ])

    f_1e = Δh / 6.0 * np.matmul(np.array([ [2.0, 1.0], [1.0, 2.0] ]), h_1)
    f_2e = Δh / 6.0 * np.matmul(np.array([ [2.0, 1.0], [1.0, 2.0] ]), h_2)

    # Add these into the system
    for lnode, f1_a in zip(h1lnodes, f_1e):
        f[lnode*2] += f1_a
    for lnode, f2_a in zip(h2lnodes, f_2e):
        f[lnode*2+1] += f2_a

# Set Dirichlet condition rows and columns to 0 for both dofs at node 0
K[:, 0] = K[0, :] = K[:, 1] = K[1, :] = 0.0
# Set the diagonals to zero
K[0, 0] = K[1, 1] = 1.0
# Correct for Dirichlet conditions on RHS
f[0] = f[1] = 0.0

# Fix the top left hand side
fixedTLC = boundaryConditions['Dirichlet_TopLeft_u-x']
K[:, fixedTLC] = K[fixedTLC, :] = 0.0
K[fixedTLC, fixedTLC] = 1.0
f[fixedTLC] = 0.0

# Symmetric boundary conditions
#   ______________________________
#  |                              |
#  |                              |
#  |______________________________|
#  |o   |o   |o   |o   |o   |o   |o
bottomDofs = boundaryConditions['Dirichlet_u-x']
K[:, bottomDofs] = K[bottomDofs, :] = 0.0
for i in bottomDofs:
    K[i, i] = 1.0
f[bottomDofs] = 0.0

u = np.linalg.solve(K, f)

nodes_x, nodes_y = meshGen.nodeDimensions()

u1 = u[0::2].reshape(nodes_y, nodes_x)
u2 = u[1::2].reshape(nodes_y, nodes_x)

X, Y = np.meshgrid(np.linspace(0., 16., nodes_x), np.linspace(0., 2., nodes_y))

plt.subplot(3, 1, 1)
plt.pcolormesh(X, Y, u1, cmap='jet')
plt.title('Displacement u1')
plt.colorbar()

plt.subplot(3, 1, 2)
plt.pcolormesh(X, Y, u2, cmap='jet')
plt.title('Displacement u2')
plt.colorbar()

plt.subplot(3, 1, 3)
plt.pcolormesh(X, Y, np.sqrt(u2**2 + u1**2), cmap='jet')
plt.title('Displacement magnitude')
plt.colorbar()

# plt.subplots_adjust(bottom=-0.6)
plt.tight_layout()

plt.savefig('Incomp_FI.pdf', bbox_inches='tight')
