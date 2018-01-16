
# Volumetric strain energy functions from Doll and Schweizerhof

import numpy as np

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Change the font size of the legend
mpl.rcParams['legend.fontsize'] = 16
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16

# The range of values to plot the normalised stress
Jvec = np.arange(0.0, 4.0, 0.01)

U1 = [ 0.5*(J-1.0)**2 for J in Jvec]
U2 = [ np.log(J)**2/2.0 for J in Jvec]
U3 = [ 0.25*( (J-1.0)**2 + np.log(J)**2) for J in Jvec]
U5 = [ J*np.log(J) - J + 1.0 for J in Jvec]

dU1 = [ J-1.0 for J in Jvec]
dU2 = [ np.log(J)/J for J in Jvec]
dU3 = [ 0.5*( J-1.0 + np.log(J)/J) for J in Jvec]
dU5 = [ np.log(J) for J in Jvec]

plt.figure()
plt.plot(Jvec, U1, linewidth=2, label=r'$U_1/K = (J-1)^2/2$')
plt.plot(Jvec, U2, linewidth=2, label=r'$U_2/K = \ln(J)^2/2$')
plt.plot(Jvec, U3, linewidth=2, label=r'$U_3/K = ((J-1)^2 + (\lnJ)^2)/4$')
plt.plot(Jvec, U5, linewidth=2, label=r'$U_4/K = J\ln(J) - J + 1$')
plt.grid(True)

plt.legend(loc='upper center')
plt.xticks(range(0, 5))
plt.xlim([0, 4])
plt.ylim([0, 4])
plt.xlabel('Jacobian determinant, J')
plt.ylabel('Scaled energy U / K')
plt.tight_layout()
plt.savefig('VolumetricStrainEnergy.pdf')

plt.figure()
plt.plot(Jvec, dU1, linewidth=2, label=r'$\partial U_1/K = J-1$')
plt.plot(Jvec, dU2, linewidth=2, label=r'$\partial U_2/K = \ln(J)/J$')
plt.plot(Jvec, dU3, linewidth=2, label=r'$\partial U_3/K = ( J-1 + (\lnJ)/J)/2$')
plt.plot(Jvec, dU5, linewidth=2, label=r'$\partial U_4/K = \ln(J)$')
plt.grid(True)

plt.legend(loc='upper left')
plt.xticks(range(0, 5))
plt.xlim([0, 4])
plt.ylim([-4, 4])
plt.xlabel('Jacobian determinant, J')
plt.ylabel('Derivative scaled energy dU\' / K')
plt.tight_layout()
plt.savefig('VolumetricStrainEnergyDerivative.pdf')
