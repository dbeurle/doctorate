
import numpy as np
import matplotlib.pyplot as plt

# Material properties
elasticModulus  = 0.01E9    # 0.01 GPa
linearViscosity = 4.0E7

# Time to solve equations over
t = np.linspace(0, 10.0, 100)

fig = plt.figure()

# Maxwell model of viscoelasticity
#        |--------
# -------|   |----------/\/\/\/\/\-------
#        |--------

relaxationTime = linearViscosity / elasticModulus

# Constant stress
stress_0 = 100.0e5
strain = stress_0 * (1.0 / linearViscosity * t + 1.0 / elasticModulus)

# Constant strain
strain_0 = 0.001
stress = elasticModulus * np.exp(-t / relaxationTime)

plt.figure(1)
plt.subplot(2, 1, 1)
plt.plot(t, strain)
plt.title('Maxwell model')
plt.ylabel('Strain [-]')

plt.subplot(2, 1, 2)
plt.plot(t, stress)
plt.xlabel('Time [s]')
plt.ylabel('Stress [Pa]')

# Kelvin model of viscoelasticity
#             |----------
#        |----|    |-----------|
# <------|    |----------      |----->
#        |                     |
#        |----/\/\/\/\/\/------|

# Constant stress
strain = stress_0 / elasticModulus * (1 - np.exp(-t / relaxationTime))

# Constant strain
stress = strain_0 * elasticModulus * np.ones(strain.size)

plt.figure(2)
plt.subplot(2, 1, 1)
plt.plot(t, strain)
plt.title('Kelvin model')
plt.ylabel('Strain [-]')

plt.subplot(2, 1, 2)
plt.plot(t, stress)
plt.xlabel('Time [s]')
plt.ylabel('Stress [Pa]')

plt.show()
