
import numpy as np
import matplotlib.pyplot as plt

# This script implements the entropy elastic material model from
# the Physics of Rubber Elasticity, 1975 by Treloar

molarMass = 1.0

ρ = 1000.0  # Density
R = 8.314   # Gas constant
T = 273.15  # Temperature in Kelvin

# Elasticity constant
E = 3.0 * ρ / molarMass * R * T

# Stretch parameter
λ = np.arange(0.5, 4, 0.01)

# Stress
σ = E / 3.0 * (λ - λ**(-2))

plt.figure(1)
plt.plot(λ, σ)
plt.xlabel('Stretch λ')
plt.ylabel(r'Stress $\sigma$ N/mm2')
plt.grid(True)
plt.show()
