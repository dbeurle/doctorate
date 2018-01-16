
import matplotlib.pyplot as plt
import numpy as np

def inverseLangevin(x):
    return 3.0 * x * ( (35.0 - 12.0*x**2) / (35.0 - 33.0*x**2) )

# Plot the free energy function of the model by Kuhn and Grün using
# the Langevin statistical model

N = 1;                      # Chains
k_b = 1.38064852*10**(-23)  # Boltzmann constant
T = 298;                    # Temperature

# Stretches
λ_r = np.linspace(1.0, 6.0, num=100) / np.sqrt(N);

print (λ_r)

# Free energy function
ψ = N*k_b*T*(λ_r * inverseLangevin(λ_r) + np.log( inverseLangevin(λ_r) / np.sinh(inverseLangevin(λ_r))) );

plt.plot(λ_r, ψ);
plt.show();

