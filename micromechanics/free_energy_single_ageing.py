
import numpy as np

import math
import random as rd

import matplotlib.pyplot as plt

# This script will investigate the discrete probability functions associated
# with cross-link formation and the conformation probability

conformation_probability = 1.0

def boltzmann_constant():
    return 1.38064852e-23

def temperature():
    return 298

def number_of_chains():
    return 1e10

segments_per_chain = 100

# join_p = np.floor(segments_per_chain * np.random.rand())
# join_q = np.floor(segments_per_chain * np.random.rand())
#
# print("Cross-link sites:", join_p, join_q)
#
# segments_per_chain_post = segments_per_chain - (join_q - join_p)

active_fraction_rng = np.linspace(0, segments_per_chain, num=5)

k = boltzmann_constant()
T = temperature()
N = number_of_chains()

for active_fraction in active_fraction_rng:

    # Free energy with reduction due to cross-link
    W = []
    λ1_range = np.linspace(0.1, 3.0, num=50)
    for λ1 in λ1_range:
        # Uniaxial tension with incompressiblity constraint
        λ2 = λ3 = 1.0 / np.sqrt(λ1)

        W.append(1 / 2 * k * T * N * (active_fraction * (λ1**2 + λ2**2 + λ3**2) - 3))

    plt.plot(λ1_range, W, label=str(active_fraction))
    plt.title('Free energy for different active chains')
    plt.xlabel('Stretch (x)')
    plt.ylabel('Free energy')

plt.legend()
plt.show()
