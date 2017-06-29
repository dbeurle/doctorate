
import numpy as np
import matplotlib.pyplot as plt

# Physical parameters in base SI units
temperature = 298.0
boltzmann_const = 1.38064852e-23

number_of_chains = initial_number_of_chains = 1000

number_of_time_steps = 1000
time_step_size = 1.0

mean_decay_rate, stdev_decay_rate = 0.0001, 0.000001

mean_segments_per_chain, stdev_segments_per_chain = 1000, 100

reaction_mean_per_time_step = 0.001 * time_step_size;

affine_forces = []
nonaffine_forces = []

for t in range(0, number_of_time_steps):

    time = t * time_step_size

    mean_segments_per_chain -= mean_decay_rate * time;
    stdev_segments_per_chain -= stdev_decay_rate * time;

    stretch = 3.0

    # Compute a random stretch for each of the chains in the network
    affine_force = nonaffine_force = 0.0

    for i in range(0, number_of_chains):

        # Probably not correct (pun intended)
        segments_per_chain = int(np.random.normal(loc = mean_segments_per_chain,
                                                  scale = stdev_segments_per_chain))

        relative_stretch = stretch * np.sqrt(segments_per_chain)

        nonaffine_force += (3 * segments_per_chain - stretch**2) / (segments_per_chain - stretch**2)

    # Put these forces into a list for visualisation
    affine_forces.append(3.0 * boltzmann_const * temperature * stretch * number_of_chains)

    nonaffine_forces.append(nonaffine_force * boltzmann_const * temperature * stretch)

print('Mean number of segments per chain', mean_segments_per_chain)
print('Standard deviation of segments per chain', stdev_segments_per_chain)

plt.plot(affine_forces)
plt.title('Mechanical model response with static n')
plt.xlabel('Time steps')
plt.ylabel('Force (N)')

plt.plot(nonaffine_forces)
plt.xlabel('Time steps')
plt.ylabel('Nonaffine force (N)')
plt.legend(['Affine', 'Nonaffine'])
plt.savefig('BasicEvolution.pdf')
