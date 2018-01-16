
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import numpy as np

number_of_bins = 100
time_steps = 13000

chains = []

def update_histogram(num, data):
    plt.cla()
    mu, sigma = np.mean(data[num]), np.std(data[num])

    n, bins, patches = plt.hist(data[num], number_of_bins, normed=True, facecolor='green', alpha=0.75)

    # add a 'best fit' line
    y = mlab.normpdf( bins, mu, sigma)

    l = plt.plot(bins, y, 'r--', linewidth=1)

    plt.text(1200, .0003, r'$\mu=' + str(round(mu)) + ',\ \sigma = ' + str(round(sigma)) + '$')
    plt.xlabel('Segments per chain')
    plt.ylabel('Pr(N)')


for i in range(0, time_steps):
    file = open("chains_" + str(i) + ".txt", "r")
    x = []
    for i in file:
        try:
            x.append(int(i))
        except ValueError:
            continue
    chains.append(x)

# Scale the results by the initial number of chains
initial_number_of_chains = len(chains[0])

print("Initial number of chains ", initial_number_of_chains)

for chain in chains:
    chain = [float(i / float(initial_number_of_chains)) for i in chain]

fig = plt.figure()

n, bins, patches = plt.hist(chains[0], number_of_bins, normed=True, facecolor='green', alpha=0.75)
plt.xlabel('Segments per chain')
plt.ylabel('Pr(N)')


mu, sigma = np.mean(chains[0]), np.std(chains[0])

# add a 'best fit' line
y = mlab.normpdf(bins, mu, sigma)
plt.plot(bins, y, 'r--', linewidth=1)

number_of_frames = len(chains)

animation = animation.FuncAnimation(fig, update_histogram, number_of_frames, interval=1, repeat=False, fargs=(chains,))

plt.show()
