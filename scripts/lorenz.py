import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import nengo

from base import sorted_spikes

tau = 0.1
sigma = 10
beta = 8.0/3
rho = 28

model = nengo.Model('Lorenz attractor')
model.make_ensemble('State', nengo.LIF(2000), 3, radius=60)

def feedback(x):
    dx0 = -sigma * x[0] + sigma * x[1]
    dx1 = -x[0] * x[2] - x[1]
    dx2 = x[0] * x[1] - beta * (x[2] + rho) - rho

    return [dx0 * tau + x[0],
            dx1 * tau + x[1],
            dx2 * tau + x[2]]

model.connect('State', 'State',
              function=feedback, filter=tau)
model.probe('State', filter=tau)
model.probe('State.spikes')  # Very expensive!!

sim = model.simulator()
sim.run(6)

state_spikes = sorted_spikes(sim, "State", iterations=250, every=80)
#state_spikes = [np.linspace(0, 6, 6 * 100)]

plt.figure(figsize=(7,4.55))
ax = plt.subplot2grid((2,3), (0,0), colspan=2, rowspan=2, projection='3d')
ax.plot(sim.data('State')[:,0], sim.data('State')[:,1], sim.data('State')[:,2],
        color='k')
# ax.set_xticklabels(())
# ax.set_yticklabels(())
# ax.set_zticklabels(())
ax.dist = 8.5

ax = plt.subplot2grid((2,3), (0,2))
ax.plot(sim.data(model.t), sim.data('State'))
ax.set_ylabel("Amplitude")
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.set_xticks(())
ax.yaxis.tick_right()
ax.yaxis.set_label_position("right")

ax = plt.subplot2grid((2,3), (1,2))
ax.eventplot(state_spikes, colors=[(0.1,0.1,0.1)], linewidths=0.05,
             rasterized=True)
ax.axis([0, 6, len(state_spikes)-0.5, -0.5])
ax.set_xlabel("Time (s)")
ax.set_ylabel("Neuron")
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.tick_right()
ax.yaxis.set_label_position("right")

plt.tight_layout(h_pad=0.12)
plt.subplots_adjust(left=0)
plt.savefig("../figures/lorenz_res.pdf")
