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
state = nengo.Ensemble(nengo.LIF(2000), 3, radius=60, label="State")

def feedback(x):
    dx0 = -sigma * x[0] + sigma * x[1]
    dx1 = -x[0] * x[2] - x[1]
    dx2 = x[0] * x[1] - beta * (x[2] + rho) - rho

    return [dx0 * tau + x[0],
            dx1 * tau + x[1],
            dx2 * tau + x[2]]

nengo.DecodedConnection(state, state, function=feedback, filter=tau)
st_val = nengo.Probe(state, 'decoded_output', filter=tau)
st_spikes = nengo.Probe(state, 'spikes')  # Very expensive!!

sim = nengo.Simulator(model)
sim.run(6)

sim_state = next(ens for ens in sim.model.objs if ens.label == "State")
state_spikes = sorted_spikes(sim.data(model.t_probe), sim_state,
                             sim.data(st_spikes), iterations=250, every=80)

mm_to_inches = 0.0393701
figsize = (100. * mm_to_inches, 80. * mm_to_inches)
plt.figure(figsize=figsize)
ax = plt.subplot2grid((2,3), (0,0), colspan=2, rowspan=2, projection='3d')
ax.plot(sim.data(st_val)[:,0], sim.data(st_val)[:,1], sim.data(st_val)[:,2],
        color='k')
ax.dist = 9

ax = plt.subplot2grid((2,3), (0,2))
ax.plot(sim.data(model.t_probe), sim.data(st_val))
ax.set_xlim(0, 6)
ax.set_ylabel("Amplitude")
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.set_xticks(())
ax.yaxis.tick_right()
ax.yaxis.set_label_position("right")

ax = plt.subplot2grid((2,3), (1,2))
ax.eventplot(state_spikes, colors=[(0,0,0)], rasterized=True, linewidths=0.1)
ax.axis([0, 6, len(state_spikes)-0.5, -0.5])
ax.set_xlabel("Time (s)")
ax.set_ylabel("Neuron")
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.tick_right()
ax.yaxis.set_label_position("right")

plt.tight_layout(pad=0.1, h_pad=0.12)
plt.subplots_adjust(left=0)
plt.savefig("../figures/lorenz_res.svg")
