import numpy as np
import matplotlib.pyplot as plt

import nengo
from nengo.helpers import white_noise

from base import sorted_spikes

model = nengo.Model("Communication Channel")

in_ = nengo.Node(output=white_noise(1, 5, seed=60))

a = nengo.Ensemble(nengo.LIF(30), 1, label="A")
b = nengo.Ensemble(nengo.LIF(30), 1, label="B")

nengo.Connection(in_, a)
nengo.DecodedConnection(a, b)

in_p = nengo.Probe(in_, 'output')
a_val = nengo.Probe(a, 'decoded_output', filter=0.01)
a_spikes = nengo.Probe(a, 'spikes')
b_val = nengo.Probe(b, 'decoded_output', filter=0.01)
b_spikes = nengo.Probe(b, 'spikes')

sim = nengo.Simulator(model)
sim.run(1)

t = sim.data(model.t_probe)
sim_a = next(ens for ens in sim.model.objs if ens.label == "A")
a_spikes = sorted_spikes(t, sim_a, sim.data(a_spikes))
sim_b = next(ens for ens in sim.model.objs if ens.label == "B")
b_spikes = sorted_spikes(t, sim_b, sim.data(b_spikes))

def adjust(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.tick_right()
    ax.set_yticks((-1, 0, 1))
    ax.axis([0, 1, -1.25, 1.25])

mm_to_inches = 0.0393701
figsize = (85. * mm_to_inches / 2., (39.31 + 4) * mm_to_inches)
plt.figure(figsize=figsize)
ax = plt.subplot(3,1,1)
ax.plot(t, sim.data(in_p), color='k')
ax.text(0.5, 1.0, "Input", ha='center', va='center', fontsize='large')
adjust(ax)
ax.spines['bottom'].set_visible(False)
ax.set_xticks(())

ax = plt.subplot(3,1,2)
ax.eventplot(a_spikes, colors=[(0,0,0)], rasterized=True, linewidths=0.1)
ax.spines['bottom'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.set_yticks(())
ax.axis([0, 1, -0.5, len(a_spikes)-0.5])

ax = ax.twinx()
ax.plot(t, sim.data(a_val), color='k')
ax.text(0.5, 1.0, "A", ha='center', va='center', fontsize='large',
        bbox=dict(ec='none', fc='w', alpha=0.8))
ax.spines['bottom'].set_visible(False)
adjust(ax)
ax.set_xticks(())

ax = plt.subplot(3,1,3)
ax.eventplot(b_spikes, colors=[(0,0,0)], rasterized=True, linewidths=0.1)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.xaxis.set_ticks_position('bottom')
ax.set_yticks(())
ax.axis([0, 1, -0.5, len(b_spikes)-0.5])
ax.set_xlabel("Time (s)")

ax = ax.twinx()
ax.plot(t, sim.data(b_val), color='k')
ax.text(0.5, 1.0, "B", ha='center', va='center', fontsize='large',
        bbox=dict(ec='none', fc='w', alpha=0.8))
adjust(ax)

plt.tight_layout(pad=0.1, h_pad=0.15)
plt.savefig("../figures/comm_channel_res.svg")
