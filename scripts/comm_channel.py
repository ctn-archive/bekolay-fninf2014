import numpy as np
import matplotlib.pyplot as plt
import nengo
from nengo.helpers import sorted_neurons, white_noise

model = nengo.Model("Communication Channel")

model.make_node("Input", output=white_noise(1, 5, seed=60))

model.make_ensemble("A", nengo.LIF(30), 1)
model.make_ensemble("B", nengo.LIF(30), 1)

model.connect("Input", "A")
model.connect("A", "B")

model.probe("Input")
model.probe("A", filter=0.01)
model.probe("A.spikes")
model.probe("B", filter=0.01)
model.probe("B.spikes")

sim = model.simulator()
sim.run(1)

t = sim.data(model.t)
def sorted_spikes(pop):
    indices = sorted_neurons(sim.model.get(pop))
    spikes = [t[sim.data(pop + ".spikes")[:,i] > 0].flatten() for i in indices]
    for ix in xrange(len(spikes)):
        if spikes[ix].shape == (0,):
            spikes[ix] = np.array([-1])
    return spikes

a_spikes = sorted_spikes("A")
b_spikes = sorted_spikes("B")

def adjust(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.tick_right()
    ax.set_yticks((-1, 0, 1))
    ax.axis([0, 1, -1.25, 1.25])

plt.figure(figsize=(4,5))
ax = plt.subplot(3,1,1)
ax.plot(t, sim.data("Input"), color='k')
ax.text(0.5, 1.0, "Input", ha='center', va='center', fontsize=20)
adjust(ax)
ax.spines['bottom'].set_visible(False)
ax.set_xticks(())

ax = plt.subplot(3,1,2)
ax.eventplot(a_spikes, colors=[(0.75,0.75,0.75)], rasterized=True)
ax.spines['bottom'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.set_yticks(())
ax.axis([0, 1, 0, len(a_spikes)])

ax = ax.twinx()
ax.plot(t, sim.data("A"), color='k')
ax.text(0.5, 1.0, "A", ha='center', va='center', fontsize=20,
        bbox=dict(ec='none', fc='w', alpha=0.8))
ax.spines['bottom'].set_visible(False)
adjust(ax)
ax.set_xticks(())

ax = plt.subplot(3,1,3)
ax.eventplot(b_spikes, colors=[(0.75,0.75,0.75)], rasterized=True)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.xaxis.set_ticks_position('bottom')
ax.set_yticks(())
ax.axis([0, 1, 0, len(b_spikes)])
ax.set_xlabel("Time (s)")

ax = ax.twinx()
ax.plot(t, sim.data("B"), color='k')
ax.text(0.5, 1.0, "B", ha='center', va='center', fontsize=20,
        bbox=dict(ec='none', fc='w', alpha=0.8))
adjust(ax)

plt.tight_layout()
plt.savefig("../figures/comm_channel_res.pdf")
