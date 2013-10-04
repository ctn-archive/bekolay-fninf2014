import numpy as np
import matplotlib.pyplot as plt

import nengo
from nengo.objects import Uniform

from base import sorted_spikes

## Encoding

model = nengo.Model("Encoding")
model.make_node("Input signal", output=lambda t: t * 2 - 1)
model.probe("Input signal")

intercepts = np.linspace(-0.9, 0.9, 8)
encoders = np.asarray([1,-1,1,-1,1,-1,1,-1])
intercepts *= encoders
encoders.shape = (8,1)
model.make_ensemble("A", nengo.LIF(8), 1, intercepts=intercepts,
                    max_rates=Uniform(80, 100), encoders=encoders)
model.connect("Input signal", "A")
model.probe("A.spikes")
model.probe("A", filter=0.03)

model.make_node("Dummy", output=lambda t: 0)
cos = model.connect("A", "Dummy", function=np.cos, filter=0.03)
model.probe(cos)

## Encoding plot
figsize = (3.5, 7)

sim = model.simulator()
sim.run(1)

t = sim.data(model.t)
spikes = [t[sim.data("A.spikes")[:,i] > 0].flatten()
          for i in xrange(model.get("A").n_neurons)]
A = sim.model.get("A")
A.eval_points.sort(axis=0)
J = np.dot(A.eval_points, A.encoders.T)
activities = A.neurons.rates(J)

plt.figure(figsize=figsize)

plt.subplot(3,1,1)
plt.title("Encoding")
plt.plot(A.eval_points, activities, lw=2)
plt.ylabel("Firing rate (Hz)")
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().xaxis.set_ticks_position('bottom')
plt.gca().yaxis.set_ticks_position('left')

plt.subplot(3,1,2)
plt.plot(sim.data(model.t)[1:], sim.data("Input signal")[1:], lw=2)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().yaxis.set_ticks_position('left')
plt.xlim(0, 1)
plt.xticks(())
plt.ylabel("Input signal")

plt.subplot(3,1,3)
color_cycle = plt.gca()._get_lines.color_cycle
colors = [next(color_cycle) for _ in xrange(model.get("A").n_neurons)]
plt.axis([0, 1, model.get("A").n_neurons-0.5, -0.5])
plt.eventplot(spikes, colors=colors)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().xaxis.set_ticks_position('none')
plt.gca().yaxis.set_ticks_position('none')
plt.xlabel("Time (s)")
plt.ylabel("Neuron")

plt.tight_layout(pad=0.1, h_pad=0.1)
plt.savefig("../figures/nef_summary_enc.pdf")

## Decoding
# NB: We don't make a new network so that the plots match

# Hack!
model.get("A").probes['spikes'] = []
del model.get("A").connections_out[0]
del model.probed["A.spikes"]
# End hack
model.probe("A.spikes", filter=0.03)

## Decoding plot

sim = model.simulator()
sim.run(1)

plt.figure(figsize=figsize)

plt.subplot(3,1,1)
plt.title("Decoding")
color_cycle = plt.gca()._get_lines.color_cycle
colors = [next(color_cycle) for _ in xrange(model.get("A").n_neurons)]
plt.axis([0, 1, model.get("A").n_neurons-0.5, -0.5])
plt.eventplot(spikes, colors=colors)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().yaxis.set_ticks_position('none')
plt.xlim(0, 1)
plt.xticks(())

plt.subplot(3,1,2)
scale = 0.15
for i in xrange(model.get("A").n_neurons):
    plt.plot(sim.data(model.t), sim.data("A.spikes")[:,i] - i*scale)
plt.xlim(0, 1)
plt.ylabel("Neuron")
plt.yticks(np.arange(scale/1.8, (-model.get("A").n_neurons+1)  * scale, -scale),
           np.arange(model.get("A").n_neurons))
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().yaxis.set_ticks_position('none')
plt.axis([0, 1, (-model.get("A").n_neurons+1) * scale, scale])
plt.xticks(())

plt.subplot(3,1,3)
plt.plot(sim.data(model.t), sim.data("Input signal"))
plt.plot(sim.data(model.t), np.cos(sim.data("Input signal")))
plt.plot(sim.data(model.t), sim.data("A"), label="Input")
plt.plot(sim.data(model.t), sim.data(cos), label="cos(Input)")
plt.legend(loc="best")
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().xaxis.set_ticks_position('bottom')
plt.gca().yaxis.set_ticks_position('left')
plt.xlabel("Time (s)")
plt.axis([0, 1, -1.25, 1.25])

plt.tight_layout(pad=0.1, h_pad=0.5)
plt.savefig("../figures/nef_summary_dec.pdf")


## Transformation

model = nengo.Model("NEF summary")
model.make_node("Input signal", output=lambda t: np.sin(np.pi*t))
model.make_ensemble("A", nengo.LIF(50), dimensions=1)
model.connect("Input signal", "A")
model.probe("A.spikes")
model.probe("A", filter=0.01)

model.make_ensemble("B", nengo.LIF(50), dimensions=1)
model.connect("A", "B", function=np.negative)
model.probe("B.spikes")
model.probe("B", filter=0.01)

model.make_ensemble("C", nengo.LIF(50), dimensions=1)
model.connect("B", "C", function=np.square)
model.probe("C.spikes")
model.probe("C", filter=0.01)

## Transformation plots

sim = model.simulator()
sim.run(2)
t = sim.data(model.t)
grey = [(0.75, 0.75, 0.75)]

plt.figure(figsize=figsize)

plt.subplot(3,1,1)
plt.title("Transformation")
spikes = sorted_spikes(sim, "A")
plt.eventplot(spikes, colors=grey, rasterized=True, linewidths=0.1)
plt.axis([0, 2, model.get("A").n_neurons-0.5, -0.5])
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.yticks(())

plt.gca().twinx()
plt.plot(sim.data(model.t), sim.data("A"))
plt.text(1, 1, "A", ha='center', va='center', fontsize=16,
         bbox=dict(ec='none', fc='w', alpha=0.8))
plt.axhline(0, color='k')
plt.xticks(())
plt.axis([0, 2, -1.25, 1.25])
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().yaxis.set_ticks_position('left')

plt.subplot(3,1,2)
spikes = sorted_spikes(sim, "B")
plt.eventplot(spikes, colors=grey, rasterized=True, linewidths=0.1)
plt.axis([0, 2, model.get("B").n_neurons-0.5, -0.5])
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.yticks(())

plt.gca().twinx()
plt.plot(sim.data(model.t), sim.data("B"))
plt.text(1, 1, "B=-A", ha='center', va='center', fontsize=16,
         bbox=dict(ec='none', fc='w', alpha=0.8))
plt.axhline(0, color='k')
plt.xticks(())
plt.axis([0, 2, -1.25, 1.25])
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().yaxis.set_ticks_position('left')

plt.subplot(3,1,3)
spikes = sorted_spikes(sim, "C")
plt.eventplot(spikes, colors=grey, rasterized=True, linewidths=0.1)
plt.axis([0, 2, model.get("C").n_neurons-0.5, -0.5])
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().xaxis.set_ticks_position('bottom')
plt.yticks(())
plt.xlabel("Time (s)")

plt.gca().twinx()
plt.plot(sim.data(model.t), sim.data("C"), label="C=B$^2$")
plt.text(1, 1, "C=B$^2$", ha='center', va='center', fontsize=16,
         bbox=dict(ec='none', fc='w', alpha=0.8))
plt.axhline(0, color='k')
plt.xlabel("Time (s)")
plt.axis([0, 2, -1.25, 1.25])
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().yaxis.set_ticks_position('left')
plt.gca().xaxis.set_ticks_position('bottom')

plt.tight_layout(pad=0.1, h_pad=0.5)
plt.savefig("../figures/nef_summary_trans.pdf")

## Dynamics

model = nengo.Model("NEF summary")
model.make_node('Input signal', output=lambda t: [1,0] if t < 0.1 else [0,0])
model.make_ensemble('Oscillator', nengo.LIF(200), dimensions=2)
model.connect('Input signal','Oscillator')
model.connect('Oscillator', 'Oscillator', transform=[[1,1],[-1,1]], filter=0.1)
model.probe('Oscillator', filter=0.02)

## Dynamics plotting

sim = model.simulator()
sim.run(3)

plt.figure(figsize=figsize)

plt.subplot(2,1,1)
plt.title("Dynamics")
plt.plot(sim.data(model.t), sim.data('Oscillator'))
plt.xlabel('Time (s)')
plt.ylim(-1.25, 1.25)
plt.gca().spines['top'].set_visible(False)
plt.gca().xaxis.set_ticks_position('bottom')
plt.gca().spines['right'].set_visible(False)
plt.gca().yaxis.set_ticks_position('left')

plt.subplot(2,1,2)
plt.plot(sim.data('Oscillator')[:,0], sim.data('Oscillator')[:,1])
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.axis([-1.2, 1.2, -1.2, 1.2])
plt.grid()
plt.axvline(color='k')
plt.axhline(color='k')
plt.gca().yaxis.set_ticks_position('none')
plt.gca().xaxis.set_ticks_position('none')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.tight_layout(pad=0.1, h_pad=2)
plt.savefig("../figures/nef_summary_dyn.pdf")
