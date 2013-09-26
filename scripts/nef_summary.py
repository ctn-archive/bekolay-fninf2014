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

sim = model.simulator()
sim.run(1)

t = sim.data(model.t)
spikes = [t[sim.data("A.spikes")[:,i] > 0].flatten()
          for i in xrange(model.get("A").n_neurons)]
A = sim.model.get("A")
A.eval_points.sort(axis=0)
J = np.dot(A.eval_points, A.encoders.T)
activities = A.neurons.rates(J)

plt.figure(figsize=(4,9))
plt.subplot(3,1,1)
plt.plot(sim.data(model.t)[1:], sim.data("Input signal")[1:], lw=2)
plt.title("Input signal")
plt.xlim(0, 1)

plt.subplot(3,1,2)
plt.plot(A.eval_points, activities, lw=2)
plt.xlabel("Input signal")
plt.ylabel("Firing rate (Hz)")

plt.subplot(3,1,3)
color_cycle = plt.gca()._get_lines.color_cycle
colors = [next(color_cycle) for _ in xrange(model.get("A").n_neurons)]
plt.axis([0, 1, model.get("A").n_neurons-0.5, -0.5])
plt.eventplot(spikes, colors=colors)

plt.tight_layout()
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

plt.figure(figsize=(4,9))

plt.subplot(3,1,1)
color_cycle = plt.gca()._get_lines.color_cycle
colors = [next(color_cycle) for _ in xrange(model.get("A").n_neurons)]
plt.axis([0, 1, model.get("A").n_neurons-0.5, -0.5])
plt.eventplot(spikes, colors=colors)

plt.subplot(3,1,2)
scale = 0.15
for i in xrange(model.get("A").n_neurons-1, -1, -1):
    plt.plot(sim.data(model.t), sim.data("A.spikes")[:,i] + i*scale)
plt.xlim(0, 1)
plt.ylabel("Neuron")
plt.yticks(np.arange(scale/2, model.get("A").n_neurons * scale, scale),
           np.arange(model.get("A").n_neurons, 0, -1)-1)

plt.subplot(3,1,3)
plt.plot(sim.data(model.t), sim.data("Input signal"), label="Input signal")
plt.plot(sim.data(model.t), sim.data("A"), label="Decoded estimate of input")
plt.plot(sim.data(model.t), np.cos(sim.data("Input signal")),
         label="cos(Input signal)")
plt.plot(sim.data(model.t), sim.data(cos),
         label="Decoded estimate of cos(input)")
plt.legend(loc="best")
plt.xlim(0, 1)

plt.tight_layout()
plt.savefig("../figures/nef_summary_dec.pdf")


## Transformation

model = nengo.Model("NEF summary")
model.make_node("Input signal", output=lambda t: np.sin(np.pi*t))
model.make_ensemble("A", nengo.LIF(40), dimensions=1)
model.connect("Input signal", "A")
model.probe("A.spikes")
model.probe("A", filter=0.01)

model.make_ensemble("B", nengo.LIF(40), dimensions=1)
model.connect("A", "B", function=np.negative)
model.probe("B.spikes")
model.probe("B", filter=0.01)

model.make_ensemble("C", nengo.LIF(40), dimensions=1)
model.connect("B", "C", function=np.square)
model.probe("C.spikes")
model.probe("C", filter=0.01)

## Transformation plots

sim = model.simulator()
sim.run(2)

plt.figure(figsize=(10,6.5))
plt.subplot(3,2,1)
plt.plot(sim.data(model.t), sim.data("A"))
plt.axhline(0, color='k')
plt.title("A")
plt.xticks(())
plt.xlim(0, 2)

plt.subplot(3,2,3)
plt.plot(sim.data(model.t), sim.data("B"))
plt.axhline(0, color='k')
plt.title("B = -A")
plt.xticks(())
plt.xlim(0, 2)

plt.subplot(3,2,5)
plt.plot(sim.data(model.t), sim.data("C"))
plt.axhline(0, color='k')
plt.title("C = B$^2$")
plt.xlabel("Time (s)")
plt.xlim(0, 2)

t = sim.data(model.t)
spikes = [t[sim.data("A.spikes")[:,i] > 0].flatten() for i in xrange(model.get("A").n_neurons)]
ax = plt.subplot(3,2,2)
color_cycle = ax._get_lines.color_cycle
colors = [next(color_cycle) for _ in xrange(model.get("A").n_neurons)]
plt.axis([0, 2, model.get("A").n_neurons-0.5, -0.5])
plt.eventplot(spikes, colors=colors, rasterized=True)
plt.title("A")
plt.xticks(())
plt.ylabel("Neuron")

spikes = [t[sim.data("B.spikes")[:,i] > 0].flatten() for i in xrange(model.get("B").n_neurons)]
ax = plt.subplot(3,2,4)
color_cycle = ax._get_lines.color_cycle
colors = [next(color_cycle) for _ in xrange(model.get("B").n_neurons)]
plt.axis([0, 2, model.get("B").n_neurons-0.5, -0.5])
plt.eventplot(spikes, colors=colors, rasterized=True)
plt.title("B")
plt.xticks(())
plt.ylabel("Neuron")

spikes = [t[sim.data("C.spikes")[:,i] > 0].flatten() for i in xrange(model.get("C").n_neurons)]
for ix in xrange(len(spikes)):
    if spikes[ix].shape == (0,):
        spikes[ix] = np.array([-1])
ax = plt.subplot(3,2,6)
color_cycle = ax._get_lines.color_cycle
colors = [next(color_cycle) for _ in xrange(model.get("C").n_neurons)]
plt.axis([0, 2, model.get("C").n_neurons-0.5, -0.5])
plt.eventplot(spikes, colors=colors, rasterized=True)
plt.title("C")
plt.xlabel("Time (s)")
plt.ylabel("Neuron")

plt.tight_layout()
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

plt.figure(figsize=(5,5))
plt.subplot(2,1,1)
plt.plot(sim.data(model.t), sim.data('Oscillator'))
plt.xlabel('Time (s)')

plt.subplot(2,1,2)
plt.plot(sim.data('Oscillator')[:,0], sim.data('Oscillator')[:,1])
plt.xlabel('$x_0$')
plt.ylabel('$x_1$')

plt.tight_layout()
plt.savefig("../figures/nef_summary_dyn.pdf")
