import numpy as np
import matplotlib.pyplot as plt

import nengo
from nengo.objects import Uniform

from base import sorted_spikes

## Encoding

model = nengo.Model("Encoding")
in_ = nengo.Node(output=lambda t: t * 2 - 1)
in_p = nengo.Probe(in_, 'output')

intercepts = np.linspace(-0.9, 0.9, 8)
encoders = np.asarray([1,-1,1,-1,1,-1,1,-1])
intercepts *= encoders
encoders.shape = (8,1)
a = nengo.Ensemble(nengo.LIF(8), 1, intercepts=intercepts,
                   max_rates=Uniform(80, 100), encoders=encoders)
nengo.Connection(in_, a)
a_spikes = nengo.Probe(a, 'spikes')
a_val = nengo.Probe(a, 'decoded_output', filter=0.03)

dummy = nengo.Node()
nengo.DecodedConnection(a, dummy, function=np.cos, filter=0.03)
cos_out = nengo.Probe(dummy, 'output')

## Encoding plot
mm_to_inches = 0.0393701
figsize = (180. * mm_to_inches / 4.0, 180. * mm_to_inches / 2.0)

sim = nengo.Simulator(model)
sim.run(1)

a = sim.get(a)
t = sim.data(model.t_probe)
spikes = [t[sim.data(a_spikes)[:,i] > 0].flatten() for i in xrange(a.n_neurons)]
a.eval_points.sort(axis=0)
J = np.dot(a.eval_points, a.encoders.T)
activities = a.neurons.rates(J)

plt.figure(figsize=figsize)

plt.subplot(3,1,1)
plt.title("Encoding")
plt.plot(a.eval_points, activities, lw=2)
plt.ylabel("Firing rate (Hz)")
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().xaxis.set_ticks_position('bottom')
plt.gca().yaxis.set_ticks_position('left')

plt.subplot(3,1,2)
plt.plot(sim.data(model.t_probe)[1:], sim.data(in_p)[1:], lw=2)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().yaxis.set_ticks_position('left')
plt.xlim(0, 1)
plt.xticks(())
plt.ylabel("Input signal")

plt.subplot(3,1,3)
color_cycle = plt.gca()._get_lines.color_cycle
colors = [next(color_cycle) for _ in xrange(a.n_neurons)]
plt.axis([0, 1, a.n_neurons-0.5, -0.5])
plt.eventplot(spikes, colors=colors)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().xaxis.set_ticks_position('none')
plt.gca().yaxis.set_ticks_position('none')
plt.xlabel("Time (s)")
plt.ylabel("Neuron")

plt.tight_layout(pad=0.1, h_pad=0.1)
plt.savefig("../figures/nef_summary_enc.svg")

## Decoding
model = nengo.Model("Encoding")
in_ = nengo.Node(output=lambda t: t * 2 - 1)
in_p = nengo.Probe(in_, 'output')

intercepts = np.linspace(-0.9, 0.9, 8)
encoders = np.asarray([1,-1,1,-1,1,-1,1,-1])
intercepts *= encoders
encoders.shape = (8,1)
a = nengo.Ensemble(nengo.LIF(8), 1, intercepts=intercepts,
                   max_rates=Uniform(80, 100), encoders=encoders)
nengo.Connection(in_, a)
a_filtered = nengo.Probe(a, "spikes", filter=0.03)
a_val = nengo.Probe(a, 'decoded_output', filter=0.03)

dummy = nengo.Node()
nengo.DecodedConnection(a, dummy, function=np.cos, filter=0.03)
cos_out = nengo.Probe(dummy, 'output')

## Decoding plot

sim = nengo.Simulator(model)
sim.run(1)

plt.figure(figsize=figsize)

plt.subplot(3,1,1)
plt.title("Decoding")
color_cycle = plt.gca()._get_lines.color_cycle
colors = [next(color_cycle) for _ in xrange(a.n_neurons)]
plt.axis([0, 1, a.n_neurons-0.5, -0.5])
plt.eventplot(spikes, colors=colors)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().yaxis.set_ticks_position('none')
plt.xlim(0, 1)
plt.xticks(())

plt.subplot(3,1,2)
scale = 0.15
for i in xrange(a.n_neurons):
    plt.plot(sim.data(model.t_probe), sim.data(a_filtered)[:,i] - i*scale)
plt.xlim(0, 1)
plt.ylabel("Neuron")
plt.yticks(np.arange(scale/1.8, (a.n_neurons+1)  * scale, -scale),
           np.arange(a.n_neurons))
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().yaxis.set_ticks_position('none')
plt.axis([0, 1, (-a.n_neurons+1) * scale, scale])
plt.xticks(())

plt.subplot(3,1,3)
plt.plot(sim.data(model.t_probe), sim.data(in_p))
plt.plot(sim.data(model.t_probe), np.cos(sim.data(in_p)))
plt.plot(sim.data(model.t_probe), sim.data(a_val), label="Input")
plt.plot(sim.data(model.t_probe), sim.data(cos_out), label="cos(Input)")
plt.legend(loc="best")
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().xaxis.set_ticks_position('bottom')
plt.gca().yaxis.set_ticks_position('left')
plt.xlabel("Time (s)")
plt.axis([0, 1, -1.25, 1.25])

plt.tight_layout(pad=0.1, h_pad=0.5)
plt.savefig("../figures/nef_summary_dec.svg")

## Transformation

model = nengo.Model("NEF summary")
in_ = nengo.Node(output=lambda t: np.sin(np.pi*t))
a = nengo.Ensemble(nengo.LIF(50), dimensions=1, label="A")
nengo.Connection(in_, a)
a_spikes = nengo.Probe(a, "spikes")
a_val = nengo.Probe(a, 'decoded_output', filter=0.01)

b = nengo.Ensemble(nengo.LIF(50), dimensions=1, label="B")
nengo.DecodedConnection(a, b, function=np.negative)
b_spikes = nengo.Probe(b, "spikes")
b_val = nengo.Probe(b, 'decoded_output', filter=0.01)

c = nengo.Ensemble(nengo.LIF(50), dimensions=1, label="C")
nengo.DecodedConnection(b, c, function=np.square)
c_spikes = nengo.Probe(c, "spikes")
c_val = nengo.Probe(c, 'decoded_output', filter=0.01)

## Transformation plots

sim = nengo.Simulator(model)
sim.run(2)
t = sim.data(model.t_probe)
grey = [(0, 0, 0)]

plt.figure(figsize=figsize)

plt.subplot(3,1,1)
plt.title("Transformation")
sim_a = next(ens for ens in sim.model.objs if ens.label == "A")
spikes = sorted_spikes(t, sim_a, sim.data(a_spikes))
plt.eventplot(spikes, colors=grey, rasterized=True, linewidths=0.1)
plt.axis([0, 2, a.n_neurons-0.5, -0.5])
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.yticks(())

plt.gca().twinx()
plt.plot(t, sim.data(a_val))
plt.text(1, 1, "A", ha='center', va='center', fontsize='large',
         bbox=dict(ec='none', fc='w', alpha=0.8))
plt.axhline(0, color='k')
plt.xticks(())
plt.axis([0, 2, -1.25, 1.25])
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().yaxis.set_ticks_position('left')

plt.subplot(3,1,2)
sim_b = next(ens for ens in sim.model.objs if ens.label == "B")
spikes = sorted_spikes(t, sim_b, sim.data(b_spikes))
plt.eventplot(spikes, colors=grey, rasterized=True, linewidths=0.1)
plt.axis([0, 2, b.n_neurons-0.5, -0.5])
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.yticks(())

plt.gca().twinx()
plt.plot(t, sim.data(b_val))
plt.text(1, 1, "B=-A", ha='center', va='center', fontsize='large',
         bbox=dict(ec='none', fc='w', alpha=0.8))
plt.axhline(0, color='k')
plt.xticks(())
plt.axis([0, 2, -1.25, 1.25])
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().yaxis.set_ticks_position('left')

plt.subplot(3,1,3)
sim_c = next(ens for ens in sim.model.objs if ens.label == "C")
spikes = sorted_spikes(t, sim_c, sim.data(c_spikes))
plt.eventplot(spikes, colors=grey, rasterized=True, linewidths=0.1)
plt.axis([0, 2, c.n_neurons-0.5, -0.5])
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().xaxis.set_ticks_position('bottom')
plt.yticks(())
plt.xlabel("Time (s)")

plt.gca().twinx()
plt.plot(t, sim.data(c_val), label="C=B$^2$")
plt.text(1, 1, "C=B$^2$", ha='center', va='center', fontsize='large',
         bbox=dict(ec='none', fc='w', alpha=0.8))
plt.axhline(0, color='k')
plt.xlabel("Time (s)")
plt.axis([0, 2, -1.25, 1.25])
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().yaxis.set_ticks_position('left')
plt.gca().xaxis.set_ticks_position('bottom')

plt.tight_layout(pad=0.1, h_pad=0.5)
plt.savefig("../figures/nef_summary_trans.svg")

## Dynamics

model = nengo.Model("NEF summary")
in_ = nengo.Node(output=lambda t: [1,0] if t < 0.1 else [0,0])
oscillator = nengo.Ensemble(nengo.LIF(200), dimensions=2)
nengo.Connection(in_, oscillator)
nengo.DecodedConnection(
    oscillator, oscillator, transform=[[1,1],[-1,1]], filter=0.1)
oscillator_val = nengo.Probe(oscillator, 'decoded_output', filter=0.02)

## Dynamics plotting

sim = nengo.Simulator(model)
sim.run(3)

plt.figure(figsize=figsize)

plt.subplot(2,1,1)
plt.title("Dynamics")
plt.plot(sim.data(model.t_probe), sim.data(oscillator_val))
plt.xlabel('Time (s)')
plt.ylim(-1.25, 1.25)
plt.gca().spines['top'].set_visible(False)
plt.gca().xaxis.set_ticks_position('bottom')
plt.gca().spines['right'].set_visible(False)
plt.gca().yaxis.set_ticks_position('left')

plt.subplot(2,1,2)
plt.plot(sim.data(oscillator_val)[:,0], sim.data(oscillator_val)[:,1])
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
plt.savefig("../figures/nef_summary_dyn.svg")
