import numpy as np

from nengo.helpers import sorted_neurons

def sorted_spikes(sim, pop, iterations=100, every=None):
    t = sim.data(sim.model.t)
    indices = sorted_neurons(sim.model.get(pop), iterations=iterations)
    if every is not None:
        indices = indices[::every]
    spikes = [t[sim.data(pop + ".spikes")[:,i] > 0].flatten() for i in indices]
    for ix in xrange(len(spikes)):
        if spikes[ix].shape == (0,):
            spikes[ix] = np.array([-1])
    return spikes
