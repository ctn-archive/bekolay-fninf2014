import numpy as np

from nengo.helpers import sorted_neurons

def sorted_spikes(t, ens, spike_data, iterations=100, every=None):
    indices = sorted_neurons(ens, iterations=iterations)
    if every is not None:
        indices = indices[::every]
    spikes = [t[spike_data[:,i] > 0].flatten() for i in indices]
    for ix in xrange(len(spikes)):
        if spikes[ix].shape == (0,):
            spikes[ix] = np.array([-1])
    return spikes
