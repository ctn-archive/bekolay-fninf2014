import numpy as np
import matplotlib.pyplot as plt

import nengo
from nengo.networks import CircularConvolution

def circconv(a, b):
    A = np.fft.fft(a)
    B = np.fft.fft(b)
    return np.fft.ifft(A * B).real

a_in = np.asarray([-0.21, 0.5, 0.12, 0.06])
b_in = np.asarray([-0.18, 0.28, 0.18, -0.52])

model = nengo.Model("Circular convolution")
in_a = nengo.Node(a_in)
a = nengo.Ensemble(nengo.LIF(512), 4, label="A")
nengo.Connection(in_a, a)
in_b = nengo.Node(b_in)
b = nengo.Ensemble(nengo.LIF(512), 4, label="B")
nengo.Connection(in_b, b)

cconv = CircularConvolution(neurons=nengo.LIF(1032), dimensions=4)
nengo.DecodedConnection(a, cconv.A)
nengo.DecodedConnection(b, cconv.B)

result = nengo.Ensemble(nengo.LIF(512), 4, label="Result")
nengo.Connection(cconv.output, result)

a_val = nengo.Probe(a, 'decoded_output', filter=0.02)
b_val = nengo.Probe(b, 'decoded_output', filter=0.02)
res_val = nengo.Probe(result, 'decoded_output', filter=0.02)

sim = nengo.Simulator(model)
sim.run(0.5)

actual_out = circconv(a_in, b_in)

def plot(actual, ens):
    color_cycle = plt.gca()._get_lines.color_cycle
    colors = [next(color_cycle) for _ in xrange(4)]
    plt.gca().set_color_cycle(colors)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().xaxis.set_ticks_position('bottom')
    plt.gca().yaxis.tick_right()
    plt.gca().yaxis.set_label_position("right")
    plt.plot(sim.data(model.t_probe), sim.data(ens))
    for color, l in zip(colors, actual):
        plt.axhline(l, color=color)
    plt.xlim(0, 0.2)

mm_to_inches = 0.0393701
figsize = (40 * mm_to_inches, 49.16 * mm_to_inches)
plt.figure(figsize=figsize)
plt.subplot(3, 1, 1)
plot(a_in, a_val)
plt.gca().spines['bottom'].set_visible(False)
plt.xticks(())
plt.ylabel("A")

plt.subplot(3, 1, 2)
plot(b_in, b_val)
plt.gca().spines['bottom'].set_visible(False)
plt.xticks(())
plt.ylabel("B")

plt.subplot(3, 1, 3)
plot(actual_out, res_val)
plt.xlabel("Time (s)")
plt.ylabel("Result")

plt.tight_layout(pad=0.1, h_pad=0.12)
plt.savefig("../figures/cconv_res.svg")
