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
model.make_node("Input A", a_in)
model.make_ensemble("A", nengo.LIF(512), 4)
model.connect("Input A", "A")

model.make_node("Input B", b_in)
model.make_ensemble("B", nengo.LIF(512), 4)
model.connect("Input B", "B")

cconv = model.add(CircularConvolution(
    "Circular Convolution", neurons=nengo.LIF(1032), dimensions=4))
model.connect("A", cconv.A)
model.connect("B", cconv.B)

model.make_ensemble("Result", nengo.LIF(512), 4)
model.connect(cconv, "Result")

model.probe("A", filter=0.02)
model.probe("B", filter=0.02)
model.probe("Result", filter=0.02)

sim = model.simulator()
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
    plt.plot(sim.data(model.t), sim.data(ens))
    for color, l in zip(colors, actual):
        plt.axhline(l, color=color)
    plt.xlim(0, 0.5)

plt.figure(figsize=(5,5))
plt.subplot(3, 1, 1)
plot(a_in, "A")
plt.gca().spines['bottom'].set_visible(False)
plt.xticks(())
plt.ylabel("A")

plt.subplot(3, 1, 2)
plot(b_in, "B")
plt.gca().spines['bottom'].set_visible(False)
plt.xticks(())
plt.ylabel("B")

plt.subplot(3, 1, 3)
plot(actual_out, "Result")
plt.xlabel("Time (s)")
plt.ylabel("Result")

plt.tight_layout(h_pad=0.12)
plt.savefig("../figures/cconv_res.pdf")
