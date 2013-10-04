"""
Simulating Lorenz attractor for 10 seconds with 1ms timestep
"""

from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

data = OrderedDict()
data['Brian'] = OrderedDict()
data['NEST'] = OrderedDict()
data['NEURON'] = OrderedDict()
data['JavaNengo'] = OrderedDict()
data['PyNengo\nreference'] = OrderedDict()
data['PyNengo\nOpenCL'] = OrderedDict()
colors = ['#238B45', '#FF7F00', '#984EA3', 'k', '#E41A1C', '#08519C']

data['PyNengo\nOpenCL'][100] = [
    0.823611974716,0.827681064606,0.824057102203,0.826804876328,0.825788021088]
data['PyNengo\nOpenCL'][500] = [
    1.03150510788,1.03527617455,1.03847002983,1.04004907608,1.08543992043]
data['PyNengo\nOpenCL'][1000] = [
    1.27273511887,1.27891302109,1.26637387276,1.27474498749,1.26921701431]
data['PyNengo\nOpenCL'][2000] = [
    1.73656892776,1.73761916161,1.72962117195,1.72728586197,1.73043680191]

data['PyNengo\nreference'][100] = [
    1.09255409241,1.09260296822,1.09535098076,1.04782700539,1.09734082222]
data['PyNengo\nreference'][500] = [
    1.37857794762,1.36459708214,1.35432505608,1.36031198502,1.35320615768]
data['PyNengo\nreference'][1000] = [
    1.62529683113,1.61883807182,1.64285898209,1.61444401741,1.63075304031]
data['PyNengo\nreference'][2000] = [
    2.17722296715,2.18516111374,2.20153999329,2.20053505898,2.19265985489]

data['JavaNengo'][100] = [
    3.18400001526,3.3599998951,3.22500014305,3.28499984741,3.30400013924]
data['JavaNengo'][500] = [
    5.78500008583,5.55000019073,5.67199993134,5.05799984932,5.44799995422]
data['JavaNengo'][1000] = [
    8.76200008392,8.84599995613,8.36500000954,7.40699982643,8.35899996758]
data['JavaNengo'][2000] = [
    14.5569999218,14.4720001221,15.6730000973,16.0230000019,14.9340000153]

data['Brian'][100] = [
    39.7258241177,40.3354530334,44.0468769073,43.5285320282,38.9826159477]
data['Brian'][500] = [
    245.399748087,226.683187008,250.801561117,239.863961935,219.887932062]
data['Brian'][1000] = [
    551.249857903,551.689132929,582.092389822,569.137719154,568.181658983]
data['Brian'][2000] = [
    1448.42626905,1461.23177814,1475.119735,1485.89479995,1462.42504406]

data['NEURON'][100] = [
    0.393055915833,0.394006967545,0.392350912094,0.398588895798,0.395780801773]
data['NEURON'][500] = [
    2.99810099602,2.97593998909,2.95858383179,2.93473315239,2.96808600426]
data['NEURON'][1000] = [
    9.42938303947,9.283867836,9.34721088409,9.22296404839,9.32386898994]
data['NEURON'][2000] = [
    20.4385299683,20.4057679176,20.3683900833,20.3296558857,20.2264928818]

data['NEST'][100] = [
    23.4582211971,22.8383898735,23.7844650745,24.908190012,24.6342151165]
data['NEST'][500] = [
    281.606958151,283.047708988,285.237877131,265.189471006,263.031030893]
data['NEST'][1000] = [
    220.25969696,219.951852798,227.322075844,245.524618864,223.129948139]
data['NEST'][2000] = [
    561.650036097,559.695315838,560.368263006,661.86067009,660.920286894]

cvs = []

plt.figure(figsize=(4,5))
ax = plt.subplot(1,1,1)

xval = data.values()[0].keys()
xlabels = ["%d\nneurons" % n for n in xval]

for ix, name in enumerate(data.keys()):
    runtimes = []
    for n, times in data[name].items():
        cvs.append(np.std(times) / np.mean(times))
        runtimes.append(np.mean(times))
    plt.plot(xval, runtimes, '.-', ms=15, color=colors[ix])
    if name == "Brian":
        plt.text(xval[2] + 60, runtimes[2], name, color=colors[ix],
                 ha='left', va='center', fontsize=14)
    elif name == "NEST":
        plt.text(xval[-1] + 60, runtimes[-1], name, color=colors[ix],
                 ha='left', va='center', fontsize=14)

plt.xticks(xval, xlabels)
plt.xlim(50, 2100)
plt.ylim(0, 650)
plt.title('Lorenz attractor benchmarks')
plt.ylabel('Simulation time (s)')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.xaxis.set_ticks_position('bottom')
ax.xaxis.set_tick_params(direction='out')
ax.yaxis.set_ticks_position('left')

inset = zoomed_inset_axes(ax, 17.5,
                          bbox_to_anchor=(0.95, 0.73),
                          bbox_transform=plt.gcf().transFigure)

for ix, name in enumerate(data.keys()):
    runtimes = []
    for n, times in data[name].items():
        cvs.append(np.std(times) / np.mean(times))
        runtimes.append(np.mean(times))
    plt.plot(xval, runtimes, '.-', ms=15, color=colors[ix])
    yoffset = 0
    if name == "PyNengo\nreference":
        yoffset = 1.5
    elif name == "PyNengo\nOpenCL":
        yoffset = -1.5
    if 'Nengo' in name or name == 'NEURON':
        plt.text(xval[-1] + 4, runtimes[-1] + yoffset, name, color=colors[ix],
                 ha='left', va='center', fontsize=14)
inset.set_xlim(1995, 2042)
inset.set_ylim(-2, 21.5)
inset.xaxis.set_ticks_position('bottom')
inset.xaxis.set_tick_params(direction='out')
inset.yaxis.set_ticks_position('right')
inset.yaxis.set_label_position('right')
inset.set_xticks(())

mark_inset(ax, inset, loc1=3, loc2=4, fc="none", ec="0.5")

plt.subplots_adjust(left=0.15, bottom=0.1, top=0.94, right=0.85)
plt.savefig("../figures/bench_lorenz.pdf")

print "Max CV:", max(cvs)
print "All CVs:", sorted(cvs)
