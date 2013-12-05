"""
Simulating communication channel for 10 seconds with 1ms timestep
"""

from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

data = OrderedDict()
data['Brian'] = OrderedDict()
data['NEURON'] = OrderedDict()
data['NEST'] = OrderedDict()
data['Nengo 1.4'] = OrderedDict()
data['Nengo\nOpenCL'] = OrderedDict()
data['Nengo\nreference'] = OrderedDict()
colors = ['#238B45', '#984EA3', '#FF7F00', 'k', '#08519C', '#E41A1C']

data['Nengo\nOpenCL'][(100, 1)] = [
    0.794183015823,0.797194957733,0.798620939255,0.796317100525,0.799592971802]
data['Nengo\nOpenCL'][(250, 1)] = [
    0.836793899536,0.837684869766,0.840667963028,0.837545871735,0.837010145187]
data['Nengo\nOpenCL'][(500, 10)] = [
    1.11762309074,1.11406683922,1.11392784119,1.11361694336,1.11517691612]
data['Nengo\nOpenCL'][(1000, 50)] = [
    4.37966895103,4.36161804199,4.36030006409,4.36526703835,4.36328911781]

data['Nengo\nreference'][(100, 1)] = [
    1.79912400246,1.72637414932,1.77085995674,1.72308516502,1.75330209732]
data['Nengo\nreference'][(250, 1)] = [
    1.97227215767,1.98903083801,1.96172499657,2.02151703835,2.0146241188]
data['Nengo\nreference'][(500, 10)] = [
    2.4279820919,2.41257190704,2.44041395187,2.39347505569,2.40128898621]
data['Nengo\nreference'][(1000, 50)] = [
    3.12998580933,3.08235788345,3.18121695518,3.12098288536,3.13230895996]

data['Nengo 1.4'][(100, 1)] = [
    4.13199996948,4.29999995232,3.9509999752,3.97099995613,4.41100001335]
data['Nengo 1.4'][(250, 1)] = [
    5.25999999046,5.44099998474,4.79100012779,4.75,4.85299992561]
data['Nengo 1.4'][(500, 10)] = [
    6.51999998093,7.07800006866,6.37999987602,6.38900017738,6.81500005722]
data['Nengo 1.4'][(1000, 50)] = [
    12.6990001202,14.260999918,12.8940000534,13.2829999924,13.9590001106]

data['Brian'][(100, 1)] = [
    70.6575419903,70.4931750298,69.5717558861,70.6680939198,71.4764060974]
data['Brian'][(250, 1)] = [
    179.561398029,176.377717018,175.385248899,181.00644207,176.64626503]
data['Brian'][(500, 10)] = [
    367.776983976,363.590926886,353.305940151,372.290659904,357.998761177]
data['Brian'][(1000, 50)] = [
    771.436607122,752.722177029,795.356714964,753.433661938,767.162684917]

data['NEURON'][(100, 1)] = [
    1.32816815376,1.28712105751,1.37577104568,1.31121301651,1.33905911446]
data['NEURON'][(250, 1)] = [
    6.10494995117,5.62887692451,6.42468309402,6.0553560257,6.13251495361]
data['NEURON'][(500, 10)] = [
    25.2068440914,24.658962965,24.5855839252,25.9286489487,24.4852700233]
data['NEURON'][(1000, 50)] = [
    84.1481909752,81.0623428822,83.5875000954,85.2272210121,81.6756131649]

data['NEST'][(100, 1)] = [
    1.74608612061,1.53785896301,2.27631497383,1.49289417267,1.46029901505]
data['NEST'][(250, 1)] = [
    4.70604491234,4.53917694092,4.82926082611,4.86097407341,4.57820892334]
data['NEST'][(500, 10)] = [
    15.7866659164,17.0758199692,16.5210909843,17.3852770329,19.0535829067]
data['NEST'][(1000, 50)] = [
    87.3746140003,66.3134651184,95.0474739075,56.9202830791,56.1858429909]

cvs = []

mm_to_inches = 0.0393701
figsize = (180. * mm_to_inches / 3., 180. * mm_to_inches / 2.5)
plt.figure(figsize=figsize)
ax = plt.subplot(1,1,1)

x = data.values()[0].keys()
xval = [n for (n, d) in x]
xlabels = ["%d neurons\n%d dimension" % (n, d) if d == 1 else
           "%d neurons\n%d dimensions" % (n, d) for (n, d) in x]
for ix, name in enumerate(data.keys()):
    runtimes = []
    for (n, d), times in data[name].items():
        cvs.append(np.std(times) / np.mean(times))
        runtimes.append(np.mean(times))
    plt.plot(xval, runtimes, '.-', ms=8, color=colors[ix], label=name)
    if name == "Brian":
        plt.text(xval[-3] + 40, runtimes[-3], name, color=colors[ix],
                 ha='left', va='center', fontsize='large')
    elif name in ("NEURON", "NEST"):
        plt.text(xval[-1] + 40, runtimes[-1], name, color=colors[ix],
                 ha='left', va='center', fontsize='large')

del xval[1]
del xlabels[1]
plt.xticks(xval, xlabels)
plt.xlim(70, 1130)
plt.ylim(0, 200)
plt.ylabel('Simulation time (s)')
plt.text(720, 206.5, 'Communication channel benchmarks',
         ha='center', va='center', fontsize='large')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.xaxis.set_ticks_position('bottom')
ax.xaxis.set_tick_params(direction='out')
ax.yaxis.set_ticks_position('left')

inset = zoomed_inset_axes(ax, 7,
                          bbox_to_anchor=(0.955, 0.935),
                          bbox_transform=plt.gcf().transFigure)

xval = [n for (n, d) in x]
for ix, name in enumerate(data.keys()):
    runtimes = []
    for (n, d), times in data[name].items():
        cvs.append(np.std(times) / np.mean(times))
        runtimes.append(np.mean(times))
    inset.plot(xval, runtimes, '.-', ms=8, color=colors[ix], label=name)
    yoffset = 0
    if name == "Nengo\nOpenCL":
        yoffset = 0.85
    elif name == "Nengo\nreference":
        yoffset = -0.85
    if 'Nengo' in name:
        plt.text(xval[-1] + 7, runtimes[-1] + yoffset, name, color=colors[ix],
                 ha='left', va='center', fontsize='large')


inset.set_xlim(980, 1065)
inset.set_ylim(0, 15)
inset.xaxis.set_ticks_position('bottom')
inset.xaxis.set_tick_params(direction='out')
inset.yaxis.set_ticks_position('right')
inset.yaxis.set_label_position('right')
inset.set_xticks(())

mark_inset(ax, inset, loc1=3, loc2=4, fc="none", ec="0.5")

plt.subplots_adjust(left=0.15, bottom=0.1, top=0.935, right=0.75)
plt.savefig("../figures/bench_cchannel.svg")

print "Max CV:", max(cvs)
print "All CVs:", sorted(cvs)
