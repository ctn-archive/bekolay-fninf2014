from collections import OrderedDict

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

neurons = {
    5:4224, 10:8448, 20:17408, 50:44288, 100:89088, 200:178688, 500:447488
}

data = OrderedDict()

data['JavaNengo'] = OrderedDict()
data['PyNengo\nreference'] = OrderedDict()
data['Intel Core\ni7-3770'] = OrderedDict()
data['Intel Xeon\nE5540'] = OrderedDict()
data['Intel Xeon\nE5-2620'] = OrderedDict()
data['NVidia\nQuadro\nK4000'] = OrderedDict()
data['NVidia Tesla\nC2050'] = OrderedDict()
data['ATI Radeon\nHD7970'] = OrderedDict()
colors = ['k', '#E41A1C', '#023858', '#045A8D', '#0570B0',
          '#3690C0', '#74A9CF', '#A6BDDB']

data['JavaNengo'][100] = 9
data['JavaNengo'][200] = 18
data['JavaNengo'][500] = 45
data['PyNengo\nreference'][5] = 2.62999796867
data['PyNengo\nreference'][10] = 5.21562504768
data['PyNengo\nreference'][20] = 10.5295498371
data['PyNengo\nreference'][50] = 27.0069601536
data['Intel Core\ni7-3770'][5] = 0.168757915497
data['Intel Core\ni7-3770'][10] = 0.256566047668
data['Intel Core\ni7-3770'][20] = 0.482143878937
data['Intel Core\ni7-3770'][50] = 1.10272312164
data['Intel Core\ni7-3770'][100] = 2.19662594795
data['Intel Core\ni7-3770'][200] = 5.19148278236
data['Intel Core\ni7-3770'][500] = 17.1793811321
data['ATI Radeon\nHD7970'][5] = 0.0806539058685
data['ATI Radeon\nHD7970'][10] = 0.0720150470734
data['ATI Radeon\nHD7970'][20] = 0.0996639728546
data['ATI Radeon\nHD7970'][50] = 0.127480983734
data['ATI Radeon\nHD7970'][100] = 0.172482967377
data['ATI Radeon\nHD7970'][200] = 0.290589809418
data['ATI Radeon\nHD7970'][500] = 0.946373224258
data['NVidia Tesla\nC2050'][5] = 0.115026950836
data['NVidia Tesla\nC2050'][10] = 0.117651939392
data['NVidia Tesla\nC2050'][20] = 0.13679599762
data['NVidia Tesla\nC2050'][50] = 0.269647121429
data['NVidia Tesla\nC2050'][100] = 0.503059148788
data['NVidia Tesla\nC2050'][200] = 1.04531693459
data['NVidia Tesla\nC2050'][500] = 3.5938680172
data['Intel Xeon\nE5540'][5] = 0.211802959442
data['Intel Xeon\nE5540'][10] = 0.294693946838
data['Intel Xeon\nE5540'][20] = 0.505737066269
data['Intel Xeon\nE5540'][50] = 1.15038609505
data['Intel Xeon\nE5540'][100] = 2.24994087219
data['Intel Xeon\nE5540'][200] = 4.77657580376
data['Intel Xeon\nE5540'][500] = 16.3166491985
data['NVidia\nQuadro\nK4000'][5] = 0.174999952316
data['NVidia\nQuadro\nK4000'][10] = 0.174999952316
data['NVidia\nQuadro\nK4000'][20] = 0.21799993515
data['NVidia\nQuadro\nK4000'][50] = 0.470999956131
data['NVidia\nQuadro\nK4000'][100] = 0.890000104904
data['NVidia\nQuadro\nK4000'][200] = 1.90800023079
data['NVidia\nQuadro\nK4000'][500] = 6.67699980736
data['Intel Xeon\nE5-2620'][5] = 0.216000080109
data['Intel Xeon\nE5-2620'][10] = 0.287999868393
data['Intel Xeon\nE5-2620'][20] = 0.450999975204
data['Intel Xeon\nE5-2620'][50] = 0.893999814987
data['Intel Xeon\nE5-2620'][100] = 1.68300008774
data['Intel Xeon\nE5-2620'][200] = 3.46999979019
data['Intel Xeon\nE5-2620'][500] = 12.3960001469

plt.figure(figsize=(4,4))
ax = plt.subplot(1,1,1)

for ix, name in enumerate(data.keys()):
    n = [neurons[d] for d in data[name].keys()]
    plt.plot(n, data[name].values(), '.-', ms=15, color=colors[ix])
    yoffset = 0
    if "3770" in name:
        yoffset = 1.2
    elif "5540" in name:
        yoffset = -0.6
    elif "Quadro" in name:
        yoffset = 0.5
    elif "Tesla" in name:
        yoffset = 0.3
    elif "Radeon" in name:
        yoffset = 0.2

    if name == "JavaNengo":
        plt.text(n[1] + 12000, data[name].values()[1], name, color=colors[ix],
                 ha='left', va='center', fontsize=12)
    else:
        plt.text(n[-1] + 12000, data[name].values()[-1] + yoffset, name,
                 color=colors[ix], ha='left', va='center', fontsize=12)

dims = [5, 200, 500]  # Ignoring 10, 20, 50, 100
xval = [neurons[d] for d in dims]
xlabels = ["%d neurons\n%d dimensions" % (neurons[d], d) for d in dims]
plt.xticks(xval, xlabels)
plt.xlim(0, 500000)
plt.ylim(0, 27.5)
ax.tick_params(axis='x', which='major', labelsize=11)
plt.ylabel('Simulation time (s)')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('left')

plt.subplots_adjust(left=0.12, bottom=0.11, top=0.97, right=0.82)
plt.savefig("../figures/bench_cconv.pdf")
