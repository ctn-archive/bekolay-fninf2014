import os.path
import subprocess

import svgutils.transform as sg


pt_to_px = 1.25
px_to_pt = 1.0 / pt_to_px
mm_to_px = 3.54


def el(char, path, x, y, w=None, h=None, scale=1, offset=(0, 0)):
    svg, textel = [], []

    if char is not None:
        textel = [sg.TextElement(x + offset[0] + 0.5, y + offset[1] + 11, char,
                                 size=14, weight='bold', font='Arial')]

    if path is not None:
        svg = sg.fromfile(path)
        if w is not None and h is not None:
            svg.set_size((w, h))
        svg = svg.getroot()
        svg.moveto(str(x), str(y), scale)
        svg = [svg]
    return svg + textel


def svgfig(w, h):
    w = str(w) + 'px'
    h = str(h) + 'px'
    return sg.SVGFigure(w, h)


def savefig(fig, out):
    fig.save('%s.svg' % out)
    print "Saved %s.svg" % out
    subprocess.call(['inkscape',
                     '--export-pdf=%s.pdf' % out,
                     '%s.svg' % out])
    print "Saved %s.pdf" % out
    subprocess.call(['inkscape', '--export-text-to-path',
                     '--export-eps=%s.eps' % out,
                     '%s.svg' % out])
    print "Saved %s.eps" % out
    subprocess.call(['inkscape',
                     '--export-dpi=1200',
                     '--export-png=%s.png' % out,
                     '%s.svg' % out])
    print "Saved %s.png" % out
    subprocess.call(['convert',
                     '%s.png' % out,
                     # '-compress', 'LZW',
                     '%s.tiff' % out])
    print "Saved %s.tiff" % out

def check_fig(in_svg, out_svg):
    for svg in in_svg:
        if not os.path.exists(svg):
            return False

    if os.path.exists(out_svg + ".svg"):
        return False

    print "Generating " + out_svg + ".svg"
    return True


def fig1():
    in_svg = ['../figures/nef_summary_enc.svg',
              '../figures/nef_summary_dec.svg',
              '../figures/nef_summary_trans.svg',
              '../figures/nef_summary_dyn.svg']
    out_svg = '../figures/fig1'

    if not check_fig(in_svg, out_svg): return

    w = 127 * pt_to_px  # pt
    h = 255 * pt_to_px  # pt
    fig = svgfig(w * 4, h)
    fig.append(el('A', in_svg[0], 0, 0, scale=pt_to_px))
    fig.append(el('B', in_svg[1], w, 0, scale=pt_to_px))
    fig.append(el('C', in_svg[2], w * 2, 0, scale=pt_to_px))
    fig.append(el('D', in_svg[3], w * 3, 0, scale=pt_to_px))
    savefig(fig, out_svg)


def fig2():
    in_svg = ['../figures/sim.svg']
    out_svg = '../figures/fig2'
    if not check_fig(in_svg, out_svg): return
    fig = svgfig(637.88269, 523.79523)
    fig.append(el(None, in_svg[0], 0, 0))
    savefig(fig, out_svg)


def fig3():
    in_svg = ['../figures/comm_channel.svg',
              '../figures/comm_channel_code.svg',
              '../figures/comm_channel_res.svg']
    out_svg = '../figures/fig3'
    if not check_fig(in_svg, out_svg): return
    w = [301.16302, 150.05182, 120 * pt_to_px]
    h = [42.223442, 137.28253, 122 * pt_to_px]
    fig = svgfig(max(w), h[0] + max(h[1:]) + 0.8 * mm_to_px)
    fig.append(el('A', in_svg[0], 0, 1 * mm_to_px,
                  offset=(0, -1 * mm_to_px)))
    fig.append(el('B', in_svg[1], 0, h[0] + 3.5 * mm_to_px,
                  offset=(0, -3.5 * mm_to_px)))
    fig.append(el('C', in_svg[2], w[1], h[0] + 0.8 * mm_to_px,
                  offset=(-2*mm_to_px,0), scale=pt_to_px))
    savefig(fig, out_svg)


def fig4():
    in_svg = ['../figures/pynn.svg']
    out_svg = '../figures/fig4'
    if not check_fig(in_svg, out_svg): return
    fig = svgfig(637.80621, 472.07712)
    fig.append(el(None, in_svg[0], 0, 0))
    savefig(fig, out_svg)


def fig5():
    in_svg = ['../figures/lorenz_code.svg',
              '../figures/lorenz.svg',
              '../figures/lorenz_res.svg']
    out_svg = '../figures/fig5'
    if not check_fig(in_svg, out_svg): return
    w = [124.01178, 177.19678, 283 * pt_to_px * 0.5]
    h = [120.48283, 52.740223, 226 * pt_to_px * 0.5]
    fig = svgfig(w[0] + max(w[1:]), max([h[0], sum(h[1:])]))
    fig.append(el('A', in_svg[0], 0, 10 * mm_to_px,
                  offset=(0, -3.5 * mm_to_px)))
    fig.append(el('B', in_svg[1], w[0], 0))
    fig.append(el('C', in_svg[2], w[0], h[1], scale=pt_to_px * 0.5,
                  offset=(0, 1 * mm_to_px)))
    fig.append(el('D', None, w[0], h[1], offset=(27 * mm_to_px, 1 * mm_to_px)))
    fig.append(el('E', None, w[0], h[1], offset=(27 * mm_to_px, 33 * mm_to_px)))
    savefig(fig, out_svg)


def fig6():
    in_svg = ['../figures/cconv_code.svg',
              '../figures/cconv.svg',
              '../figures/cconv_res.svg']
    out_svg = '../figures/fig6'
    if not check_fig(in_svg, out_svg): return
    w = [141.74609, 354.01544, 113 * pt_to_px]
    h = [114.64618, 174.19658, 139 * pt_to_px]
    fig = svgfig(sum(w), max(h))
    fig.append(el('A', in_svg[0], 0, 3 * mm_to_px, offset=(0, -3 * mm_to_px)))
    fig.append(el('B', in_svg[1], w[0], 0))
    fig.append(el('C', in_svg[2], w[0] + w[1], 0, scale=pt_to_px,
                  offset=(-1.5 * mm_to_px, 0)))
    savefig(fig, out_svg)


def fig7():
    in_svg = ['../figures/bench_cchannel.svg',
              '../figures/bench_lorenz.svg',
              '../figures/bench_cconv.svg']
    out_svg =  '../figures/fig7'
    if not check_fig(in_svg, out_svg): return

    w = 170 * pt_to_px  # pt
    h = 204 * pt_to_px  # pt
    fig = svgfig(w * 3, h)
    fig.append(el('A', in_svg[0], 0, 0, scale=pt_to_px))
    fig.append(el('B', in_svg[1], w, 0, scale=pt_to_px))
    fig.append(el('C', in_svg[2], w * 2, 0, scale=pt_to_px))
    savefig(fig, out_svg)

if __name__ == '__main__':
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    fig6()
    fig7()
