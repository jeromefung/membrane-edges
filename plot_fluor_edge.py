import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def make_plot(fname, real_img, grad_img, edge1):
    #gs = mpl.gridspec.GridSpec(2, 2)
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)

    ax1.imshow(real_img, cmap = 'gray')
    edge1x, edge1y = edge1.transpose()
#    edge2x, edge2y = edge2.transpose()
    ax1.plot(edge1y, edge1x, 'b')
#    ax1.plot(edge2y, edge2x, 'g')

    ax2.imshow(grad_img, cmap = 'gray')
    ax2.plot(edge1y, edge1x, 'b')
#    ax2.plot(edge2y, edge2x, 'g')
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_ylim(ax1.get_ylim())

    plt.suptitle(fname)
    
    plt.savefig(fname, dpi = 100)
    plt.close()
