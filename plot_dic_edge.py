import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def make_plot(fname, real_img, edge1):
    #gs = mpl.gridspec.GridSpec(2, 2)
    ax1 = plt.subplot(111)

    ax1.imshow(real_img, cmap = 'gray')
    edge1x, edge1y = edge1.transpose()
    ax1.plot(edge1y, edge1x, 'b')
    xmax, ymax = real_img.shape
    plt.xlim((0, ymax))
    plt.ylim((xmax, 0))

    plt.suptitle(fname)
    
    plt.savefig(fname, dpi = 100)
    plt.close()
