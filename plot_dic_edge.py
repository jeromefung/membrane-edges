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


def make_thread_plot(fname, real_img, edge1, edge2, edge3):
    #gs = mpl.gridspec.GridSpec(2, 2)
    ax1 = plt.subplot(111)

    ax1.imshow(real_img, cmap = 'gray')
    edge1x, edge1y = edge1.transpose()
    ax1.plot(edge1y, edge1x, 'b')
    ax1.plot(edge2[:,1], edge2[:,0], 'g')
    ax1.plot(edge3[:,1], edge3[:,0], 'g')
    xmax, ymax = real_img.shape
    plt.xlim((0, ymax))
    plt.ylim((xmax, 0))

    plt.suptitle(fname)
    
    plt.savefig(fname, dpi = 100)
    plt.close()

def make_layer_plot(fname, real_img, edge1, edge2):
    #gs = mpl.gridspec.GridSpec(2, 2)
    ax1 = plt.subplot(111)

    ax1.imshow(real_img, cmap = 'gray')
    edge1x, edge1y = edge1.transpose()
    ax1.plot(edge1y, edge1x, 'b')
    ax1.plot(edge2[:,1], edge2[:,0], 'g')
    xmax, ymax = real_img.shape
    plt.xlim((0, ymax))
    plt.ylim((xmax, 0))

    plt.suptitle(fname)
    
    plt.savefig(fname, dpi = 100)
    plt.close()
