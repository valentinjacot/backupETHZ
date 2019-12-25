# *
# *  plotting.py
# *  PyPi4U
# *
# *  Authors:
# *     Philipp Mueller  - muellphi@ethz.ch
# *     Georgios Arampatzis - arampatzis@collegium.ethz.ch
# *
# *  Copyright 2018 ETH Zurich. All rights reserved.
# *


import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import scipy.interpolate
import argparse


def plot_histogram(ax, theta):
    """Plot histogram of theta to diagonal"""
    num_bins = 50
    for i in range(theta.shape[1]):
        axi = ax[i, i] if (theta.shape[1]-1>0) else ax
        hist, bins, _ = axi.hist(theta[:, i], num_bins, normed=1,
                                      color=(51/255, 1, 51/255), ec='black')
        if i == 0:

            # Rescale hist to scale of theta -> get correct axis titles
            hist = hist / np.max(hist) * (axi.get_xlim()[1] -
                                          axi.get_xlim()[0])
            bottom = axi.get_xlim()[0]

            widths = np.diff(bins)
            axi.cla()
            axi.bar(bins[:-1], hist, widths,
                         color=(51/255, 1, 51/255), ec='black', bottom=bottom)
            axi.set_ylim(axi.get_xlim())

            axi.set_xticklabels([])

        elif i == theta.shape[1] - 1:
            axi.set_yticklabels([])
        else:
            axi.set_xticklabels([])
            axi.set_yticklabels([])
        axi.tick_params(axis='both', which='both', length=0)


def plot_upper_triangle(ax, theta, lik=None):
    """Plot scatter plot to upper triangle of plots"""
    for i in range(theta.shape[1]):
        for j in range(i + 1, theta.shape[1]):
            if lik is None:
                ax[i, j].plot(theta[:, j], theta[:, i], '.', markersize=1)
            else:
                ax[i, j].scatter(theta[:, j], theta[:, i], marker='o', s=10,
                                 c=lik, facecolors='none', alpha=0.5)
            ax[i, j].set_xticklabels([])
            ax[i, j].set_yticklabels([])


def plot_lower_triangle(ax, theta):
    """Plot 2d histogram to lower triangle of plots"""
    for i in range(theta.shape[1]):
        for j in range(i):
            # returns bin values, bin edges and bin edges
            H, xe, ye = np.histogram2d(theta[:, j], theta[:, i], 8,
                                       normed=True)
            # plot and interpolate data
            ax[i, j].imshow(H.T, aspect="auto", interpolation='spline16',
                            origin='lower', extent=np.hstack((
                                                ax[j, j].get_xlim(),
                                                ax[i, i].get_xlim())),
                                                cmap=plt.get_cmap('jet'))
            if i < theta.shape[1]-1:
                ax[i, j].set_xticklabels([])
            if j > 0:
                ax[i, j].set_yticklabels([])


def plot_theta(file, startcol=0, endcol=-1, likelihood=False):
    theta = np.loadtxt(file, delimiter=',') # format : var1 var2 .... likelihood
    ncols = theta.shape[1]
    # use startcol - endcol for variables + -1 for likelihood
    if ( startcol != 0 or endcol != -1 ):
        theta = theta[:, np.append( np.arange(startcol,endcol+1), [ncols-1])]
    fig, ax = plt.subplots(theta.shape[1]-1, theta.shape[1]-1) # as many subplots as variables
    plot_histogram(ax, theta[:, :-1])
    if likelihood:
        plot_upper_triangle(ax, theta[:, :-1], theta[:, -1])
    else:
        plot_upper_triangle(ax, theta[:, :-1])
    plot_lower_triangle(ax, theta[:, :-1])
#    plt.jet()
    plt.tight_layout()
    #plt.show()
    plt.savefig('output_plotmatrix_hist.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot generations.')
    parser.add_argument('filename', metavar='filename', help='Select file' +
                        ' for plotting.')
    parser.add_argument('--start_col', dest='start_col', default=0, action='store', type=int, help='starting column (default = 0)')
    parser.add_argument('--end_col', dest='end_col', default=-1, action='store', type=int, help='end column (default = 0)')
    parser.add_argument("-lik", "--likelihood", action="store_true",
                        help="Plot log-likelihood value")
    args = parser.parse_args()
    #  python plotmatrix_hist_euler.py filename --likelihood -> plot all variables + likelihood
    #  python plotmatrix_hist_euler.py --start_col 0 --end_col 0 filename  -> plot only first variable 
    plot_theta(args.filename, args.start_col, args.end_col, args.likelihood)
