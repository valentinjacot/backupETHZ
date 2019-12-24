#!/usr/bin/env python
# File       : print_png.py
# Created    : Tue Nov 27 2018 06:26:27 PM (+0100)
# Description: Print png images
# Copyright 2018 ETH Zurich. All Rights Reserved.
import numpy as np
import matplotlib as mpl
mpl.use('Agg') # render without X server running
import matplotlib.pyplot as plt
import gzip
import argparse

def parseArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument('-r', '--ref', type=str, help='Reference image', required=True)
    parser.add_argument('-t', '--test', type=str, help='Test image', required=True)
    parser.add_argument('-o', '--output', type=str, help='Outfile name', required=True)
    parser.add_argument('-d', '--dim', type=int, help='Image dimesnion (square image)', required=True)

    return parser.parse_known_args()


if __name__ == "__main__":
    args, unknown = parseArgs()
    dim = args.dim
    with gzip.open(args.ref, 'rb') as f:
        ref = np.frombuffer(f.read(), dtype=np.float64).reshape( (dim,dim) )
    with gzip.open(args.test, 'rb') as f:
        test = np.frombuffer(f.read(), dtype=np.float64).reshape( (dim,dim) )

    fig, ax = plt.subplots(1,2)
    fig.set_size_inches(8,16)
    cmap = plt.cm.jet
    for a in ax:
        a.axis('off')
        a.get_xaxis().set_visible(False)
        a.get_yaxis().set_visible(False)
    ax[0].imshow(ref, cmap=cmap)
    ax[0].set_title(args.ref, fontsize=6)
    ax[1].imshow(test, cmap=cmap)
    ax[1].set_title(args.test, fontsize=6)
    fig.savefig(args.output, dpi=450, bbox_inches='tight')
