#!/usr/bin/env python2

from collections import defaultdict
from textwrap import dedent

import subprocess

def main(filename):
    # Read and split by threads.
    threads = defaultdict(list)
    for line in open(filename):
        tid, info, time = line.split()
        threads[tid].append(time)

    # Split into t_A--t_B and t_B--t_C lines.
    AB = []
    BC = []
    for tid, lines in threads.items():
        for k in range(0, len(lines), 3):
            AB.append((tid, lines[k], lines[k + 1]))
            BC.append((tid, lines[k + 1], lines[k + 2]))

    # Print the lines.
    with open('results-rearranged.txt', 'w') as f:
        for a, b, c in AB:
            f.write("{} {}\n{} {}\n\n".format(b, a, c, a))
        f.write('\n')
        for a, b, c in BC:
            f.write("{} {}\n{} {}\n\n".format(b, a, c, a))

    # Use gnuplot to plot.
    subprocess.call(['gnuplot', '-e', dedent("""
        set xlabel 'Time [s]';
        set ylabel 'Thread ID';
        set grid;
        set yrange [-1:{}];
        set terminal pngcairo;
        set output 'results.png';
        plot 'results-rearranged.txt' i 0 u 1:2 w l t 'Waiting' lw 2, 'results-rearranged.txt' i 1 u 1:2 w l t 'Critical region' lw 3;
    """.format(len(threads) + 1)).strip()])
    print("Plotted into 'results.png'.")


if __name__ == '__main__':
    main('results.txt')
