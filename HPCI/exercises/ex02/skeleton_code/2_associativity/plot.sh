#!/bin/bash

gnuplot -e "
set xlabel 'Number of arrays';
set ylabel 'GFLOPS';
set grid ytics;
set grid;
set terminal pngcairo;
set output 'results.png';
set yrange [0.0:1.0];
plot 'results.txt' i 0 u 2:3 w l t 'padding=0B', 'results.txt' i 1 u 2:3 w l t 'padding=64B';
"

