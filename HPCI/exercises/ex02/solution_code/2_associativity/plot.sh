#!/bin/bash

NAME=${1:-results}

gnuplot -e "
set xlabel 'Number of arrays';
set ylabel 'GFLOPS';
set grid ytics;
set grid;
set terminal pngcairo;
set output '${NAME}.png';
set yrange [0.0:2.0];
set arrow from 8,0 to 8,2.0 nohead;
plot '$NAME.txt' i 0 u 2:3 w l t 'padding=0B', '$NAME.txt' i 1 u 2:3 w l t 'padding=64B';
"
