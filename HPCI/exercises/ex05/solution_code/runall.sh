#!/bin/bash

# Build code
# module load openblas mkl
# make clean && make

# Clear test environment
make clear


N=1024
alpha=(0.125 0.25 0.5 1 1.5 2 4 8 16)


for a in ${alpha[@]}; do

    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    echo "alpha = $a"
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

    echo "Power method (manual):"
    ./power_manual $N $a

    echo "Power method (cblas):"
    ./power_cblas $N $a

    echo "Full eigenvalue solution (lapack):"
    ./eigenv_lapack $N $a
done
