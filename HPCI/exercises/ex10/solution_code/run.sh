#!/usr/bin/env bash
# File       : run.sh
# Created    : Wed Nov 28 2018 11:07:50 AM (+0100)
# Description: Execute floating point compressor
#              ./run.sh <compression tolerance>
# Copyright 2018 ETH Zurich. All Rights Reserved.
set -Eeuo pipefail

source ./environ.sh

tolerance="${1}"; shift
dim=4096

tol=$(python -c "print('{:.3e}'.format(${tolerance}))")

# build
make mpi_float_compression

# run compressor
mpirun -n 4 ./mpi_float_compression "${tol}" ${dim} cyclone.bin.gz

# post-process
python print_png.py \
    --ref 'cyclone.bin.gz' \
    --test "cyclone_t${tol}.bin.gz" \
    --output "cyclone_t${tol}.png" \
    --dim ${dim}

