#!/usr/bin/env bash
# File       : environ.sh
# Created    : Wed Nov 28 2018 12:21:49 PM (+0100)
# Description: The exercise has been developed and tested with this environment
# Copyright 2018 ETH Zurich. All Rights Reserved.
set -eu

echo 'Loading exercise environment'
module purge
module load new gcc/6.3.0 open_mpi/2.1.1 python/3.6.1
