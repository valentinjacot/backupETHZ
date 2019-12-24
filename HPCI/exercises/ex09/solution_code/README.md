# Exercise 09: Sparse linear algebra with MPI

    mpi.h: implementation of Mul()
    io.h: implementation of WriteMpiGather() and WriteMpiIo()


# Examples

## Diffusion equation

    ./run 
    ./plotall

## Game of Life

    CASE=1 N=1024 OUTFMT=6 ./run 

movie `mov/life.mp4`


## Wave equation

    CASE=2 N=256 OUTFMT=6 ./run 

movie `mov/wave.mp4`
