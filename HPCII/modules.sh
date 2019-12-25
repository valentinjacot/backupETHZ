module load new
module load gcc/6.3.0
module load intel/2018.1
module load impi/2018.1.163

export PATH="$HOME/usr/upcxx/bin/:$PATH"
export UPCXX_GASNET_CONDUIT=smp   udp
export UPCXX_THREADMODE=seq
export UPCXX_CODEMODE=O3
make CXX=upcxx


mpicxx -c -cxx=icpc task2b_mpi.cpp 
mpicxx -o -cxx=icpc task2b_mpi task2b_mpi.o sampler/sampler.o
mpirun ./task2b_mpi -np 12

bsub -n 24 -R fullnode upcxx-run -n 24 ./task1_n4

