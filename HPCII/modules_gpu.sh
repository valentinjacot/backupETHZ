daint:
class28
Eex9Koh4pha-

cuda
Username: ssh class28@ela.cscs.ch
Password:  Eex9Koh4pha-
0rd1_qZiu0w39
ssh daint
module load daint-gpu
module swap PrgEnv-cray PrgEnv-gnu
module load cudatoolkit

 module load mvapich2/2.1 
bsub -n 24 -R fullnode ./heat2d_cpu
bsub -n 24 -R fullnode mpirun -n 24 ./heat2d_mpi
bsub -n 4 -W 04:00 -Is bash


