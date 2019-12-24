#include <iostream>
#include <iomanip>
#include <random>
#include <cmath>
#include <mpi.h>

inline long exact(const long N){
    // TODO b): Implement the analytical solution.
    return N/2 *(N+1);
}

void reduce_mpi(const int rank, long& sum){
    // TODO e): Perform the reduction using blocking collectives.
    if(rank==0){
		MPI_Reduce(MPI_IN_PLACE, &sum, 1, MPI_LONG, MPI_SUM,0, MPI_COMM_WORLD);
	}else{
		MPI_Reduce(&sum, &sum, 1, MPI_LONG, MPI_SUM,0, MPI_COMM_WORLD);
	}
}

// PRE: size is a power of 2 for simplicity
void reduce_manual(int rank, int size, long& sum){
    // TODO f): Implement a tree based reduction using blocking point-to-point communication.
    int Nn = size/2;
    int tag =42;
    
    while(Nn>=1){
//		if(rank>Nn)break;
		if(rank<Nn){
			long buf;
			MPI_Recv(&buf,1 , MPI_LONG,rank + Nn,tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			sum+=buf;
		}else if(rank>=Nn){
			MPI_Send(&sum, 1, MPI_LONG,rank - Nn,tag, MPI_COMM_WORLD); 
		}
		Nn/=2;
	} 
}


int main(int argc, char** argv){
    const long N = 1000000;
    
    // Initialize MPI
    int rank, size;
    // TODO c): Initialize MPI and obtain the rank and the number of processes (size)
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // -------------------------
    // Perform the local sum:
    // -------------------------
    long sum = 0;
    
    // Determine work load per rank
    long N_per_rank = N / size;
    
    // TODO d): Determine the range of the subsum that should be calculated by this rank.
    long N_start = rank * N_per_rank + 1;
    long N_end = (rank+1) * N_per_rank;
    if(rank == size-1){N_end +=N % size;} 
    // N_start + (N_start+1) + ... + (N_start+N_per_rank-1)
    for(long i = N_start; i <= N_end; ++i){
        sum += i;
    }
    
    // -------------------------
    // Reduction
    // -------------------------
    //reduce_mpi(rank, sum);
    reduce_manual(rank, size, sum);
    
    // -------------------------
    // Print the result
    // -------------------------
    if(rank == 0){
        std::cout << std::left << std::setw(25) << "Final result (exact): " << exact(N) << std::endl;
        std::cout << std::left << std::setw(25) << "Final result (MPI): " << sum << std::endl;
    }
    // Finalize MPI
    // TODO c): Finalize MPI
    MPI_Finalize();
    return 0;
}
