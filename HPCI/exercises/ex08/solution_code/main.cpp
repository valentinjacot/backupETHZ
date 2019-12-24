#include <iostream>
#include <iomanip>
#include <random>
#include <cmath>
#include <mpi.h>

inline long exact(const long N){
    return N/2 *(N+1);
}

void reduce_mpi(const int rank, long& sum){
    if(rank == 0){
        MPI_Reduce(MPI_IN_PLACE, &sum, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    }else{
        MPI_Reduce(&sum, &sum, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    }
}

// PRE: size is a power of 2 for simplicity
void reduce_manual(int rank, int size, long& sum){
    const int TAG = 1337;
    
    for(int send_rec_border = size/2; send_rec_border >= 1; send_rec_border/=2)
    {
        if(rank < send_rec_border)
        {
            long buffer;
            MPI_Recv(&buffer, 1, MPI_LONG, rank+send_rec_border, TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            sum += buffer;
        }else if(rank <= send_rec_border*2){
            MPI_Send(&sum, 1, MPI_LONG, rank-send_rec_border, TAG, MPI_COMM_WORLD);
        }
    }
}


int main(int argc, char** argv){
    const long N = 1000000;
    
    // Initialize MPI
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // -------------------------
    // Perform the local sum:
    // -------------------------
    long sum = 0;
    
    // Determine work load per rank
    long N_per_rank = N / size;
    
    // Remark: Start at 1!!!
    long N_start = rank * N_per_rank + 1;
    long N_end = (rank+1) * N_per_rank;
    // the last rank has to do some more additions if size does not divide N
    if(rank == size-1){
        N_end += N % size;
    }
    
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
    MPI_Finalize();
    
    return 0;
}
