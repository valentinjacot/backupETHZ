// Skeleton code for HPCSE I (2016HS) Exam, 23.12.2016
// Prof. M. Troyer, Dr. P. Hadjidoukas
// Coding 2 : Diffusion Statistics

#include <iostream>
#include <algorithm>
#include <string>
#include <fstream>
#include <cassert>
#include <vector>
#include <cmath>
#include <mpi.h>
#include <stdio.h>
#include "timer.hpp"

struct Diagnostics
{
    double time;
    double heat;

    Diagnostics(double time, double heat) : time(time), heat(heat) {}
};


class Diffusion2D_MPI {
public:
    Diffusion2D_MPI(const double D,
                const double L,
                const int N,
                const double dt,
                const int rank,
                const int procs)
    : D_(D), L_(L), N_(N), dt_(dt), rank_(rank), procs_(procs)
    {
	// initialize to zero
	t_  = 0.0;

        /// real space grid spacing
        dr_ = L_ / (N_ - 1);
        
        /// stencil factor
        fac_ = dt_ * D_ / (dr_ * dr_);

	// number of rows per process
	local_N_ = N_ / procs_;	

	// small correction for the last process
	if (rank_ == procs - 1) local_N_ += (N_ % procs_);

	// actual dimension of a row (+2 for the ghosts)
	real_N_ = N_ + 2;
	Ntot = (local_N_ + 2) * (N_+2);

	rho_.resize(Ntot, 0.);		// zero values
	rho_tmp.resize(Ntot, 0.);	// zero values

        // check that the timestep satisfies the restriction for stability
	if (rank_ == 0)
	        std::cout << "timestep from stability condition is " << dr_*dr_/(4.*D_) << std::endl;
        
        initialize_density();
    }
    
    double advance()
    {
	MPI_Request req[4];
	MPI_Status status[4];

	int prev_rank = rank_ - 1;
	int next_rank = rank_ + 1;

	if (prev_rank >= 0) {
		MPI_Irecv(&rho_[           0*real_N_+1], N_, MPI_DOUBLE, prev_rank, 100, MPI_COMM_WORLD, &req[0]);
		MPI_Isend(&rho_[           1*real_N_+1], N_, MPI_DOUBLE, prev_rank, 100, MPI_COMM_WORLD, &req[1]);
	} else {
		req[0] = MPI_REQUEST_NULL;
		req[1] = MPI_REQUEST_NULL;
	}
	
	if (next_rank < procs_) {
		MPI_Irecv(&rho_[(local_N_+1)*real_N_+1], N_, MPI_DOUBLE, next_rank, 100, MPI_COMM_WORLD, &req[2]);
		MPI_Isend(&rho_[    local_N_*real_N_+1], N_, MPI_DOUBLE, next_rank, 100, MPI_COMM_WORLD, &req[3]);
	} else {
		req[2] = MPI_REQUEST_NULL;
		req[3] = MPI_REQUEST_NULL;
	}

        /// Dirichlet boundaries; central differences in space, forward Euler
        /// in time
	// update the interior rows
        for(int i = 2; i < local_N_; ++i) {
          for(int j = 1; j <= N_; ++j) {
            rho_tmp[i*real_N_ + j] = rho_[i*real_N_ + j] +
            fac_
            *
            (
             rho_[i*real_N_ + (j+1)]
             +
             rho_[i*real_N_ + (j-1)]
             +
             rho_[(i+1)*real_N_ + j]
             +
             rho_[(i-1)*real_N_ + j]
             -
             4.*rho_[i*real_N_ + j]
             );
          }
        }

	// ensure boundaries have arrived
	MPI_Waitall(4, req, status);

	// update the first and the last rows
        for(int i = 1; i <= local_N_; i += (local_N_-1)) {
          for(int j = 1; j <= N_; ++j) {
            rho_tmp[i*real_N_ + j] = rho_[i*real_N_ + j] +
            fac_
            *
            (
             rho_[i*real_N_ + (j+1)]
             +
             rho_[i*real_N_ + (j-1)]
             +
             rho_[(i+1)*real_N_ + j]
             +
             rho_[(i-1)*real_N_ + j]
             -
             4.*rho_[i*real_N_ + j]
             );
          }
        }

        /// use swap instead of rho_=rho_tmp. this is much more efficient, because it does not have to copy
        /// element by element.
        using std::swap;
        swap(rho_tmp, rho_);
        
        t_ += dt_;
        
        return t_;
    }
    
    void compute_max_density()
    {
        // This routine finds the value of max density (max_rho) and its location (max_i, max_j) - it assumes there are no duplicate values
        // It prints the value and location of maximum density in the local subdomain.
        // The overall result is correct only if the number of MPI processes (procs_) is 1.

        double max_rho;
        int max_i,  max_j;

        max_rho = rho_[1*real_N_ + 1];
        max_i = 1;
        max_j = 1;

        for(int i = 1; i <= local_N_; ++i)
            for(int j = 1; j <= N_; ++j)
            {
                if (rho_[i*real_N_ + j] > max_rho)
                {
                    max_rho = rho_[i*real_N_ + j];
                    max_i = i;
                    max_j = j;
                }
            }

	int max_gi = rank_ * (N_ / procs_) + max_i - 1;	// convert local index to global index
	int max_gj = max_j - 1;

        std::cout << "=====================================\n";
        std::cout << "Output of compute_max_density():\n";
        std::cout << "Max rho: " << max_rho << "\n";
        std::cout << "Matrix location: " << max_gi << "\t" << max_gj << "\n";
        std::cout << "Physical location: " << (max_gi)*dr_ - L_/2. <<  "\t" << (max_gj)*dr_ - L_/2 << "\n";
    }

    void compute_max_density_mpi()
    {
        // This routine finds the value of max density (max_rho) and its location (max_i, max_j) - it assumes there are no duplicate values
        // It currently includes the code of compute_max_density() (see above).
        
        // TODO: add your MPI code here for subquestion (a)
        // It must give the correct overall result (maximum density and its location) for any number of MPI processes (procs_)
        // Process (rank_) 0 must print the results
        // NOTE: you are free to adapt the code to your needs

        double max_rho;
        int max_i,  max_j;

        max_rho = rho_[1*real_N_ + 1];
        max_i = 1;
        max_j = 1;

        for(int i = 1; i <= local_N_; ++i)
            for(int j = 1; j <= N_; ++j)
            {
                if (rho_[i*real_N_ + j] > max_rho)
                {
                    max_rho = rho_[i*real_N_ + j];
                    max_i = i;
                    max_j = j;
                }
            }

	int max_gi = rank_ * (N_ / procs_) + max_i - 1;	// convert local index to global index
	int max_gj = max_j - 1;

	double *gather_max_rho = (double *)malloc(procs_*sizeof(double));
	int *gather_maxi = (int *)malloc(procs_*sizeof(int));
	int *gather_maxj = (int *)malloc(procs_*sizeof(int));

	MPI_Gather(&max_rho, 1, MPI_DOUBLE, gather_max_rho, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Gather(&max_gi, 1, MPI_INT, gather_maxi, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Gather(&max_gj, 1, MPI_INT, gather_maxj, 1, MPI_INT, 0, MPI_COMM_WORLD);
	
	double gmax_rho;
	int gmax_gi;
	int gmax_gj;

	if (rank_ == 0)
	{
		gmax_rho = gather_max_rho[0];
		gmax_gi = gather_maxi[0];
		gmax_gj = gather_maxj[0];

		for (int i = 1; i < procs_; i++)
		{
			if (gather_max_rho[i] > gmax_rho)
			{
				gmax_rho = gather_max_rho[i];
				gmax_gi = gather_maxi[i];
				gmax_gj = gather_maxj[i];
			}
		}
	}

	free(gather_max_rho);
	free(gather_maxi);
	free(gather_maxj);

	if (rank_ == 0)
	{
		// NOTE: you are free to replace max_rho, max_gi, max_gj with your own variables

        	std::cout << "=====================================\n";
        	std::cout << "Output of compute_max_density_mpi():\n";
        	std::cout << "Max rho: " << gmax_rho << "\n";
        	std::cout << "Matrix location: " << gmax_gi << "\t" << gmax_gj << "\n";
        	std::cout << "Physical location: " << (gmax_gi)*dr_ - L_/2. <<  "\t" << (gmax_gj)*dr_ - L_/2 << "\n";
	}
    }

    void compute_diagnostics(const double t)
    {
        double heat = 0.0;

        for(int i = 1; i <= local_N_; ++i)
            for(int j = 1; j <= N_; ++j)
                heat += rho_[i*real_N_ + j] * dr_ * dr_;

        MPI_Reduce(rank_ == 0? MPI_IN_PLACE: &heat, &heat, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        if (rank_ == 0) {
#if DEBUG
            std::cout << "t = " << t << " heat = " << heat << '\n';
#endif
            diag.push_back(Diagnostics(t, heat));
        }
    }

    void write_diagnostics(const std::string &filename) const
    {
        std::ofstream out_file(filename, std::ios::out);
        for (const Diagnostics &d : diag)
            out_file << d.time << '\t' << d.heat << '\n';
        out_file.close();
    }

private:
    
    void initialize_density()
    {
	int gi;
        /// initialize rho(x,y,t=0)
        double bound = 1/2.;

        for (int i = 1; i <= local_N_; ++i) {
            gi = rank_ * (N_ / procs_) + i;	// convert local index to global index
            for (int j = 1; j <= N_; ++j) {
                if (std::abs((gi-1)*dr_ - L_/2.) < bound && std::abs((j-1)*dr_ - L_/2.) < bound) {
                    rho_[i*real_N_ + j] = 1 + 0.001*((gi-1)+(j-1)); // small adjustment to get a unique final result

                } else {
                    rho_[i*real_N_ + j] = 0;
                }
            }
        }

    }
    
    double D_, L_;
    int N_, Ntot, local_N_, real_N_;
    
    double dr_, dt_, t_, fac_;

    int rank_, procs_;
    
    std::vector<double> rho_, rho_tmp;
    std::vector<Diagnostics> diag;
};


int main(int argc, char* argv[])
{
    const double D  = 1;
    const double L  = 1;
    const int  N  = 1024;
    const double dt = 1e-7;

    MPI_Init(&argc, &argv);

    int rank, procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &procs);

    if (rank == 0)
        std::cout << "Running with " << procs  << " MPI processes" << std::endl;
    
    Diffusion2D_MPI system(D, L, N, dt, rank, procs);

    const double tmax = 10000 * dt;
    double time = 0;
    
    timer t;
    
    int i = 0;
    t.start();
    while (time < tmax) {
        time = system.advance();
#ifndef _PERF_
        system.compute_diagnostics(dt * i);
#endif
        i++;
	if (i == 100) break; // 100 steps are enough
    }
    t.stop();
    

    if (rank == 0)
      std::cout << "Timing: " << N << " " << t.get_timing() << std::endl;
    
    if (procs == 1) {
        system.compute_max_density();
    }
    else {
        system.compute_max_density_mpi();
    }

#ifndef _PERF_
    if (rank == 0)
        system.write_diagnostics("diagnostics_mpi.dat");
#endif

    MPI_Finalize();
    return 0;
}
