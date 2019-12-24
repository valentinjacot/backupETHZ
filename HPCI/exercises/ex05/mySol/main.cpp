#include <iostream>
#include <random>
#include <cblas.h>
#include <chrono>

void initialize_matrix(const double alpha, double* const A, const int N){
	std::default_random_engine g(0);
	std::uniform_real_distribution<double> u;
	
	for (int i=0; i<N;i++){
		for(int j=i+1; j<N; j++){
			const double rand = u(g);
			A[i*N + j] = rand;
			A[j*N + i] = rand;
		}
		A[i*N+i] = (i+1)*alpha;
	}
}

void dgePowerMethod(int n, const double *A, double *b, int maxSteps, double eps){
	double *v = new double[n], *res= new double[n];
	int steps;long int i=0;
	for( ; i < maxSteps; i++){
		cblas_dgemv(CblasRowMajor, CblasNoTrans, n, n, 1.0, A, n,b,1,0.0,v,1);
		double nrm = cblas_dnrm2(n,v,1);
		cblas_dscal(n,1.0/nrm,v,1);
		
		cblas_dcopy(n,v,1,res,1);
		cblas_daxpy(n,-1.0,b,1,res,1);
		cblas_dcopy(n,v,1,b,1);
		steps=i;
		double error = cblas_dnrm2(n,res,1);
		if (error < n * eps)
			break;
	}
    //double lambda1 = cblas_ddot(N, q0, 1, q1, 1);

	maxSteps =i;
	delete[] v;
	delete[] res;
	
}
int main(int argc, char *argv[]){
	if(argc!=3) {
        std::cout << "\nUsage: ./power_cblas <N> <alpha>\n" << std::endl;
        exit(1);
    }

    const int N = atoi(argv[1]);
    const double alpha = atof(argv[2]);
	double* A = new double[N*N];
	initialize_matrix(alpha,A,N);
	int iter = 1E8;
	
	double* q0 = new double[N];
//	double* q1 = new double[N];
	
	for (int i = 0;  i < N; i++)
		q0[i]=0;
	q0[0]=1;
	
	auto tstart = std::chrono::steady_clock::now();
	dgePowerMethod(N,A,q0,iter,1E-12);
	auto tend = std::chrono::steady_clock::now();
	auto time = std::chrono::duration_cast<std::chrono::milliseconds>(tend - tstart).count();

    std::cout << "  Largest eigenvalue = " << q0[0] << ",  iterations = " << iter << std::endl;

	std::cout << time <<std::endl;
	
}
