#include <stdio.h>
#include <stdlib.h>
#include <random>
#include <omp.h>

// Integrand
inline double F(double x, double y) {
  if (x * x + y * y < 1.) { // inside unit circle 
    return 4.; 
  }
  return 0.;
}

// Method 0: serial
double C0(size_t n) {
  // random generator with seed 0
	std::default_random_engine g(0); 
  // uniform distribution in [0, 1]
	std::uniform_real_distribution<double> u; 

	double s = 0.; // sum
	for (size_t i = 0; i < n; ++i) {
		double x = u(g);
		double y = u(g);
		s += F(x, y);
	}
  return s / n;
}

// Method 1: openmp, no arrays 
// TODO: Question 1a.1
double C1(size_t n) {
  // random generator with seed 0
  // uniform distribution in [0, 1]
	double s = 0.; // sum
	int thds;
#pragma omp parallel 
{
	std::default_random_engine g(omp_get_thread_num()); 
	std::uniform_real_distribution<double> u; 
	double l_s = 0.; // local_sum
	thds = omp_get_num_threads();
	printf("%d , %d \n", thds, omp_get_thread_num());
	size_t l_n = n/thds;
	for (size_t i = 0; i < l_n; ++i) {
		double x = u(g);
		double y = u(g);
		l_s += F(x, y);
	}
#pragma omp critical
	s+=l_s;
}	
  return s / n;

}


// Method 2, only `omp parallel for reduction`, arrays without padding
// TODO: Question 1a.2
double C2(size_t n) {
//	int mt = omp_get_max_threads();
	int nt = omp_get_max_threads();
  // random generator with seed 0
	auto* g = new std::default_random_engine[nt]; 
	for(int i = 0; i<nt; i++){
		g[i].seed(48+i);
	}
  // uniform distribution in [0, 1]
	double s = 0.; // sum
#pragma omp parallel for reduction(+:s)
	for (size_t i = 0; i < n; ++i) {
		std::uniform_real_distribution<double> u; 
		double x = u(g[omp_get_thread_num()]);
		double y = u(g[omp_get_thread_num()]);
		s += F(x, y);
	}
	
  delete [] g;
		//printf("%d , %d \n", omp_get_num_threads(), omp_get_thread_num());	
  return s / n;
}

// Method 3, only `omp parallel for reduction`, arrays with padding
// TODO: Question 1a.3
double C3(size_t n) {
	int nt = omp_get_max_threads();
	struct GP{
		std::default_random_engine g;
		char pad[64];
	};
  // random generator with seed 0
	auto* gg = new GP[nt];
	
	for(int i = 0; i<nt; i+=64){
		gg[i].g.seed(48+i);
	}
  // uniform distribution in [0, 1]
	double s = 0.; // sum
#pragma omp parallel for reduction(+:s)
	for (size_t i = 0; i < n; ++i) {
		std::uniform_real_distribution<double> u; 
		double x = u(gg[omp_get_thread_num()].g);
		double y = u(gg[omp_get_thread_num()].g);
		s += F(x, y);
	}
	
  delete [] gg;
		//printf("%d , %d \n", omp_get_num_threads(), omp_get_thread_num());	
  return s / n;
}

// Returns integral of F(x,y) over unit square (0 < x < 1, 0 < y < 1).
// n: number of samples
// m: method
double C(size_t n, size_t m) {
  switch (m) {
    case 0: return C0(n);
    case 1: return C1(n); 
    case 2: return C2(n); 
    case 3: return C3(n); 
    default: printf("Unknown method '%ld'\n", m); abort();
  }
}


int main(int argc, char *argv[]) {
  // default number of samples
  const size_t ndef = 1e8;

  if (argc < 2 || argc > 3 || std::string(argv[1]) == "-h") {
    fprintf(stderr, "usage: %s METHOD [N=%ld]\n", argv[0], ndef);
    fprintf(stderr, "Monte-Carlo integration with N samples.\n\
METHOD:\n\
0: serial\n\
1: openmp, no arrays\n\
2: `omp parallel for reduction`, arrays without padding\n\
3: `omp parallel for reduction`, arrays with padding\n"
        );
    return 1;
  } 

  // method
  size_t m = atoi(argv[1]);
  // number of samples
	size_t n = (argc > 2 ? atoi(argv[2]) : ndef);
  // reference solution
  double ref = 3.14159265358979323846; 

	double wt0 = omp_get_wtime();
  double res = C(n, m);
	double wt1 = omp_get_wtime();

	printf("res:  %.20f\nref:  %.20f\nerror: %.20e\ntime: %.20f\n", 
      res, ref, res - ref, wt1 - wt0);

	return 0;
}
/**
 * RESULTS:
 * C0:
 * res:  3.14176444000000021362
	ref:  3.14159265358979311600
	error: 1.71786410207097617331e-04
	time: 8.8113232130000564978
 * C1:
 * res:  3.14172095999999978488
	ref:  3.14159265358979311600
	error: 1.28306410206668886076e-04
	time: 4.77398843799977257163
 * C2:
 * res:  3.14172095999999978488
	ref:  3.14159265358979311600
	error: 1.28306410206668886076e-04
	time: 4.79209297499955333421
 * C3:
 * res:  3.14166643999999983805
	ref:  3.14159265358979311600
	error: 7.37864102067220528625e-05
	time: 4.62176349299999156273
 * 
* **/
