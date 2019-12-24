#include "rkintegrator.hpp"
#include "order.hpp"

//! \file rk3prey.cpp Solution for Problem 1b, solving prey/predator model with 3-step RK method

int main(void) {
    
    // Implementation of butcher scheme
    unsigned int s = 3;
    Eigen::MatrixXd A(s,s);
    Eigen::VectorXd b(s);
    A << 0,      0,      0,
         1./3.,  0,      0,
         0,      2./3.,  0;
    b << 1./4.,  0,      3./4.;
    
    // Dimension of state space
    unsigned int d = 2;
    
    // Final time for model
    double T = 10.;
    
    // Initial value for model
    Eigen::VectorXd y0(d);
    y0 << 100, 5;
    
    
    // Array of number of steps (for convergence study)
    std::vector<unsigned int> N = {128, 256, 512, 1024, 2048, 4096, 8192, 16384};
    
    // Exact value y(10) at final time T = 10 (approximated)
    Eigen::VectorXd yex(d);
    yex << 0.319465882659820, 9.730809352326228;
    
    // Coefficients and handle for prey/predator model
    double alpha1 = 3.;
    double alpha2 = 2.;
    double beta1 = 0.1;
    double beta2 = 0.1;
    // TODO: implement functor f for rhs of y(t)' = f(y(t))
    auto f=[&alpha1, &alpha2, &beta1, &beta2] (const Eigen::VectorXd & y){
			auto temp =y;
			temp(0)*=(alpha1-beta1*y(1));
			temp(1)*=(beta2*y(0)-alpha2);
			return temp;
		};
    
    // Initialize RK with Butcher scheme
    RKIntegrator<Eigen::VectorXd> RK(A,b);
    
    // Start convergence study
    std::cout  << std::setw(15) << "N"  << std::setw(15) << "error" << std::setw(15) << "rate" << std::endl;
    // TODO: tabulate error for each N
    double errold=0;
    for(unsigned int i = 0; i < N.size(); ++i){
		auto res = RK.solve(f,T,y0,N[i]);
		double errnew = (yex-res.back()).norm();
		
	    std::cout  << std::setw(15) << N[i] << std::setw(15) << errnew<< std::setw(15) << log2(errold/errnew) << std::endl;
		errold=errnew;
	}
	std::cout<< "----------------------------------------------"<< std::endl;
	errors(f,T,y0,A,b);
		
}



