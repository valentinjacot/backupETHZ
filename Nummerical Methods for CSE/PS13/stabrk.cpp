#include "rkintegrator.hpp"
#include <cmath>
//! \file stabrk.cpp Solution for Problem 2, PS13, solving prey/predator model with RK-SSM method

int main(void) {
    
    // Implementation of butcher scheme
    unsigned int d=2;
    unsigned int T=1;
    unsigned int s = 3;
    Eigen::MatrixXd A(s,s);
    Eigen::VectorXd b(s);
    A << 0,      0,      0,
         1.,     0,      0,
         1./4.,  1./4.,  0;
    b << 1./6.,  1./6.,  2./3.;
    // Initial value for model
    Eigen::VectorXd y0(d);
    y0 << 100, 5;
    
    RKIntegrator<Eigen::MatrixXd> rk(A,b);
    
    // Coefficients and handle for prey/predator model
    double alpha1 = 1.;
    double alpha2 = 1.;
    double beta1 = 1.;
    double beta2 = 1.;
    // TODO: implement functor f for rhs of y(t)' = f(y(t))
    auto f=[&alpha1, &alpha2, &beta1, &beta2] (const Eigen::VectorXd & y){
			auto temp =y;
			temp(0)*=(alpha1-beta1*y(1));
			temp(1)*=(beta2*y(0)-alpha2);
			return temp;
		};
    double error_old=0;
    Eigen::VectorXd sol(d);
    Eigen::VectorXd sol_exact(d);
    unsigned N0= pow(2,14);
    sol_exact=rk.solve(f, T,y0,N0).back();
    std::cout << " stepsize 1./ : \t" << std::setw(10) << "error : " << std::setw(10) << "convg : " << std::endl;
    for (int j=2;j<14;j++){
		unsigned N = pow(2, j); 
		sol = rk.solve(f,T,y0,N).back();
		double error_new = (sol-sol_exact).norm();
		double convergence= log2(error_old/error_new);
		std::cout << "\t\t"<<N <<"\t"<< std::setw(10) << error_new << std::setw(10) <<convergence << std::endl;
		error_old=error_new;
		}
    // TODO: solve IVP of Problem 2 and plot error vs. num. of steps (use uniform timestepping)
}
