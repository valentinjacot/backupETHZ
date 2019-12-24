// Include function "gauleg" for the computation of weights/nodes of Gauss quadrature
#include "gauleg.hpp"

#include <iomanip>
#include <iostream>

#include <cmath>

//! \brief Compute \int_a^b f(x) dx \approx \sum w_i f(x_i) (with scaling of w and x)
//! \tparam func template type for function handle f (e.g. lambda func.)
//! \param[in] a left boundary in [a,b]
//! \param[in] b right boundary in [a,b]
//! \param[in] f integrand
//! \param[in] Q Structure containing weights and nodes for a quadrature
//! \return Approximation of integral \int_a^b f(x) dx
template <class func>
double evalgaussquad(double a, double b, func&& f, const QuadRule & Q) {
    int n= Q.weights.size();
    double wf=0;
    for (int i=0; i<n; i++){
		double w=((b-a)/2.)*Q.weights[i];
		double x=a+((b-a)/2.)*(Q.nodes[i]+1);
		wf+=f(x)*w;
	}
	return wf;
    // TODO: implement gauss quadratur
}

//! \brief Compute double integral \int_\bigtriangleup f(x,b) dx dy using nested Gauss quadrature
//! \tparam func template type for function handle f (e.g. lambda func.), having operator (double x, double y) -> double
//! \param[in] f integrand, f(x,y) must be defined
//! \param[in] N number of quadrature points (in each direction)
//! \return Approximation of integral \int_\bigtriangleup f(x,b) dx dy
template <class func>
double gaussquadtriangle(func&& f, unsigned int N) {
    QuadRule Q=gauleg(N);
	auto f2=[&f,&Q](double y){return evalgaussquad(0,1-y,[&f,&y](double x){return f(x,y);},Q);};
	return evalgaussquad(0,1,f2,Q);
}

int main(void) {
    // Parameters
    double alpha = 1.5;
    double p = 0.1, q = 0.2;
    // Laser beam intensity
    auto I = [alpha, p, q] (double x, double y) { return std::exp(- alpha * ( (x-p)*(x-p) + (y-q)*(y-q) ) ); };
    
    // Max num of Gauss points to use (in each direction)
    unsigned int max_N = 13;
    // "Exact" integral
    double I_ex = 0.366046550000405;
    
    for(unsigned int N = 1; N < max_N; ++N) {
		double temp=gaussquadtriangle(I,N);
		std::cout << temp << std::setw(15) <<std::abs(temp-I_ex )<< std::endl;
		
    }
}
