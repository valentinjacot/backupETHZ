#include <Eigen/Dense>

#include <iostream>
#include <iomanip>

#include <cmath>

//! \brief Solve the autonomous IVP y' = f(y), y(0) = y0 using Rosenbrock method
//! Use semi-implicit Rosenbrock method using Jacobian evaluation. Equidistant steps of size T/N.
//! \tparam Func function type for r.h.s. f
//! \tparam DFunc function type for Jacobian df
//! \tparam StateType type of solution space y and initial data y0
//! \param[in] f r.h.s. func f
//! \param[in] df Jacobian df of f
//! \param[in] y0 initial data y(0)
//! \param[in] N number of equidistant steps
//! \param[in] T final time
//! \return vector of y_k for each step k from 0 to N
template <class Func, class DFunc, class StateType>
std::vector<StateType> solveRosenbrock(const Func & f, const DFunc & df, const StateType & y0, unsigned int N, double T) {
    const double h = T/N;
    const double a = 1. / (std::sqrt(2) + 2.);
    std::vector<StateType> y1;
    y1.push_back(y0);
    Eigen::MatrixXd W;
    Eigen::VectorXd k1,k2;
    for (unsigned i=1;i<N+1; ++i){
		StateType yprev=y1.at(i-1);
		W=Eigen::MatrixXd::Identity(2,2)-a*h*df(yprev);
		k1=W.partialPivLu().solve(f(yprev));
		k2=W.partialPivLu().solve(f(yprev+0.5*h*k1)-a*h*df(yprev)*k1);
		y1.push_back(yprev+h*k2);
	}
    return y1;
    // TODO: implement rosenbrock method
}


int main() {
    // Final time
    const double T = 10;
    // All mesh sizes
    const std::vector<int> N = {16, 32, 64, 128, 256, 512, 1024, 2048, 4096};
    // Reference mesh size
    const int N_ref = 16384;
    // Initial data
    Eigen::VectorXd y0;
    y0 << 1., 1.;
	double lambda =1.;
	Eigen::MatrixXd R;
    R << 0., -1., 1., 0.;
    // Function and his Jacobian
    // TODO: implement r.h.s. and Jacobian
    auto f =  [&R, &lambda] (const Eigen::Vector2d & y) { return R*y + lambda*(1. - std::pow(y.norm(),2))*y; };
	auto df = [&lambda] (const Eigen::VectorXd & y) {
        double x = 1 - std::pow(y.norm(), 2);
        Eigen::MatrixXd J(2,2);
        J << lambda*x - 2*lambda*y(0)*y(0),
        -1 - 2*lambda*y(1)*y(0),
        1 - 2*lambda*y(1)*y(0),
        lambda*x - 2*lambda*y(1)*y(1);
        return J;
    };    
    auto sol_ex = solveRosenbrock(f,df,y0,N_ref,T);
    double sol_exact= sol_ex.back().norm();
    double solprev=0.;
    std::cout << "N:" << std::setw(15) << "error" << std::setw(15) << "conv:" <<std::endl;
    for (unsigned int j=0;j<N.size(); j++){
		auto sol= solveRosenbrock(f,df,y0,N[j],T);
		auto sol1= sol.back().norm();	
        std::cout <<N[j] << std::setw(15) << sol1-sol_exact;
        if(j>0){std::cout << log2(solprev/sol1);}
        std::cout << std::endl;
        solprev= sol1;   
    // TODO: compute reference solutions, solution, error and output approximate order of convergence
    }
}
                                    
