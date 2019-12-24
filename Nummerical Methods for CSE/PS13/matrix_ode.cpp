#include "ode45.hpp"

#include <iostream>
#include <iomanip>

#include <Eigen/Dense>
#include <Eigen/QR>

//! \file stabrk.cpp Solution for Problem 1, PS13, involving ode45 and matrix ODEs

//! \brief Solve matrix IVP Y' = -(Y-Y')*Y using ode45 up to time T
//! \param[in] Y0 Initial data Y(0) (as matrix)
//! \param[in] T final time of simulation
//! \return Matrix of solution of IVP at t = T
Eigen::MatrixXd matode(const Eigen::MatrixXd & Y0, double T) {
	auto f=[](const Eigen::MatrixXd & Y){return (-(Y-Y.transpose())*Y);};
	ode45<Eigen::MatrixXd> O(f);
    O.options.rtol = 10e-8;
    O.options.atol = 10e-10;
    return O.solve(Y0,T).back().first;
    // TODO: evolve Y0 up to T using ode45 (read ode45.hpp)
}


//! \brief Find if invariant is preserved after evolution with matode
//! \param[in] Y0 Initial data Y(0) (as matrix)
//! \param[in] T final time of simulation
//! \return true if invariant was preserved (up to round-off), i.e. if norm was less than 10*eps
bool checkinvariant(const Eigen::MatrixXd & M, double T) {
	return (matode(M,0.1).norm()==matode(M,T).norm());
		// TODO: check if invariant is preserved applying matode
}

//! \brief Implement ONE step of explicit Euler applied to Y0, of ODE Y' = A*Y
//! \param[in] A matrix A of the ODE
//! \param[in] Y0 Initial state
//! \param[in] h step size
//! \return next step
Eigen::MatrixXd expeulstep(const Eigen::MatrixXd & A, const Eigen::MatrixXd & Y0, double h) {
    return Y0+h*A*Y0;
    // TODO: ose step of EE
}

//! \brief Implement ONE step of implicit Euler applied to Y0, of ODE Y' = A*Y
//! \param[in] A matrix A of the ODE
//! \param[in] Y0 Initial state
//! \param[in] h step size
//! \return next step
Eigen::MatrixXd impeulstep(const Eigen::MatrixXd & A, const Eigen::MatrixXd & Y0, double h) {
    size_t n = Y0.cols();
    Eigen::MatrixXd Y1(n,n);
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(n,n);    
    return (I-h*A).partialPivLu().inverse()*Y0;
    // TODO: ose step of IE
}

//! \brief Implement ONE step of implicit midpoint ruler applied to Y0, of ODE Y' = A*Y
//! \param[in] A matrix A of the ODE
//! \param[in] Y0 Initial state
//! \param[in] h step size
//! \return next step
Eigen::MatrixXd impstep(const Eigen::MatrixXd & A, const Eigen::MatrixXd & Y0, double h) {
    size_t n = Y0.cols();
    Eigen::MatrixXd Y1(n,n);
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(n,n);    
    return (I-h*0.5*A).partialPivLu().inverse()*(I+h*0.5*A)*Y0;    
    // TODO: ose step of IMP
}

int main() {
    
    double T = 1;
    unsigned int n = 3;
    
    Eigen::MatrixXd M(n,n);
    M << 8,1,6,3,5,7,4,9,2;
    
    std::cout << "SUBTASK 1. c)" << std::endl;
    // Test preservation of orthogonality
    
    // Build Q
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(M.rows(), M.cols());
    qr.compute(M);
    Eigen::MatrixXd Q = qr.householderQ();
    
    // Build A
    Eigen::MatrixXd A(n,n);
    A << 0, 1, 1, -1, 0, 1, -1, -1, 0;
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(n,n);
    
    // TODO: compute norm of Y'Y-I for 20 steps and print table
    Eigen::MatrixXd EE(n,n);
    EE=Q;
    Eigen::MatrixXd IE(n,n);
    IE=Q;
    Eigen::MatrixXd IM(n,n);
	IM=Q;
	double EE_norm,IE_norm,IM_norm;
    std::cout << "\tStep" << std::setw(15) << "expl. Euler" << std::setw(15) << "impl. Euler" << std::setw(15) << "Impl.  Midpoint" << std::endl;
    for (int i=0; i< 21; i++){
		double h=0.01;
		EE=expeulstep(A, EE, h);		
		IE=impeulstep(A, IE, h);		
		IM=impstep(A, IM, h);		
		EE_norm= (EE.transpose()*EE-I).norm();
		IE_norm= (IE.transpose()*IE-I).norm();
		IM_norm= (IM.transpose()*IM-I).norm();
		std::cout << "\t"<< i << std::setw(15) << EE_norm << std::setw(15) << IE_norm << std::setw(15) << IM_norm << std::endl;
}
    
    
    std::cout << "SUBTASK 1. d)" << std::endl;
    // Test implementation of ode45
    Eigen::MatrixXd mat_ode =matode(M,T);
    std::cout << mat_ode <<std::endl;
    std::cout << "test matode norm : "  << (mat_ode.transpose()*mat_ode-I).norm() << std::endl;
    // TODO: TEST matode
    
    std::cout << "SUBTASK 1. g)" << std::endl;
    // Test whether invariant was preserved or not

    if (checkinvariant(M,T)){std::cout << " Invariant preserved" << std::endl;}
    else {std::cout << " Invariant not preserved" << std::endl;}
    // TODO: TEST if matode preserves invariant using checkinvariant
    
    
    return 0;
}
