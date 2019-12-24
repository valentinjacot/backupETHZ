#include "order.hpp"



int main(void) {
	auto f1=[] (const Eigen::VectorXd & y){
		Eigen::VectorXd temp =y;
		temp*=(1.-y(0));
		return temp;
	};
	auto f2=[] (const Eigen::VectorXd & y){
		Eigen::VectorXd temp(1);temp << std::abs(1.1 - y(0)) +1.;
		return temp;
	};
	
	Eigen::VectorXd y1(1);
	y1<<1./2.;
	Eigen::VectorXd y2(1);
	y2<<0;
	// Dimension of state space
    unsigned int d = 1;
    
    // Final time for model
    double T = 0.1;
    
    unsigned int s = 1;
    Eigen::MatrixXd A(s,s);
    Eigen::VectorXd b(s);
    A<<0;
    b<<1;

    ///------------------------------------------------
    s=2;
    Eigen::MatrixXd A2(s,s);
    Eigen::VectorXd b2(s);
    A2<< 0,	0,
		1.,	0;
	b2<< 1./2., 1./2.;


    ///------------------------------------------------
	s=3;
    Eigen::MatrixXd A3(s,s);
    Eigen::VectorXd b3(s);
    A3 << 0,      0,      0,
         1./2.,  0,      0,
         -1.,      2.,  0;
    b3 << 1./6.,  2./3.,      1./6.;
	///------------------------------------------------
    s=4;
    Eigen::MatrixXd A4(s,s);
    Eigen::VectorXd b4(s);
    A4 << 0,      0,      0,		0,
         1./2.,  0,      0,		0,
         0,      1./2.,	0,		0,
         0,		0,		1.,		0;
    b4 << 1./6.,  2./6.,   2./6.,	1./6.;
    
	
    std::cout << "Method of order 1" << std::endl;
   	errors(f1,T,y1,A,b);
	
	std::cout << "Method of order 2" << std::endl;
   	errors(f1,T,y1,A2,b2);
   
	std::cout << "Method of order 3" << std::endl;
	errors(f1,T,y1,A3,b3);

	std::cout << "Method of order 4" << std::endl;
	errors(f1,T,y1,A4,b4);
	
std::cout << " -------------------------------------- " << std::endl;
   	std::cout << "second function" << std::endl;
   	
	std::cout << "Method of order 1" << std::endl;
   	errors(f2,T,y2,A,b);
	
	std::cout << "Method of order 2" << std::endl;
	errors(f2,T,y2,A2,b2);    

	std::cout << "Method of order 3" << std::endl;
    errors(f2,T,y2,A3,b3);
    
   	std::cout << "Method of order 4" << std::endl;
	errors(f2,T,y2,A4,b4);

    
    return 0;
}
