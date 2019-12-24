//test for PS12 problem 1
#include <Eigen/Dense>
#include <iostream>



int main(){
	
	
    unsigned int s = 3;
	Eigen::VectorXd c_(s);
    Eigen::MatrixXd A_(s,s);
    Eigen::VectorXd b_(s);
    A_ << 0,      0,      0,
         1./3.,  0,      0,
         0,      2./3.,  0;
    b_ << 1./4.,  0,      3./4.;
    
    short n=A_.rows();
	assert(n==b_.size());
	c_.resize(n);
	std::cout << n;
	for (int i=0; i<n;i++){
		c_(i)=0;
		for (int j=0;j<i+1;j++){
				c_(i)+=A_(i,j);
			}
		}
	std::cout << A_ << std::endl;
	std::cout << c_ << std::endl;
	
	
	
	
	
}
