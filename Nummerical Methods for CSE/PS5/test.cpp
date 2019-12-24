#include <iostream>
#include <Eigen/Sparse>



int main(){
	Eigen::Matrix<double, 4,4> A;
	Eigen::VectorXd x;
	x.fill(0);
	int n = x.size();

	A=Eigen::MatrixXd::Random(4,4);
	A(0,1)=1;
	std::cout << A << std::endl << n << std::endl;
	
	return 0;

}

