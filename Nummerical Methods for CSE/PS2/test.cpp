#include <Eigen/Dense>
#include <iostream>


int main(){
	Eigen::MatrixXd X(2,2);
	X << 2,1,-1, 3;
	std::cout << X << std::endl;
	Eigen::VectorXd y;
	y=Eigen::MatrixXd::Map(X.data(),4,1);
	std::cout << y << std::endl;
	return 0;	
}
