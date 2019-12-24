#include <iostream>
#include <Eingen/Dense>



int main(){
	//Eigen::SparseMatrix<double, 4,4> A;

	A=Matrix4d::Random(4,4);
	A(0,1)=1;
	std::cout << A << std::endl;
	
	return 0;

}

