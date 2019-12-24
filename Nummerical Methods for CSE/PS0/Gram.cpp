#include <Eigen/Dense>
#include <iostream>
#include <cassert>


template <class matrix>
matrix gramschmidt(const matrix &A){
	unsigned n=A.cols();
	matrix Q=A;
	Q.col(0).normalize();

	for(int i=1;i<n;i++){
		Q.col(i)-= Q.leftCols(i)*(Q.leftCols(i).transpose() * A.col(i));
	//assert(Q.col(i).norm()<10e-16*A.col(i).norm() && "can't compute because A has lin dependant rows");
	Q.col(i).normalize();
	}
	return Q;
	
};

int main() {
	int n=5;
	Eigen::MatrixXd A;
	Eigen::MatrixXd Q;
	A = Eigen::MatrixXd::Random(n,n);
	Q=gramschmidt(A);
	std::cout << A<< std::endl;
	Q*=Q.transpose();
	/*for(int i=0;i<n;i++){
		for(int j=0;j<n;j++){
			if (abs(Q(i,j))<10e-10)Q(i,j)=0;}}
	
*/	
	std::cout << Q << std::endl;
	return 0;
}
