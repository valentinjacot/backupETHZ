#include <Eigen/Dense>
#include <iostream>
#include <vector>

template <class Matrix>
void kron(const Matrix & A, const Matrix & B, Matrix & C)
{	
	/*int n=A.cols();
	int m=A.rows();*/
	C= Matrix(A.rows()*B.rows(),A.cols()*B.cols());
	for (int i=0;i<A.rows();++i){
		for (int j=0;j<A.cols();++j){
			C.block(i*B.rows(),j*B.cols(),B.rows(),B.cols())=A(i,j)*B;
		}
	}
	// TODO
}
