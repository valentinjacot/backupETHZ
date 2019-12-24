#include <Eigen/Sparse>
#include <iostream>
#include <vector>
#include "kron.hpp"


Eigen::SparseMatrix<double> buildC(const Eigen::MatrixXd &A){
	int n= A.cols();
	Eigen::SparseMatrix<double> C(n*n,n*n);
	Eigen::MatrixXd I= Eigen::MatrixXd::Identity(n,n);
	Eigen::MatrixXd C1,C2;
	kron(A,I,C1);kron(I,A,C2);
	C1+=C2;
	//std::cout << C1 << std::endl;
	std::vector <Eigen::Triplet<double> > triplets;
	for (int i=0; i<n*n; ++i){
		for (int j=0; j<n*n; ++j){
			if (C1(i,j)!=0){
				Eigen::Triplet<double> trpl(i,j,C1(i,j));
				triplets.push_back(trpl);
				}
		}
	}
	C.setFromTriplets(triplets.begin(),triplets.end());
	C.makeCompressed();
	return C;
	//TODO
};

void solveLyapunov(const Eigen::MatrixXd &A, Eigen::MatrixXd &X){
	int n= A.cols();
	Eigen::SparseMatrix<double> C;
	C= buildC(A);
	Eigen::MatrixXd I=Eigen::MatrixXd::Identity(n,n);
	Eigen::VectorXd b=Eigen::MatrixXd::Map(I.data(),n*n,1);
	Eigen::VectorXd x;
	Eigen::SparseLU<Eigen::SparseMatrix<double> > solver; solver.compute(C);
	x=solver.solve(b);
	X=Eigen::MatrixXd::Map(x.data(),n,n);
};


int main(){
	int n=5;
	Eigen::MatrixXd A(n,n),X(n,n);
    A<<10, 2, 3, 4, 5, 6, 20, 8, 9, 1, 1, 2, 30, 4, 5, 6, 7, 8, 20, 0, 1, 2, 3, 4, 10;
	std::cout << A << std::endl;
	///Teilaufgabe 1g
	/*Eigen::SparseMatrix<double> C;
	C= buildC(A);
	std::cout << C << std::endl;*/   
	solveLyapunov(A,X);
	std::cout << X << std::endl;
	
	///test
	/*Eigen::VectorXd y;
	y=Eigen::MatrixXd::Map(A.data(),4,1);
	std::cout << y << std::endl;*/
	return 0;	
}
