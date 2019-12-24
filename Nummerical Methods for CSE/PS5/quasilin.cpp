#include <iostream>
#include <Eigen/Sparse>
#include <functional>


template <class func, class Vector>
void fixed_point_step(func&& A, const Vector & b, const
Vector & x, Vector & x_new){
	int  n= x.size();
	Eigen::SparseMatrix<double> A_eval(n,n) = A(x);
	x_new = x - ( A_eval + x*x.transposed()/x.norm()).cwiseInverse()*(A_eval*x-b);
};

int main(){
	Eigen::VectorXd x, b, x_new, x_old;
	b=Eigen::VectorXd::Random(10,10);
	x_new = b;
	double tol =2e-15;
	
	
	auto A = [ x_new ] (const Eigen::VectorXd & x) ->
	Eigen::SparseMatrix<double> & {
	int const n = x.size();
	Eigen::SparseMatrix<double, n, n> A;
	double x_norm = x.norm();
	x_norm +=3;
	A(0,0)=x_norm;A(0,1)=1;
	A(n-1,n-1)=x_norm;A(n-1,n-2)=1;
	for (int i= 1; i<n-1; i++){
		A(i,i)=x_norm;
		A(i+1,i)=1;
		A(i,i+1)=1;		
	}
	return A;
	};
	
	int j=0;
	while (std::abs(x_old.norm()-x.norm())<tol){
		i++
		fixed_point_step(A, b, x, x_new)
		x_old=x;
		x=x_new;
	}
	std::cout << x << std::endl;
	
	return 0;

}
