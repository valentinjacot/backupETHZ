#include <utility>
#include <iostream>
#include <Vector>
#include <cmath>
#include <iomanip>
#include <Eigen/Dense>
// pas terminé, 

template <typename arg, class func, class jac>
void mod_newt_step_system(const arg & x, arg & x_next, function&& f, jacobian&& df){
	arg y;
	y=x+f(x)/df(x);
	x_next = y - f(y)/df(x);
}

void mod_newt_step_exec(){
	using Vector = Eigen::VectorXd;
	using Matrix = Eingen::MatrixXd;
	
	Vector x;
	Vector c = Vector::random(4);
	Matrix A = Matrix::random(4,4);
	A = (A*A.transpose());
	c = c.cwiseAbs(); 
	
	auto F = [&A,&c] (const Vector & x) 
	{ Vector tmp =  A*x + c.cwiseProduct(x.array().exp().Matrix()).eval(); return tmp; };
    std::function<Matrix(const Vector &)> dF = [&A, &c] (const Vector & x) 
    { Matrix C = A; Vector temp = c.cwiseProduct(x.array().exp().Matrix()); C += temp.asDiagonal(); return C; };

	double tol = 1.e-15;
	
	int i=0;

	while (tol){
		i++;
		mod_newt_step_system(x, x_next,f,df );
		std::cout << i << "th interation,    x_next is " << x_next << "        and x_star - x_next : " << eval_err(x_next) << std::endl;
		if (i> 15){
			std::cout << " Broke " << std::endl;
			break;}
			x= x_next;
	}

}












int main () {
	mod_newt_step_exec();
	const double a = 0.123;
	std::cout << "tan(a) : " << std::tan(a) << std::endl;
	return 0;
	
}
