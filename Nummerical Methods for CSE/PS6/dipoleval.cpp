#include <iostream>
#include <Eigen/Dense>

Eigen::VectorXd dipoleval(const Eigen::VectorXd & x, const Eigen::VectorXd & t, const Eigen::VectorXd & y){
	int x_size= x.size();
	int w=1;
	int n= t.size();
	Eigen::VectorXd p=y;
	Eigen::VectorXd dP=Eigen::MatrixXd::Zero(1,n);
	for (int i=1; i<n; i++){//im
		for (int j=i-1; j>0;j--){//i0
			//% compute dp(i)'s
            dP(j) = p(j+1) + (x(w)-t(j))*dP(j+1) - p(j) - (x(w)-t(i))*dP(j);
            dP(j) = dP(j) / ( t(i) - t(j) );
            //% compute p(i)'s
            p(j)  = (x(w)-t(j)*p(j+1) - (x(w)-t(i))*p(j));
            p(j)  = p(j) / ( t(i) - t(j) );
            }
			std::cout <<"test\n";

	}
	return p.row(0);
} 

int main(){
	int n=6;
	Eigen::VectorXd x=Eigen::MatrixXd::Random(n,1);
	Eigen::VectorXd y=Eigen::MatrixXd::Random(n,1);
	Eigen::VectorXd t=Eigen::MatrixXd::Random(n,1);
	
	Eigen::VectorXd v;
	v=dipoleval(x,t,y);
	std::cout << v << std::endl;
}

