#include <iostream>
#include <Eigen/Dense>
#include <cmath>

Eigen::VectorXd poly(int const n, double & x){
	Eigen::VectorXd v(n);
	v(0)=1;
	if(n>1) v(1)=x;
	for (int i=2; i < n;i++){
		v(i)=2*x*v(i-1)-v(i-2);
	}
	return v;
}

Eigen::VectorXd zeroes(const int n){
	Eigen::VectorXd v(n);
	for (int i=0;i<n;i++){
		double temp=((2.*i-1.)/n*2)*M_PI;
		v(i)=cos(temp);
	}
	return v;
}

template <typename Function>
void bestpolchebnodes (const Function&f, Eigen::VectorXd &alpha){
		size_t n= alpha.size();
		Eigen::VectorXd v;
		Eigen::VectorXd fn(n+1);
		for (int i=0; i<n+1; i++){
			double temp=cos(M_PI*(2*k+1)/2/(n+1));
			fn(i)=f(temp);
		}
		Eigen::MatrixXd scal(n+1,n+1);
		for (int j=0; j<n+1; j++) {
			v=poly(n,cos(M_PI*(2*j+1)/2/(n+1)));
			for (int k=0; k<n+1; k++) scal(j,k)=V[k];
		}
		for (int l=0;l<n;l++){
			alpha(l)=0;
			for (int k=0;k<n;k++){
				alpha(l)+=2*(1./(n+1))*fn(k)*v(k);
			}
		}
		alpha(0)=alpha(0)/2;

}
using namespace std;
int main(){
    auto f = [] (double & x) {return 1/(pow(5*x,2)+1);};
    int n=20;
    Eigen::VectorXd alpha(n+1);
    bestpolchebnodes(f, alpha);
    
    //Compute the error
    Eigen::VectorXd X = Eigen::VectorXd::LinSpaced(1e6,-1,1);
    auto qn = [&alpha,&n] (double & x) {
        double temp;
        vector<double> V=poly(n,x);
        for (int k=0; k<n+1; k++) temp+=alpha(k)*V[k];
        return temp;
    };
    double err_max=abs(f(X(0))-qn(X(0)));
    for (int i=1; i<1e6; i++) err_max=std::max(err_max,abs(f(X(i))-qn(X(i))));
    cout<<"Error: "<< err_max <<endl;
}
