#include <Eigen/Dense>
#include <iostream>

using namespace std;
using namespace Eigen;

void rankoneinvit(const VectorXd & d, const double & tol, double & lmin) {
	MatrixXd M;
	VectorXd ev = d;
	lmin = 0.;
	double lnew = d.cwiseAbs().minCoeff();
	while (abs(lnew-lmin) > tol*lmin){
		lmin = lnew;
		M = d.asDiagonal(); M+=ev*ev.transpose();
		ev = M.lu().solve(ev);
		ev.normalize();
		lnew = ev.transpose()*M* ev;
		}
	lmin=lnew;
}


int main(){
    double tol=1e-3;
    double lmin;
    int n=10;
    
    VectorXd d=VectorXd::Random(n);
    rankoneinvit(d,tol,lmin);
    cout<<"lmin = "<<lmin<<endl;
    
    //Compare runtimes
    unsigned int repeats = 3;
    
    for(unsigned int p = 2; p <= 9; p++) {
        unsigned int n = pow(2,p);
        
        for(unsigned int r = 0; r < repeats; ++r) {
            
//             d = VectorXd::Random(n);
            d = VectorXd::LinSpaced(n,1,2);
            rankoneinvit(d,tol,lmin);
           
            }
        
    }
return 0;}



