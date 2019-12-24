#include <iostream>
#include <Eigen/Dense>
#include "timer.h"

void rankoneinvit(const Eigen::VectorXd & d, const double & tol, double & lmin){
	lmin=0.;
	Eigen::VectorXd ev;
	int n=d.size();
	Eigen::MatrixXd M;
	ev=d;
	double lnew = d.cwiseAbs().minCoeff();
	while(std::abs(lnew-lmin)> tol*lmin){
		lmin=lnew;
		M=d.asDiagonal();
		M+=ev*ev.transpose();
		ev= M.partialPivLu().solve(ev);
		ev.normalize();
		lnew= ev.transpose()*M*ev;
	}
	lmin=lnew;
} 


using namespace std;
using namespace Eigen;
int main(){
    srand((unsigned int) time(0));
    double tol=1e-3;
    double lmin;
    int n=10;
    
    // Check correctedness of the fast version
    VectorXd d=VectorXd::Random(n);
    rankoneinvit(d,tol,lmin);
    cout<<"lmin = "<<lmin<<endl;
    //rankoneinvit_fast(d,tol,lmin);
    //cout<<"lmin = "<<lmin<<endl;
    
    //Compare runtimes
    unsigned int repeats = 3;
    /*
    for(unsigned int p = 2; p <= 9; p++) {
        tm_slow.reset();
        tm_fast.reset();
        unsigned int n = pow(2,p);
        
        for(unsigned int r = 0; r < repeats; ++r) {
            
//             d = VectorXd::Random(n);
            d = VectorXd::LinSpaced(n,1,2);
            rankoneinvit(d,tol,lmin);
            
            rankoneinvit_fast(d,tol,lmin);
        }
        
        cout << "The slow method took:    " <<  tm_slow.min().count() / 1000000. << " ms for n = " <<n<<endl;
        cout << "The fast method took:    " <<  tm_fast.min().count() / 1000000. << " ms for n = " <<n<<endl;
    }*/
}
