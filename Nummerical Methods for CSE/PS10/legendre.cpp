#include <Eigen/Dense>
#include <iostream>


using namespace std;
using namespace Eigen;

void legvals(const VectorXd &x, MatrixXd &Lx, MatrixXd &DLx){
    int n = Lx.cols()-1;
    int N = x.size();
	for (int i=0;i<N;i++){
		Lx(i,0)=1.;
		Lx(i,1)=x(i);
		DLx(i,0)=0;
		DLx(i,1)=1.;
		for (int j=2;j<n+1;j++){
			Lx(i,j)=((2*j-1.)/j)*x(i)*Lx(i,j-1)-((j-1.)/j)*Lx(i,j-2);
			DLx(i,j)=((2*j-1.)/j)*(Lx(i,j)+x(i)*DLx(i,j-1))-((j-1.)/j)*DLx(i,j-2);
		}

	}

}

double Pnx (double x, int n){
	if (n==0){return 1.;}
	else if(n==1){return x;}
	else{
		return (((2*n-1.)/n)*x*Pnx(x,n-1)-((n-1.)/n)*Pnx(x,n-2));
	}
}

// Find the Gauss points using the secant method with regula falsi. The standard secant method may be obtained by commenting out lines 50 and 52.
MatrixXd gaussPts(int n, double rtol=1e-10, double atol=1e-12) {
	MatrixXd Pkn(n,n);
	double f0, fn,x0,x1,s;
	for (int k=1;k<n+1;k++){// k de 1-n
		for (int j=1;j<k+1;j++){ // j de 1 - k

			if (j==1) x0 = -1.;
            else      x0 = Pkn(j-2,k-2);
            if (j==k) x1 = 1.;
            else      x1 = Pkn(j-1,k-2);

			f0 = Pnx(x0,k);
			for (int i=0;i<1e4;i++){
				fn=Pnx(x1,k);
				s=fn*(x1-x0)/(fn-f0);
				if (Pnx(x1 - s,k)*fn<0) { x0 = x1; f0 = fn;} // without this correstion, the zero KI8-6 is not a zero, because the initial guess is to far away for the sought zero ---> not the case of a local convergence (I got 0.273438)
				// x0 = x1; f0 = fn;
                x1=x1-s;
				if (abs(s)<max(atol,rtol*min(abs(x0),abs(x1)))){Pkn(j-1,k-1)=x1;break;}
			}
		}

}
return Pkn;

}
// Test the implementation.
int main(){
    int n = 8;
    MatrixXd zeros = gaussPts(n);
    cout<<"Zeros: "<<endl<< zeros <<endl;

    for (int k=1; k<n+1; k++) {
        VectorXd xi = zeros.block(0, k-1, k, 1);
        MatrixXd Lx(k,n+1), DLx(k,n+1);
        legvals(xi, Lx, DLx);
        cout<<"Values of the "<<k<<"-th polynomial in the calculated zeros: "<<endl;
        cout<<Lx.col(k).transpose() <<endl;
    }
}
