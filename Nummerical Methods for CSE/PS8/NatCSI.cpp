#include <Eigen/Dense> 
#include <Eigen/Sparse> 
#include <iostream>
#include <cassert>
#include <vector>

class NatCSI{
public:
	NatCSI(const std::vector<double> & t, const std::vector<double> & y): 
	t_(t),
	y_(y), 
	h(t.size()-1),
	c(t.size()) {
		assert ((t_.size() == y_.size()) && "size missmatch (t & y)");
		m=t.size();//n+1
		for (int i = 0; i<(m-1); i++){
			h(i)=t_[i+1]-t_[i];
			assert( ( h(i) > 0 ) && "Error: array t must be sorted!");
		}
		
		Eigen::SparseMatrix<double> A(m,m);
        Eigen::VectorXd b(m);

        A.reserve(3);
		
		A.coeffRef(0,0)=2/h(0);
		A.coeffRef(0,1)=1/h(0);
		A.coeffRef(m-1,m-1)=2/h(m-2);
		A.coeffRef(m-1,m-2)=1/h(m-2);
		double bold =(y[1]-y[0])/(h(0)*h(0));
		b(0)= 3*bold;
		
		for (int i=1;i<m; ++i){
			//A
			A.coeffRef(i,i-1)=1/h(i);
			A.coeffRef(i,i)=2/h(i)+2/h(i-1);
			A.coeffRef(i,i+1)=1/h(i);/// not exactly matching with the theory!
			//b
			double bnew =(y[i+1]-y[i])/(h(i)*h(i));
			b(i)= 3*(bnew+bold);
			bold=bnew;
			
		}
		b(m-1) =3*bold;
		A.makeCompressed();
		std::cout << A << std::endl;
		
		Eigen::SparseLU<Eigen::SparseMatrix<double> > lu;
		lu.compute(A);
		c = lu.solve(b);
		
		
	}
	
	double operator () (double x) const {
		assert( (x>=t_[0] && x<=t_[m-1]) && "Error: x is not in [t(0),t(n)]");
		
		auto j = ((std::lower_bound(t_.begin(), t_.end(), x)-t_.begin())-1);
		if (j==-1) j++;
		
		double tau = x - t_[j]/h(j);
		double tau2 = tau*tau;
		double tau3 = tau2*tau;
		return y_[j]  *  (1.-3.*tau2 + 2* tau3) +
				y_[j+1] *(3*tau2-2*tau3) + 
				h(j)*c(j) *(tau-2*tau2+tau3) +
				h(j)*c(j+1)*(-tau2+tau3);	
	};
	
private:
	std::vector<double> t_, y_;
	Eigen::VectorXd h, c;
	size_t m; //m=n+1
	
};


int main() {
    
    int n = 8, m = 100;
    std::vector<double> t;
    std::vector<double> y;
    t.resize(n);
    y.resize(n);
    for(int i = 0; i < n; ++i) {
        t[i] = (double) i / (n-1);
        y[i] = cos(t[i]);
    }
    
    NatCSI N(t,y);
    
    Eigen::VectorXd x = Eigen::VectorXd::LinSpaced(m, 0, 1);
    for(int i = 0; i < x.size(); ++i) {
        x(i) = N(x(i));
    }
    std::cout << x << std::endl;
}


