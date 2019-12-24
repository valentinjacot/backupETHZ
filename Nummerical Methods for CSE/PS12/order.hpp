#include "rkintegrator.hpp"
#include <vector>
#include <cassert>

#include <iostream>
#include <iomanip>

#include <Eigen/Dense>

template <class Function>
void errors(const Function &f, const double &T, const Eigen::VectorXd &y0, 
	const Eigen::MatrixXd &A, const Eigen::VectorXd &b){
		std::vector<unsigned int> N = {1,2,4,8,16,32,64,128, 256, 512, 1024, 2048, 4096, 8192,16384,32768};
		unsigned int N_f=32768;
		double sum=0;
		double count =0;
		std::vector<double> err_vect;
		err_vect.push_back(1.);
		RKIntegrator<Eigen::VectorXd> RK(A,b);
		auto y_f=RK.solve(f,T,y0,N_f);
		for(unsigned int i = 0; i < N.size(); ++i){
			auto temp = RK.solve(f,T,y0,N[i]);
			double err= (temp.back()-y_f.back()).norm();
			double order = log2(err_vect.back()/err);
			order= std::abs(order);
			std::cout << "n = " << N[i] << std::setw(15)  << "error = " << err << std::setw(15)<< "order is : " << order<< std::endl;
			err_vect.push_back(err);
			if(i>1 && err>1e-16){
				sum+=order; count+=1;
			}
		}

std::cout << " average order : " <<sum/count  << std::endl;

};
