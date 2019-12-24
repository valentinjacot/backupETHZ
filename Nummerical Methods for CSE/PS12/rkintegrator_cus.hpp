#pragma once

#include <vector>
#include <cassert>

#include <iostream>
#include <iomanip>

#include <Eigen/Dense>

//! \file rkintegrator.hpp Solution for Problem 1a, implementing RkIntegrator class

//! \brief Implements a Runge-Kutta explicit solver for a given Butcher tableau for autonomous ODEs
//! \tparam State a type representing the space in which the solution lies, e.g. R^d, represented by e.g. Eigen::VectorXd.
template <class State>
class RKIntegrator {
public:
    //! \brief Constructor for the RK method.
    //! Performs size checks and copies A and b into internal storage
    //! \param[in] A matrix containing coefficents of Butcher tableau, must be (strictly) lower triangular (no check)
    //! \param[in] b vector containing coefficients of lower part of Butcher tableau
    RKIntegrator(const Eigen::MatrixXd & A, const Eigen::VectorXd & b): A_(A),b_(b){
		n=A_.rows();
		assert(n==b_.size() && "size missmatch custom error");
		assert(A_.rows()==A_.cols() && "matrix not square custom error");
		
        // TODO: implement size checks and initialize internal data (DONE?)
    }
    
    //! \brief Perform the solution of the ODE
    //! Solve an autonomous ODE y' = f(y), y(0) = y0, using a RK scheme given in the Butcher tableau provided in the
    //! constructor. Performs N equidistant steps upto time T with initial data y0
    //! \tparam Function type for function implementing the rhs function. Must have State operator()(State x)
    //! \param[in] f function handle for rhs in y' = f(y), e.g. implemented using lambda funciton
    //! \param[in] T final time T
    //! \param[in] y0 initial data y(0) = y0 for y' = f(y)
    //! \param[in] N number of steps to perform. Step size is h = T / N. Steps are equidistant.
    //! \return vector containing all steps y^n (for each n) including initial and final value
    template <class Function>
    std::vector<State> solve(const Function &f, double T, const State & y0, unsigned int N) const {
		double h = T/N;
		std::vector<State> y_f;
		y_f.reserve(N);
		y_f.push_back(y0);
		
		State ytemp1 = y0;
        State ytemp2 = y0;
        // Pointers to swap previous value
        State * yold = &ytemp1;
        State * ynew = &ytemp2;
		
		for (int i =1; i<N;++i){
			step(f,h,*yold,*ynew);
			y_f.push_back(*ynew);
			std::swap(yold,ynew);
		}
		return y_f;
        // TODO: implement solver from 0 to T, calling function step appropriately
    }
    
private:
    
    //! \brief Perform a single step of the RK method for the solution of the autonomous ODE
    //! Compute a single explicit RK step y^{n+1} = y_n + \sum ... starting from value y0 and storing next value in y1
    //! \tparam Function type for function implementing the rhs. Must have State operator()(State x)
    //! \param[in] f function handle for ths f, s.t. y' = f(y)
    //! \param[in] h step size
    //! \param[in] y0 initial state 
    //! \param[out] y1 next step y^{n+1} = y^n + ...
    template <class Function>
    void step(const Function &f, double h, const State & y0, State & y1) const {
		y1 = y0;
		std::vector<State> k;
        k.reserve(n);
		for (int i=0;i<n;++i){
			State incr=y0;
			for (int j=0;j<i;++j){
				incr+=h*A_(i,j)*k.at(j);
			}
			k.push_back(f(incr));
			y1+=h*b_(i)*k.back();
		}
        // TODO: implement a single step of the RK method using provided Butcher scheme
    }
    
    //! TODO: put here suitable internal data storage
    const Eigen::VectorXd b_;
    const Eigen::MatrixXd A_;
    unsigned n;
};
