#include <Eigen/Dense>
#include <iostream>
#include <Eigen/Sparse>
using namespace std;
using namespace Eigen;

template <class scalar>
struct TripletMatrix;


vector <Triplet<double>> triplet;
triplet=
SparseMatrix<double, row major> spMat(size_t n,size_t m);
