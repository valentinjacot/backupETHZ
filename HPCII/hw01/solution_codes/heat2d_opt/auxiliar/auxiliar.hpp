#ifndef _HEAT2D_H_
#define _HEAT2D_H_

class GridLevel;

class Heat2DSetup
{
	private:

  double* U;
  double* f;
  double* sol;

	public:

  bool saveOutput; // Write output to file flag
  bool checkSolution; // Verify solution

  // Timekeeping variables:
  double* smoothingTime;
  double* residualTime;
  double* restrictionTime;
  double* prolongTime;
  double* L2NormTime;
  double totalTime;

  int problemNumber; // Problem Number
  size_t N0;
  size_t N; // 2^N0 + 1 elements per side
  int iteration;

  int gridCount; // Number of multigrid levels to use
  int downRelaxations; // Number of Jacobi iterations before restriction
  int upRelaxations; // Number of Jacobi iterations after prolongation

  double L2Norm; // L2 Norm of the residual
  double L2NormPrev; // Previous L2 Norm
  double L2NormDiff; // L2Norm Difference compared to previous step
  double tolerance; // L2 Difference Tolerance before reaching convergence.

  // Setup and finish functions
  Heat2DSetup();
  void loadProblem();
  void setGridCount(int count);
  double getInitial(size_t x, size_t y) { return U[x*N + y]; }
  double getRHS(size_t x, size_t y) { return f[x*N + y]; }
  void saveSolution(size_t x, size_t y, double val) { U[x*N + y] = val; }

  // Timekeeping surrogates for the solver functions
  void applyJacobi_(GridLevel* g, int l, int relaxations);
  void applyProlongation_(GridLevel* g, int l);
  void applyRestriction_(GridLevel* g, int l);
  void calculateResidual_(GridLevel* g, int l);
  void calculateL2Norm_(GridLevel* g, int l);

  // Helper Functions
  void outputSolution();
  void verifySolution();
  void printHelp();
  void printGrid(double** g, int N);
};

void heat2DSolver(Heat2DSetup& s);

// Solver functions
void applyJacobi(GridLevel* g, int l, int relaxations);
void calculateResidual(GridLevel* g, int l);
void applyRestriction(GridLevel* g, int l);
void applyProlongation(GridLevel* g, int l);
double calculateL2Norm(GridLevel* g, int l);

#endif // _HEAT2D_H_
