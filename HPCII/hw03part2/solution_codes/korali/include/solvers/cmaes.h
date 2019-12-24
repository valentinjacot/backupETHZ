#ifndef _KORALI_CMAES_H_
#define _KORALI_CMAES_H_

#include "solvers/base.h"
#include "parameters/gaussian.h"
#include <map>

namespace Korali::Solver
{

class CMAESParameter
{
 public:
	 double _initialValue;
	 double _initialStdDev;
	 double _minStdDevChange;

	 CMAESParameter();
	 void setInitialValue(double initialX) { _initialValue = initialX; }
	 void setInitialStdDev(double initialStdDev) { _initialStdDev = initialStdDev; }
	 void setMinStdDevChange(double minStdDevChange) { _minStdDevChange = minStdDevChange; }
};

class CMAES : public Korali::Solver::Base
{
  public:

  // Public Methods
	CMAES(Korali::Problem::Base* problem);

  // Runtime Methods (to be inherited from base class in the future)
  void initializeInternalVariables();
  void prepareGeneration();
  bool checkTermination();
  void printResults();
  void updateDistribution(const double *fitnessVector);
  void runSolver();
  void processSample(size_t sampleId, double fitness);

	// Configuration Methods
  void setMaxFitnessEvaluations(size_t maxFitnessEvaluations) { _maxFitnessEvaluations = maxFitnessEvaluations; }
	void setDiagonalCovarianceMatrixEvalFrequency(size_t diagonalCovarianceMatrixEvalFrequency) { _diagonalCovarianceMatrixEvalFrequency = diagonalCovarianceMatrixEvalFrequency; }
	void setCovarianceEigensystemEvaluationFrequency(size_t covarianceEigensystemEvaluationFrequency) { _covarianceEigensystemEvaluationFrequency = covarianceEigensystemEvaluationFrequency; }
	void setStopFitnessEvalThreshold(double stopFitnessEvalThreshold) { _stopFitnessEvalThreshold = stopFitnessEvalThreshold; }
	void setStopFitnessDiffThreshold(double stopFitnessDiffThreshold) { _stopFitnessDiffThreshold = stopFitnessDiffThreshold; }
	void setStopFitnessDiffHistoryThreshold(double stopFitnessDiffHistoryThreshold) { _stopFitnessDiffHistoryThreshold = stopFitnessDiffHistoryThreshold; }
	void setStopMinDeltaX(double stopMinDeltaX) { _stopMinDeltaX = stopMinDeltaX; }
	void setStopMaxStdDevXFactor(double stopMaxStdDevXFactor) { _stopMaxStdDevXFactor = stopMaxStdDevXFactor; }
	void setStopMaxTimePerEigenDecomposition(double stopMaxTimePerEigendecomposition) { _stopMaxTimePerEigendecomposition = stopMaxTimePerEigendecomposition; }
	void setStopMinFitness(double stopMinFitness) { _stopMinFitness = stopMinFitness; }
	void setMu(size_t mu, std::string muType = "Logarithmic") { _mu = mu; _muType = muType; }
	void setMuCovariance(double muCovariance) { _muCovariance = muCovariance;}
	void setSigmaCumulationFactor(double sigmaCumulationFactor) { _sigmaCumulationFactor = sigmaCumulationFactor;}
	void setDampingFactor(double dampFactor) { _dampFactor = dampFactor; }
	void setCumulativeCovariance(double cumulativeCovariance) { _cumulativeCovariance = cumulativeCovariance; }
	void setCovarianceMatrixLearningRate(double covarianceMatrixLearningRate) { _covarianceMatrixLearningRate = covarianceMatrixLearningRate; }

	CMAESParameter* operator[](int parameterId) { return _CMAESParameters[parameterId]; }
	CMAESParameter* operator[](std::string parameterName) { return _CMAESParameterMap[parameterName]; }

  private:

	// Korali Runtime Variables
  double* _fitnessVector;
  double* _samplePopulation;
  bool* _initializedSample;
  char _terminationReason[500];
  std::vector<CMAESParameter*> _CMAESParameters;
  std::map<std::string, CMAESParameter*> _CMAESParameterMap;

  size_t _finishedSamples;
  size_t _mu;
  std::string _muType;
  double* _muWeights;
  double _muEffective;
  double _muCovariance;

  double _sigmaCumulationFactor;
  double _dampFactor;
  double _cumulativeCovariance;
  double _covarianceMatrixLearningRate;

  size_t _diagonalCovarianceMatrixEvalFrequency;
  size_t _covarianceEigensystemEvaluationFrequency;

  // Stop conditions
  size_t _maxFitnessEvaluations;   // Defines maximum number of fitness evaluations
  double _stopFitnessEvalThreshold; // Defines minimum function value below which it stops
  double _stopFitnessDiffThreshold; // Defines minimum function value differences before stopping
  double _stopFitnessDiffHistoryThreshold; // Defines minimum function value differences among best values before stopping
  double _stopMinDeltaX; // Defines minimum delta of input parameters among generations before it stops.
  double _stopMaxStdDevXFactor; // Defines maximum standard deviation before it stops.
  double _stopMaxTimePerEigendecomposition; // Defines maximum time to be spent on eigensystem decomposition
  double _stopMinFitness; // Defines the minimum fitness allowed, otherwise it stops

  // Private CMAES-Specific Variables
  double sigma;  /* step size */
  Parameter::Gaussian* _gaussianGenerator;

  double *rgxmean;  /* mean x vector, "parent" */
  double *rgxbestever;
  double *curBest;
  int *index;       /* sorting index of sample pop. */
  double *arFuncValueHist;

  double chiN;
  double **C;  /* lower triangular matrix: i>=j for C[i][j] */
  double **B;  /* matrix with normalize eigenvectors in columns */
  double *rgD; /* axis lengths */

  double *rgpc;
  double *rgps;
  double *rgxold;
  double *rgout;
  double *rgBDz;   /* for B*D*z */
  double *rgdTmp;  /* temporary (random) vector used in different places */
  double *rgFuncValue;

  size_t gen; /* Generation number */
  double countevals;
  double state; /* 1 == sampled, 2 == not in use anymore, 3 == updated */

  double maxdiagC; /* repeatedly used for output */
  double mindiagC;
  double maxEW;
  double minEW;

  short flgEigensysIsUptodate;
  double genOfEigensysUpdate;
  double dMaxSignifKond;

  // Private CMAES-Specific Methods
  void reSampleSingle(int iindex);
  void adaptC2(int hsig);
  double function_value_difference();
  void updateEigensystem(int flgforce);
  void eigen( int N,  double **C, double *diag, double **Q);
  int maxIdx(const double *rgd, int len);
  int minIdx(const double *rgd, int len);
  void sorted_index(const double *rgFunVal, int *iindex, int n);
  bool isFeasible(double *pop);
  double doubleRangeMax(const double *rgd, int len);
  double doubleRangeMin(const double *rgd, int len );
};

} // namespace Korali

#endif // _KORALI_CMAES_H_
