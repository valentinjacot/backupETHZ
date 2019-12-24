void initializeSampler(size_t nSamples, size_t nParameters);
void checkResults();
double evaluateSample(double* parameters);
void getSample(size_t sampleId, double* sample);
void updateEvaluation(size_t sampleId, double eval);
