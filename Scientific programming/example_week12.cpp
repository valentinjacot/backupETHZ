#include <omp.h> 	
#pragma omp parallel for
/// !!! watch out for race condition
#pragma omp critical
/// solves race condtion, but slower
/// look for optional clauses (if, private,shared, blabla, num_threads)

