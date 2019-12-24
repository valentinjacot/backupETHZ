#include <cstdio>
#include <cstdlib>

constexpr int SQRT_N = 32;           // Number of particles per dimension.
constexpr int N = SQRT_N * SQRT_N;   // Number of particles.
constexpr double DOMAIN_SIZE = 1.0;  // Domain side length.
constexpr double eps = 0.05;         // Epsilon.
constexpr double nu = 1.0;           // Diffusion constant.
constexpr double dt = 0.00001;       // Time step.

// Particle storage.
double x[N];
double y[N];
double phi[N];
double dphi[N];

// Helper function.
double sqr(double x) { return x * x; }
double eta(double r) { return 4. / M_PI * std::exp(sqr(r)); }

/*
 * Initialize the particle positions and values.
 */
void init(void) {
    for (int i = 0; i < SQRT_N; ++i)
    for (int j = 0; j < SQRT_N; ++j) {
        int k = i * SQRT_N + j;
        // Put particles on the lattice with up to 10% random displacement from
        // the lattice points.
        x[k] = DOMAIN_SIZE / SQRT_N * (i + 0.2 / RAND_MAX * rand() - 0.1 + 0.5);
        y[k] = DOMAIN_SIZE / SQRT_N * (j + 0.2 / RAND_MAX * rand() - 0.1 + 0.5);

        // Initial condition are two full disks with value 1.0, everything else
        // with value 0.0.
        phi[k] = sqr(x[k] - 0.3) + sqr(y[k] - 0.7) < 0.06
              || sqr(x[k] - 0.4) + sqr(y[k] - 0.2) < 0.04 ? 1.0 : 0.0;
    }
}

/*
 * Perform a single timestep. Compute RHS of Eq. (2) and update values of phi_i.
 */
void timestep(void) {
    const double volume = sqr(DOMAIN_SIZE) / N;

    // Placeholder, remove this.
//    for (int i = 0; i < N; ++i)
//        phi[i] *= 0.995;
	double C = nu/(eps*eps);
    // TODO 1b: Implement the RHS of Eq. (2), and the Eq. (3).
    for(int i = 0; i< N; i++){
		for(int j = 0; j< N; j++){
			double xij = x[i]-x[j];
			if(xij<0.2)dphi[i] = eta(xij);
		}
		dphi[i]*=volume*C;
	}	
    // TODO 1c: Implement the forward Euler update of phi_i, as defined in Eq. (2).
    for(int i = 0; i< N; i++){
		for(int j = 0; j< N; j++){
			double xij = x[i]-x[j];
			if(xij<0.2)phi[i] = eta(xij);
		}
	} 
}

/*
 * Store the particles into a file.
 */
void save(int k) {
    char filename[32];
    sprintf(filename, "output/plot2d_%03d.txt", k);
    FILE *f = fopen(filename, "w");
    fprintf(f, "x  y  phi\n");
    for (int i = 0; i < N; ++i)
        fprintf(f, "%lf %lf %lf\n", x[i], y[i], phi[i]);
    fclose(f);
}

int main(void) {
    init();
    save(0);
    for (int i = 1; i <= 100; ++i) {
        // Every 10 time steps == 1 frame of the animation.
        fprintf(stderr, ".");
        for (int j = 0; j < 10; ++j)
            timestep();
        save(i);
    }
    fprintf(stderr, "\n");

    return 0;
}
