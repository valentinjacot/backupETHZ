#include <cstdio>
#include <chrono>

void measure_flops(int N, int K) {
    // TODO: Question 2b: Allocate the buffer of N * K elements.
    volatile double *buffer = new double[N * K];
    for (int i = 0; i < N * K; ++i)
        buffer[i] = 0.999 + 1e-10 * i;

    // TODO: Question 2b: Repeat `repeat` times a traversal of arrays and
    //                    measure total execution time.
    int repeat = 500 / K;
    auto start = std::chrono::steady_clock::now();
    for (int r = 0; r < repeat; ++r)
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < K; ++j)
                buffer[j * N + i] += 1.0123;
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = end - start;

    // TODO: Question 2b: Deallocate.
    delete []buffer;

    // Report.
    double time = diff.count();  /* TODO: Question 2b: time in seconds */
    double flops = (double)repeat * N * K / time;
    printf("%d  %2d  %.4lf\n", N, K, flops * 1e-9);
    fflush(stdout);
}

void run(int N) {
    printf("      N   K  GFLOPS\n");
    for (int K = 1; K <= 40; ++K)
        measure_flops(N, K);
    printf("\n\n");
}

int main() {
    // Array size. Must be a multiple of a large power of two.
    const int N = 1 << 20;

    // Power of two size --> bad.
    run(N);

    // Non-power-of-two size --> better.
    // TODO: Enable for Question 2c:
    run(N + 64 / sizeof(double));

    // NOTE (2018-10-09): If running on Euler II nodes, try larger paddings such as 2, 4, 8, 16 cache lines.

    return 0;
}
